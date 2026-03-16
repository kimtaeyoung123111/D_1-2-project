from flask import Flask, render_template, jsonify, request
import os
import sqlite3
import threading
import time
from datetime import datetime
from typing import Optional
import tempfile
from openai import OpenAI
import ros_listener

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "sales.db")

# ----------------------------
# In-memory state (web PC only)
# ----------------------------
cart = []  # list[str]
prices = {
    "pringles": 2500,
    "pepsi": 2000,
    "tuna": 3000,
    "cube": 1500,
    "gum": 500,
    "spam": 4200,
}

state = {
    "session_active": False,
    "session_seq": 0,
    "bin_full": False,
    "bin_ready": False,
    "done_trigger": False,
    "waiting_payment_latched": False,
    "prompt_stage": "idle",       # idle | membership | payment | bin
    "voice_action": "",
    "voice_action_seq": 0,
    "last_product_ts": None,
    "last_box_status": "",
    "last_payment_status": "",
    "last_waiting_payment": "",
    "last_voice_yesno": "",
    "bin_full_latched": False,
    "last_stt_text": "",
    "last_stt_result": "idle",
    "last_stt_error": "",
}

_ros_node = None
_openai_client = None
_openai_client_init = False


def _now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _db_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    os.makedirs(BASE_DIR, exist_ok=True)
    conn = _db_conn()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                tx_id TEXT PRIMARY KEY,
                total INTEGER NOT NULL,
                paid_total INTEGER NOT NULL,
                discount_amount INTEGER NOT NULL DEFAULT 0,
                discount_type TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS transaction_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tx_id TEXT NOT NULL,
                name TEXT NOT NULL,
                price INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (tx_id) REFERENCES transactions(tx_id)
            )
        """)
        conn.commit()
    finally:
        conn.close()


def _set_stt_debug(text: str = "", result: str = "", error: str = "") -> None:
    if text is not None:
        state["last_stt_text"] = text
    if result is not None and result != "":
        state["last_stt_result"] = result
    if error is not None:
        state["last_stt_error"] = error


def _set_prompt_stage(stage: str) -> None:
    stage = (stage or "idle").strip().lower()
    state["prompt_stage"] = stage
    state["last_stt_result"] = stage
    if _ros_node is not None:
        try:
            _ros_node.publish_voice_prompt(stage)
        except Exception:
            pass


def _push_voice_action(action: str) -> None:
    state["voice_action"] = action
    state["voice_action_seq"] += 1


def _start_new_session() -> None:
    global cart
    cart = []
    state["session_active"] = True
    state["session_seq"] += 1
    state["done_trigger"] = False
    state["waiting_payment_latched"] = False
    state["bin_full"] = False
    state["bin_ready"] = False
    state["bin_full_latched"] = False
    state["last_product_ts"] = time.time()
    _set_prompt_stage("idle")


def _finish_session() -> None:
    state["session_active"] = False
    state["done_trigger"] = False
    state["waiting_payment_latched"] = False
    state["bin_full_latched"] = False
    _set_prompt_stage("idle")


def _handle_waiting_for_payment_once() -> None:
    if state["waiting_payment_latched"]:
        return
    state["waiting_payment_latched"] = True
    state["done_trigger"] = True
    _set_prompt_stage("membership")


def _apply_modal_membership(answer: str) -> None:
    ans = (answer or "").strip().lower()
    _push_voice_action("membership_yes" if ans == "yes" else "membership_no")
    _set_stt_debug(ans, "membership", "")
    _set_prompt_stage("payment")


def _apply_modal_payment(answer: str) -> None:
    ans = (answer or "").strip().lower()
    _push_voice_action("payment_yes" if ans == "yes" else "payment_no")
    state["done_trigger"] = False
    _set_stt_debug(ans, "payment", "")
    _set_prompt_stage("idle")


def _dismiss_bin_modal(answer: Optional[str] = None) -> None:
    ans = (answer or "").strip().lower()
    if ans == "confirm":
        ans = "yes"
    if ans not in ("yes",):
        return

    _push_voice_action("bin_confirm")
    _set_stt_debug(ans, "bin", "")

    state["bin_full"] = False
    state["bin_ready"] = True
    state["bin_full_latched"] = False
    state["last_box_status"] = "READY"
    if _ros_node is not None:
        try:
            _ros_node.publish_box_status("READY")
        except Exception:
            pass

    _set_prompt_stage("idle")


def _save_sale(items_snapshot: list[str], membership: bool, total: int, discount_amount: int, paid_total: int) -> str:
    tx_id = datetime.now().strftime("TX%Y%m%d%H%M%S%f")
    ts = _now_iso()
    discount_type = "membership" if membership and discount_amount > 0 else ""

    conn = _db_conn()
    try:
        conn.execute(
            "INSERT INTO transactions (tx_id, total, paid_total, discount_amount, discount_type, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (tx_id, total, paid_total, discount_amount, discount_type, ts),
        )
        conn.executemany(
            "INSERT INTO transaction_items (tx_id, name, price, created_at) VALUES (?, ?, ?, ?)",
            [(tx_id, name, prices.get(name, 0), ts) for name in items_snapshot],
        )
        conn.commit()
    finally:
        conn.close()
    return tx_id


def _load_admin_data():
    init_db()
    conn = _db_conn()
    try:
        summary = [
            (row["name"], row["qty"], row["amount"])
            for row in conn.execute(
                """
                SELECT name, COUNT(*) AS qty, COALESCE(SUM(price), 0) AS amount
                FROM transaction_items
                GROUP BY name
                ORDER BY amount DESC, qty DESC, name ASC
                """
            ).fetchall()
        ]
        tx_rows = [
            (row["tx_id"], row["total"], row["paid_total"], row["discount_amount"], row["discount_type"], row["created_at"])
            for row in conn.execute(
                """
                SELECT tx_id, total, paid_total, discount_amount, discount_type, created_at
                FROM transactions
                ORDER BY created_at DESC
                LIMIT 20
                """
            ).fetchall()
        ]
        item_rows = [
            (row["name"], row["price"], row["created_at"])
            for row in conn.execute(
                """
                SELECT name, price, created_at
                FROM transaction_items
                ORDER BY id DESC
                LIMIT 50
                """
            ).fetchall()
        ]
    finally:
        conn.close()

    db_exists = os.path.exists(DB_PATH)
    db_status = f"{os.path.basename(DB_PATH)} ({'정상' if db_exists else '미생성'})"
    return summary, tx_rows, item_rows, db_status



def _get_openai_client():
    global _openai_client, _openai_client_init
    if _openai_client_init:
        return _openai_client
    _openai_client_init = True
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        _openai_client = OpenAI(api_key=api_key)
    except Exception:
        _openai_client = None
    return _openai_client


def _transcribe_uploaded_audio() -> str:
    audio = request.files.get("audio")
    if audio is None or not getattr(audio, "filename", None):
        return ""

    client = _get_openai_client()
    if client is None:
        _set_stt_debug("(audio received)", state.get("prompt_stage", "idle"), "OPENAI_API_KEY missing")
        return ""

    suffix = os.path.splitext(audio.filename)[1] or ".webm"
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            audio.save(tmp)
            temp_path = tmp.name

        with open(temp_path, "rb") as f:
            transcript = client.audio.transcriptions.create(model="whisper-1", file=f)

        text = str(getattr(transcript, "text", "") or "").strip()
        return text
    except Exception as e:
        _set_stt_debug("(audio received)", state.get("prompt_stage", "idle"), f"transcription_failed: {e}")
        return ""
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


def _parse_yesno_text(text: str) -> str:
    s = (text or "").strip().lower()
    if not s:
        return ""
    yes_tokens = ("예", "네", "응", "맞아", "확인", "yes", "yeah", "yep")
    no_tokens = ("아니", "아니요", "취소", "싫어", "no", "nope")
    if any(tok in s for tok in yes_tokens):
        return "yes"
    if any(tok in s for tok in no_tokens):
        return "no"
    return ""


def _extract_text_from_request() -> str:
    text = ""
    data = request.get_json(silent=True)
    if isinstance(data, dict):
        text = str(data.get("text", "")).strip()
    if not text:
        text = str(request.form.get("text", "")).strip()
    if not text:
        text = str(request.args.get("text", "")).strip()
    return text




def _prompt_state_heartbeat_loop() -> None:
    while True:
        try:
            if _ros_node is not None:
                # 팝업이 떠 있는 동안에는 STT 터미널이 타이밍을 놓쳐도 다시 받도록 주기적으로 재발행
                stage = state.get("prompt_stage", "idle")
                if stage in ("membership", "payment", "bin"):
                    _ros_node.publish_voice_prompt(stage)
            time.sleep(0.35)
        except Exception:
            time.sleep(0.35)

def _drain_queues_loop() -> None:
    global cart
    while True:
        try:
            while True:
                name = ros_listener.product_queue.get_nowait()
                cart.append(name)
                state["last_product_ts"] = time.time()
        except Exception:
            pass

        try:
            while True:
                bs = ros_listener.box_status_queue.get_nowait()
                state["last_box_status"] = bs
                upper_bs = bs.strip().upper()
                if upper_bs == "FULL":
                    if not state["bin_full_latched"]:
                        state["bin_full"] = True
                        state["bin_ready"] = False
                        state["bin_full_latched"] = True
                        _set_prompt_stage("bin")
                elif upper_bs == "READY":
                    state["bin_full"] = False
                    state["bin_ready"] = True
                    state["bin_full_latched"] = False
                    if state["prompt_stage"] == "bin":
                        _set_prompt_stage("idle")
        except Exception:
            pass

        try:
            while True:
                ps = ros_listener.payment_status_queue.get_nowait()
                state["last_payment_status"] = ps
                if ps.strip().lower() == "payment start":
                    _start_new_session()
        except Exception:
            pass

        try:
            while True:
                wp = ros_listener.waiting_payment_queue.get_nowait()
                state["last_waiting_payment"] = wp
                if wp.strip().lower() == "ready_for_payment":
                    _handle_waiting_for_payment_once()
        except Exception:
            pass

        try:
            while True:
                ans = ros_listener.voice_yesno_queue.get_nowait()
                state["last_voice_yesno"] = ans
                _set_stt_debug(ans, state["prompt_stage"], "")
                if ans not in ("yes", "no"):
                    continue
                if state["prompt_stage"] == "membership":
                    _apply_modal_membership(ans)
                elif state["prompt_stage"] == "payment":
                    _apply_modal_payment(ans)
                elif state["prompt_stage"] == "bin":
                    if ans == "yes":
                        _dismiss_bin_modal("confirm")
                    else:
                        _set_stt_debug(ans, "bin_waiting_confirm", "")
        except Exception:
            pass

        time.sleep(0.05)


@app.route("/")
def root():
    return render_template("pos.html")


@app.route("/pos")
def pos():
    return render_template("pos.html")


@app.route("/admin")
def admin():
    summary, tx_rows, item_rows, db_status = _load_admin_data()
    return render_template(
        "admin.html",
        today=datetime.now().strftime("%Y-%m-%d"),
        summary=summary,
        tx_rows=tx_rows,
        item_rows=item_rows,
        db_status=db_status,
    )


@app.post("/api/cart/reset")
def api_cart_reset():
    global cart
    cart = []
    state["done_trigger"] = False
    state["waiting_payment_latched"] = False
    _set_prompt_stage("idle")
    return jsonify({"ok": True})


@app.post("/api/session_start")
def api_session_start():
    _start_new_session()
    return jsonify({"ok": True, "session_seq": state["session_seq"]})


@app.post("/api/session_end")
def api_session_end():
    _finish_session()
    return jsonify({"ok": True})


@app.post("/api/modal/membership")
def api_modal_membership():
    data = request.get_json(silent=True) or {}
    answer = str(data.get("answer", "")).strip().lower()
    if answer not in ("yes", "no"):
        return jsonify({"ok": False, "error": "answer must be yes/no"}), 400
    _apply_modal_membership(answer)
    return jsonify({"ok": True})


@app.post("/api/modal/payment")
def api_modal_payment():
    data = request.get_json(silent=True) or {}
    answer = str(data.get("answer", "")).strip().lower()
    if answer not in ("yes", "no"):
        return jsonify({"ok": False, "error": "answer must be yes/no"}), 400
    _apply_modal_payment(answer)
    return jsonify({"ok": True})


@app.post("/api/modal/bin")
def api_modal_bin():
    data = request.get_json(silent=True) or {}
    answer = str(data.get("answer", "")).strip().lower()
    if answer not in ("yes", "confirm"):
        return jsonify({"ok": False, "error": "answer must be confirm"}), 400
    _dismiss_bin_modal(answer)
    return jsonify({"ok": True})


@app.post("/api/pay")
def api_pay():
    global cart
    if not cart:
        return jsonify({"ok": False, "error": "empty cart"}), 400

    items_snapshot = list(cart)
    total = sum(prices.get(name, 0) for name in items_snapshot)
    data = request.get_json(silent=True) or {}
    membership = bool(data.get("membership", False))
    discount_amount = round(total * 0.10) if membership else 0
    paid_total = max(total - discount_amount, 0)
    tx_id = _save_sale(items_snapshot, membership, total, discount_amount, paid_total)

    cart = []
    _finish_session()

    return jsonify({
        "ok": True,
        "tx_id": tx_id,
        "total": total,
        "discount_amount": discount_amount,
        "paid_total": paid_total,
    })


@app.post("/api/void")
def api_void():
    global cart
    if cart:
        cart.pop()
    return jsonify({"ok": True})


@app.post("/api/clear")
def api_clear():
    global cart
    cart = []
    state["done_trigger"] = False
    state["waiting_payment_latched"] = False
    _set_prompt_stage("idle")
    return jsonify({"ok": True})


@app.post("/api/stt_command")
def api_stt_command():
    text = _extract_text_from_request()
    if not text and "audio" in request.files:
        text = _transcribe_uploaded_audio()

    text = text.strip()
    _set_stt_debug(text or "(no speech)", "command", state.get("last_stt_error", ""))
    lowered = text.lower()
    start = any(token in lowered for token in ("계산", "결제", "checkout", "start"))
    return jsonify({"ok": True, "start": start, "text": text})


@app.post("/api/stt_yesno")
def api_stt_yesno():
    text = _extract_text_from_request()
    if not text and "audio" in request.files:
        text = _transcribe_uploaded_audio()

    answer = _parse_yesno_text(text)
    _set_stt_debug(text or "(no speech)", state["prompt_stage"], state.get("last_stt_error", ""))
    return jsonify({"ok": True, "answer": answer, "text": text})


@app.get("/api/runtime")
def api_runtime():
    return jsonify({
        "ok": True,
        "session_active": state["session_active"],
        "last_wakeword_at": state["last_payment_status"] or "-",
        "last_wakeword_conf": "-",
    })


@app.get("/api/stt_debug")
def api_stt_debug():
    return jsonify({
        "ok": True,
        "last_ts": _now_iso(),
        "last_text": state["last_stt_text"] or state["last_voice_yesno"] or state["last_payment_status"] or "-",
        "last_result": state["last_stt_result"] or state["prompt_stage"],
        "last_error": state["last_stt_error"],
    })


@app.get("/api/poll")
def api_poll():
    items = []
    total = 0
    for i, name in enumerate(cart, start=1):
        price = prices.get(name, 0)
        total += price
        items.append({"no": i, "name": name, "price": price})

    return jsonify({
        "ok": True,
        "items": items,
        "cart": items,
        "total": total,
        "done_trigger": state["done_trigger"],
        "bin_full_trigger": state["bin_full"],
        "box_ready": state["bin_ready"],
        "bin_trigger_removed": False,
        "session_active": state["session_active"],
        "session_seq": state["session_seq"],
        "prompt_stage": state["prompt_stage"],
        "voice_action": state["voice_action"],
        "voice_action_seq": state["voice_action_seq"],
        "meta": {
            "last_product_ts": state["last_product_ts"],
            "last_box_status": state["last_box_status"],
            "last_payment_status": state["last_payment_status"],
            "last_waiting_payment": state["last_waiting_payment"],
            "last_voice_yesno": state["last_voice_yesno"],
            "server_time": _now_iso(),
        }
    })


def main():
    global _ros_node
    init_db()
    _ros_node = ros_listener.start_ros_spin_in_thread()
    _set_prompt_stage("idle")
    threading.Thread(target=_drain_queues_loop, daemon=True).start()
    threading.Thread(target=_prompt_state_heartbeat_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, debug=False)


if __name__ == "__main__":
    main()
