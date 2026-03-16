#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""get_keyword.py

이 PC(웹/STT 허브)에서 실행되는 웨이크업+STT 노드.
- 기본: 웨이크업 워드 감지 후 음성 STT 수행
- '결제/계산' 의도가 있으면 /payment_status 에 "payment start" 퍼블리시
- 웹이 멤버십/결제 팝업을 띄우면 /voice_prompt_state 를 받고,
  이때는 터미널 STT가 직접 "예/아니오"를 인식해서 /voice_yesno 로 퍼블리시

즉, 웹 브라우저 마이크 없이도 우분투 STT 터미널만으로 팝업 진행 가능.
"""

import os
import time

import rclpy
import pyaudio
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import String

from voice_processing.MicController import MicController, MicConfig
from voice_processing.wakeup_word import WakeupWord
from voice_processing.stt import STT


def _has_payment_intent(text: str) -> bool:
    if not text:
        return False
    t = text.strip().lower()

    if any(k in t for k in ["취소", "아니", "안해", "그만", "스톱", "stop"]):
        return False

    payment_keywords = [
        "결제", "결재", "계산", "카드", "pay", "payment",
        "포인트", "멤버십", "회원", "영수증",
    ]
    return any(k in t for k in payment_keywords)


def _parse_yesno(text: str):
    if not text:
        return None
    t = text.strip().lower()

    yes_keywords = ["예", "네", "응", "맞아", "yes", "yep", "yeah", "확인"]
    no_keywords = ["아니", "아니요", "노", "싫어", "취소", "no", "nope"]

    if any(k in t for k in yes_keywords):
        return "yes"
    if any(k in t for k in no_keywords):
        return "no"
    return None


class GetKeyword(Node):
    def __init__(self):
        super().__init__("get_keyword_node")

        openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not openai_api_key:
            self.get_logger().warn("OPENAI_API_KEY 환경변수가 비어있습니다. STT가 실패할 수 있어요.")

        self.stt = STT(openai_api_key=openai_api_key)

        # 퍼블리셔
        self.payment_pub = self.create_publisher(String, "/payment_status", 10)
        self.voice_yesno_pub = self.create_publisher(String, "/voice_yesno", 10)

        # 웹에서 현재 팝업 단계 알려줌 (idle / membership / payment / bin)
        prompt_qos = QoSProfile(depth=1)
        prompt_qos.reliability = ReliabilityPolicy.RELIABLE
        prompt_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.voice_prompt_sub = self.create_subscription(
            String,
            "/voice_prompt_state",
            self._on_voice_prompt_state,
            prompt_qos,
        )
        self.current_prompt_state = "idle"

        # 오디오 장치 설정
        mic_config = MicConfig(
            chunk=12000,
            rate=48000,
            channels=1,
            record_seconds=5,
            fmt=pyaudio.paInt16,
            device_index=10,
            buffer_size=24000,
        )
        self.mic_controller = MicController(config=mic_config)
        self.wakeup_word = WakeupWord(mic_config.buffer_size)

        # 중복 퍼블리시 방지(쿨다운)
        self.cooldown_sec = 2.0
        self._last_publish_ts = 0.0
        self._last_yesno_ts = 0.0
        self.prompt_stt_cooldown_sec = 1.0

        self.get_logger().info("✅ get_keyword_node: 웨이크업+STT 감시 시작")

    def _on_voice_prompt_state(self, msg: String):
        new_state = (msg.data or "idle").strip().lower()
        if new_state != self.current_prompt_state:
            self.current_prompt_state = new_state
            self.get_logger().info(f"🪟 웹 팝업 상태 변경: {self.current_prompt_state}")

    def publish_payment_start(self):
        now = time.time()
        if now - self._last_publish_ts < self.cooldown_sec:
            return
        self._last_publish_ts = now

        msg = String()
        msg.data = "payment start"
        self.payment_pub.publish(msg)
        self.get_logger().warn("💳 [PUBLISH] /payment_status: payment start")

    def publish_yesno(self, answer: str):
        now = time.time()
        if now - self._last_yesno_ts < self.prompt_stt_cooldown_sec:
            return
        self._last_yesno_ts = now

        msg = String()
        msg.data = answer
        self.voice_yesno_pub.publish(msg)
        self.get_logger().warn(f"🗣️ [PUBLISH] /voice_yesno: {answer}")

    def _run_prompt_yesno_once(self):
        now = time.time()
        if now - self._last_yesno_ts < self.prompt_stt_cooldown_sec:
            return

        active_stage = self.current_prompt_state
        self.get_logger().info(f"🗣️ {active_stage} 팝업 응답 대기 → 호출어 없이 즉시 STT 진행")
        text = self.stt.speech2text()
        self.get_logger().info(f"응답 인식 결과: {text}")

        # STT가 끝나는 동안 웹에서 팝업이 닫혔으면 현재 응답은 버림
        if self.current_prompt_state != active_stage or self.current_prompt_state == "idle":
            self.get_logger().info("팝업 단계가 이미 변경되어 이번 응답은 무시합니다.")
            self._last_yesno_ts = time.time()
            return

        answer = _parse_yesno(text)
        if answer:
            self.publish_yesno(answer)
        else:
            self.get_logger().warn("예/아니오 판별 실패 (같은 팝업에서 다시 대기)")
            self._last_yesno_ts = time.time()

    def run(self):
        try:
            self.mic_controller.open_stream()
            self.wakeup_word.set_stream(self.mic_controller.stream)
        except OSError:
            self.get_logger().error("마이크를 열 수 없습니다. device_index를 확인하세요.")
            return

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.01)

            # 팝업 응답 단계: 웨이크업 없이 바로 예/아니오 STT
            if self.current_prompt_state in ("membership", "payment", "bin"):
                self._run_prompt_yesno_once()
                time.sleep(0.05)
                continue

            if self.wakeup_word.is_wakeup():
                self.get_logger().info("👂 웨이크업 감지 → STT 진행")
                text = self.stt.speech2text()
                self.get_logger().info(f"인식 결과: {text}")

                if _has_payment_intent(text):
                    self.publish_payment_start()

            time.sleep(0.05)


def main():
    rclpy.init()
    node = GetKeyword()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
