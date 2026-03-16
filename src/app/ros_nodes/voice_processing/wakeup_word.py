import os
from pathlib import Path
import shutil
import inspect

import numpy as np
from scipy.signal import resample
from openwakeword.model import Model


# ==============================
# Wakeword model path resolution
# ==============================
# Priority:
# 1) WAKEWORD_MODEL_PATH env (absolute or relative)
# 2) <this_file_dir>/models/hello_rokey_8332_32.tflite
DEFAULT_MODEL_NAME = "hello_rokey_8332_32.tflite"
THIS_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("WAKEWORD_MODEL_PATH", str(THIS_DIR / "models" / DEFAULT_MODEL_NAME))).expanduser().resolve()


def _ensure_model_available_for_openwakeword(model_path: Path) -> None:
    """Best-effort: place the custom tflite into openwakeword's resources so
    versions that only scan built-in resource folders can find it."""
    try:
        import openwakeword  # noqa
        pkg_dir = Path(openwakeword.__file__).resolve().parent
    except Exception:
        return

    # Common locations openwakeword scans (varies by version)
    candidates = [
        pkg_dir / "resources" / "models" / "tflite",
        pkg_dir / "resources" / "models",
    ]
    for d in candidates:
        try:
            d.mkdir(parents=True, exist_ok=True)
            dst = d / model_path.name
            if not dst.exists():
                shutil.copy2(str(model_path), str(dst))
        except Exception:
            # ignore any permission/path issues; we'll try other paths
            pass


def _build_model(model_path: Path) -> Model:
    """Create openwakeword Model in a way compatible with multiple versions."""
    sig = inspect.signature(Model)
    params = sig.parameters

    # Try explicit custom-model path parameters (if supported by this version)
    custom_keys = [k for k in ["custom_model_paths", "model_paths", "wakeword_model_paths", "models"] if k in params]
    fw_key = "inference_framework" if "inference_framework" in params else ("framework" if "framework" in params else None)

    if custom_keys:
        kwargs = {}
        kwargs[custom_keys[0]] = [str(model_path)]
        if fw_key:
            kwargs[fw_key] = "tflite"
        try:
            return Model(wakeword_models=[], **kwargs)
        except Exception:
            # fall back to resource-folder discovery below
            pass

    # Resource-folder discovery path: copy into openwakeword resources and load by name/stem
    _ensure_model_available_for_openwakeword(model_path)
    model_id = model_path.stem  # 'hello_rokey_8332_32'

    kwargs = {}
    if fw_key:
        kwargs[fw_key] = "tflite"

    # Some versions accept .tflite name, others want stem; try both.
    for candidate in [model_id, model_path.name]:
        try:
            return Model(wakeword_models=[candidate], **kwargs)
        except Exception:
            continue

    # As a final attempt, try passing string path directly (some versions support it)
    try:
        return Model(wakeword_models=[str(model_path)], **kwargs)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load wakeword model.\n"
            f"- MODEL_PATH: {model_path}\n"
            f"- exists: {model_path.exists()}\n"
            f"- Model signature: {sig}\n"
            f"Original error: {e}"
        )


class WakeupWord:
    def __init__(self, buffer_size: int):
        self.model: Model | None = None
        self.model_name: str | None = None
        self.stream = None
        self.buffer_size = int(buffer_size)

    def set_stream(self, stream):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Wakeword model not found: {MODEL_PATH}\n"
                f"Set WAKEWORD_MODEL_PATH or place {DEFAULT_MODEL_NAME} in {THIS_DIR / 'models'}"
            )
        self.model = _build_model(MODEL_PATH)

        # openwakeword output key is usually model stem
        self.model_name = MODEL_PATH.stem
        self.stream = stream

    def is_wakeup(self) -> bool:
        if self.model is None or self.stream is None:
            return False

        audio_chunk = np.frombuffer(
            self.stream.read(self.buffer_size, exception_on_overflow=False),
            dtype=np.int16,
        )

        # MicController uses 48k by default; openwakeword expects 16k
        audio_chunk = resample(audio_chunk, int(len(audio_chunk) * 16000 / 48000))

        outputs = self.model.predict(audio_chunk, threshold=0.1)

        # Some versions return key as stem, some as name; be defensive
        key = self.model_name
        if key not in outputs:
            alt = MODEL_PATH.name
            if alt in outputs:
                key = alt
            else:
                # pick the first key
                key = next(iter(outputs.keys()))

        confidence = float(outputs[key])
        print("confidence:", confidence)

        if confidence > 0.3:
            print("Wakeword detected!")
            return True
        return False
