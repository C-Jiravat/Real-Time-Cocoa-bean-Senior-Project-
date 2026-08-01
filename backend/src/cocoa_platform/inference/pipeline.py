"""Lazy, batch-oriented YOLO -> crop -> two ConvNeXt ONNX inference pipeline."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import os
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Literal
from uuid import uuid4

import numpy as np
from PIL import Image

from cocoa_platform.config.settings import load_registry, resolve_model_path
from cocoa_platform.grading import calculate_quality


COLOR_THAI = {"purple": "เมล็ดสีม่วง", "brown": "เมล็ดสีน้ำตาล"}
DEFECT_THAI = {
    "normal": "เมล็ดปกติ",
    "germinate": "เมล็ดงอก",
    "slaty_hard_as_rock": "เมล็ดสีเทาหินชนวน",
    "moldy": "เมล็ดขึ้นรา",
}


class InferenceUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class RuntimeInfo:
    device_requested: str
    device_actual: str
    execution_provider: str


class InferencePipeline:
    """One process-local pipeline; inference is serialized to protect local GPU/CPU resources."""

    def __init__(self, registry: dict | None = None) -> None:
        self.registry = registry or load_registry()
        self._detector = None
        self._color_session = None
        self._defect_session = None
        self._runtime: RuntimeInfo | None = None
        self._fingerprints: dict[str, str] | None = None
        self._lock = Lock()

    def health(self) -> dict:
        paths = {
            role: str(resolve_model_path(spec["path"]))
            for role, spec in self.registry.items()
            if role in {"detector", "color_classifier", "defect_classifier"}
        }
        return {
            "ready": self._runtime is not None,
            "model_paths_present": {key: Path(value).is_file() for key, value in paths.items()},
        }

    def _load(self, requested_device: Literal["auto", "cpu", "gpu"]) -> RuntimeInfo:
        if self._runtime and (
            requested_device == "auto" or requested_device == self._runtime.device_actual
        ):
            return self._runtime
        try:
            ultralytics_config_dir = resolve_model_path(".local/ultralytics")
            ultralytics_config_dir.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("YOLO_CONFIG_DIR", str(ultralytics_config_dir))
            import onnxruntime as ort
            from ultralytics import YOLO
        except ImportError as error:
            raise InferenceUnavailable(
                "ยังไม่ได้ติดตั้ง ML runtime: ให้ติดตั้ง backend/requirements-ml.txt"
            ) from error

        available = ort.get_available_providers()
        gpu_available = "CUDAExecutionProvider" in available
        if requested_device == "gpu" and not gpu_available:
            raise InferenceUnavailable("ไม่พบ CUDAExecutionProvider สำหรับคำขอ GPU")
        actual_device = (
            "gpu"
            if requested_device == "gpu" or (requested_device == "auto" and gpu_available)
            else "cpu"
        )
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if actual_device == "gpu"
            else ["CPUExecutionProvider"]
        )
        paths = {
            role: resolve_model_path(spec["path"])
            for role, spec in self.registry.items()
            if isinstance(spec, dict) and "path" in spec
        }
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise InferenceUnavailable("ไม่พบไฟล์ model ตาม registry: " + ", ".join(missing))

        self._detector = YOLO(str(paths["detector"]), task="detect")
        self._color_session = ort.InferenceSession(
            str(paths["color_classifier"]), providers=providers
        )
        self._defect_session = ort.InferenceSession(
            str(paths["defect_classifier"]), providers=providers
        )
        self._runtime = RuntimeInfo(
            device_requested=requested_device,
            device_actual=actual_device,
            execution_provider="CUDAExecutionProvider"
            if actual_device == "gpu"
            else "CPUExecutionProvider",
        )
        self._fingerprints = {
            role: sha256(path.read_bytes()).hexdigest() for role, path in paths.items()
        }
        return self._runtime

    @staticmethod
    def decode_image(raw: bytes) -> np.ndarray:
        try:
            with Image.open(BytesIO(raw)) as source:
                image = source.convert("RGB")
                return np.asarray(image)[:, :, ::-1].copy()  # OpenCV BGR
        except (OSError, ValueError) as error:
            raise ValueError("ไฟล์ภาพไม่ถูกต้อง หรือไม่ใช่ PNG/JPEG/WebP") from error

    @staticmethod
    def _preprocess(crops: list[np.ndarray], input_size: int) -> np.ndarray:
        import cv2

        normalized: list[np.ndarray] = []
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        for crop in crops:
            rgb = cv2.cvtColor(cv2.resize(crop, (input_size, input_size)), cv2.COLOR_BGR2RGB)
            tensor = (rgb.astype(np.float32) / 255.0 - mean) / std
            normalized.append(np.transpose(tensor, (2, 0, 1)))
        return np.asarray(normalized, dtype=np.float32)

    @staticmethod
    def _softmax(values: np.ndarray) -> np.ndarray:
        shifted = values - np.max(values, axis=1, keepdims=True)
        exponentiated = np.exp(shifted)
        return exponentiated / np.sum(exponentiated, axis=1, keepdims=True)

    def _predict_classifier(self, session, crops: list[np.ndarray], spec: dict) -> list[dict]:
        if not crops:
            return []
        input_name = session.get_inputs()[0].name
        batch_size = int(self.registry["classifier_batch_size"])
        result: list[dict] = []
        class_names = {int(key): value for key, value in spec["class_names"].items()}
        for offset in range(0, len(crops), batch_size):
            batch = self._preprocess(crops[offset : offset + batch_size], int(spec["input_size"]))
            logits = np.asarray(session.run(None, {input_name: batch})[0])
            probabilities = self._softmax(logits)
            for row in probabilities:
                class_id = int(np.argmax(row))
                result.append(
                    {
                        "id": class_id,
                        "key": class_names[class_id],
                        "confidence": float(row[class_id]),
                    }
                )
        return result

    def analyze(
        self,
        raw_image: bytes,
        confidence: float,
        iou: float,
        device: Literal["auto", "cpu", "gpu"] = "auto",
        *,
        include_annotations: bool = True,
    ) -> dict:
        if not 0 <= confidence <= 1 or not 0 <= iou <= 1:
            raise ValueError("confidence และ IoU ต้องอยู่ระหว่าง 0 ถึง 1")
        total_started = perf_counter()
        image = self.decode_image(raw_image)
        with self._lock:
            load_started = perf_counter()
            runtime = self._load(device)
            model_load_ms = (perf_counter() - load_started) * 1000
            detector_started = perf_counter()
            result = self._detector.predict(
                source=image,
                conf=confidence,
                iou=iou,
                verbose=False,
                device=0 if runtime.device_actual == "gpu" else "cpu",
            )[0]
            detector_ms = (perf_counter() - detector_started) * 1000
            raw_boxes = result.boxes
            detections: list[dict] = []
            crops: list[np.ndarray] = []
            crop_indexes: list[int] = []
            height, width = image.shape[:2]
            crop_started = perf_counter()
            for box in raw_boxes:
                x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
                detection = {
                    "id": str(uuid4()),
                    "bbox": [x1, y1, x2, y2],
                    "detector_confidence": float(box.conf.item()),
                    "valid_crop": False,
                    "color": None,
                    "defect": None,
                }
                left, top = max(0, int(x1)), max(0, int(y1))
                right, bottom = min(width, int(x2)), min(height, int(y2))
                crop = image[top:bottom, left:right]
                if crop.size:
                    crop_indexes.append(len(detections))
                    crops.append(crop)
                else:
                    detection["failure_reason"] = "bounding box อยู่นอกขอบเขตภาพ"
                detections.append(detection)
            crop_ms = (perf_counter() - crop_started) * 1000

            color_started = perf_counter()
            colors = self._predict_classifier(
                self._color_session, crops, self.registry["color_classifier"]
            )
            color_classifier_ms = (perf_counter() - color_started) * 1000
            defect_started = perf_counter()
            defects = self._predict_classifier(
                self._defect_session, crops, self.registry["defect_classifier"]
            )
            defect_classifier_ms = (perf_counter() - defect_started) * 1000
            for index, color, defect in zip(crop_indexes, colors, defects, strict=True):
                detections[index].update({"valid_crop": True, "color": color, "defect": defect})

            quality = calculate_quality(detections, len(detections))
            render_started = perf_counter()
            annotations = self._render_split(image, detections) if include_annotations else {}
            render_ms = (perf_counter() - render_started) * 1000
            summary = self._summary(detections)
            total_ms = (perf_counter() - total_started) * 1000
            return {
                "image": {"width": width, "height": height, **annotations},
                "detections": detections,
                "summary": summary,
                "quality": quality,
                "crop_success": {
                    "successful": len(crops),
                    "detected_total": len(detections),
                    "percent": _safe_percent(len(crops), len(detections)),
                },
                "runtime": runtime.__dict__,
                "model_fingerprint": self._fingerprint(),
                "timing": {
                    "total_ms": total_ms,
                    "model_load_ms": model_load_ms,
                    "yolo_ms": detector_ms,
                    "crop_ms": crop_ms,
                    "color_classifier_ms": color_classifier_ms,
                    "defect_classifier_ms": defect_classifier_ms,
                    "convnext_ms": color_classifier_ms + defect_classifier_ms,
                    "render_ms": render_ms,
                },
            }

    def _summary(self, detections: list[dict]) -> dict:
        result = {
            "color": {key: 0 for key in COLOR_THAI},
            "defect": {key: 0 for key in DEFECT_THAI},
        }
        for bean in detections:
            if bean["valid_crop"]:
                result["color"][bean["color"]["key"]] += 1
                result["defect"][bean["defect"]["key"]] += 1
        return result

    def _render_split(self, image: np.ndarray, detections: list[dict]) -> dict:
        import cv2

        color_image = image.copy()
        defect_image = image.copy()
        color_palette = {"purple": (190, 60, 186), "brown": (64, 140, 228)}
        defect_palette = {
            "normal": (75, 210, 80),
            "germinate": (40, 190, 245),
            "slaty_hard_as_rock": (210, 190, 65),
            "moldy": (55, 55, 235),
        }
        for bean in detections:
            x1, y1, x2, y2 = (int(value) for value in bean["bbox"])
            if bean["valid_crop"]:
                color_key = bean["color"]["key"]
                defect_key = bean["defect"]["key"]
                self._draw_box(
                    color_image,
                    (x1, y1, x2, y2),
                    color_palette[color_key],
                    f"{color_key.title()} {bean['color']['confidence']:.0%}",
                )
                self._draw_box(
                    defect_image,
                    (x1, y1, x2, y2),
                    defect_palette[defect_key],
                    f"{defect_key.replace('_', ' ').title()} {bean['defect']['confidence']:.0%}",
                )
            else:
                self._draw_box(color_image, (x1, y1, x2, y2), (80, 80, 220), "Crop failed")
                self._draw_box(defect_image, (x1, y1, x2, y2), (80, 80, 220), "Crop failed")
        ok_color, encoded_color = cv2.imencode(".png", color_image)
        ok_defect, encoded_defect = cv2.imencode(".png", defect_image)
        if not ok_color or not ok_defect:
            raise InferenceUnavailable("ไม่สามารถสร้างภาพผลลัพธ์ได้")
        return {
            "color_annotated_base64": base64.b64encode(encoded_color.tobytes()).decode("ascii"),
            "defect_annotated_base64": base64.b64encode(encoded_defect.tobytes()).decode("ascii"),
        }

    @staticmethod
    def _draw_box(
        image: np.ndarray, box: tuple[int, int, int, int], color: tuple[int, int, int], label: str
    ) -> None:
        import cv2

        x1, y1, x2, y2 = box
        longest_edge = max(image.shape[:2])
        font_scale = max(0.72, min(1.35, longest_edge / 1100))
        line_width = max(2, round(font_scale * 2.2))
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, line_width
        )
        label_top = max(0, y1 - text_height - baseline - 14)
        label_bottom = max(text_height + baseline + 14, y1)

        cv2.rectangle(image, (x1, y1), (x2, y2), color, line_width)
        cv2.rectangle(image, (x1, label_top), (x1 + text_width + 14, label_bottom), color, -1)
        cv2.putText(
            image,
            label,
            (x1 + 7, label_bottom - baseline - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            line_width,
            cv2.LINE_AA,
        )

    @staticmethod
    def render_detection_overlay(
        raw_image: bytes,
        boxes: list[list[float]],
        color: tuple[int, int, int],
        labels: list[str] | None = None,
        colors: list[tuple[int, int, int]] | None = None,
    ) -> str:
        """Render benchmark previews without exposing a server-side image path."""
        import cv2

        image = InferencePipeline.decode_image(raw_image)
        if labels and len(labels) != len(boxes):
            raise ValueError("จำนวน label ต้องเท่ากับจำนวน Bounding Box")
        if colors and len(colors) != len(boxes):
            raise ValueError("จำนวนสีต้องเท่ากับจำนวน Bounding Box")
        line_width = max(2, round(max(image.shape[:2]) / 600))
        for index, box in enumerate(boxes):
            x1, y1, x2, y2 = (int(value) for value in box)
            box_color = colors[index] if colors else color
            if labels:
                InferencePipeline._draw_box(image, (x1, y1, x2, y2), box_color, labels[index])
            else:
                cv2.rectangle(image, (x1, y1), (x2, y2), box_color, line_width)
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise InferenceUnavailable("ไม่สามารถสร้างภาพเปรียบเทียบ Benchmark ได้")
        return base64.b64encode(encoded.tobytes()).decode("ascii")

    def _fingerprint(self) -> dict:
        return self._fingerprints or {}


def _safe_percent(numerator: int, denominator: int) -> float:
    return (numerator / denominator) * 100 if denominator else 0.0
