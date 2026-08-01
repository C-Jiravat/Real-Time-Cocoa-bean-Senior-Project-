from io import BytesIO
import zipfile

import pytest

from cocoa_platform.benchmark.service import (
    BenchmarkInputError,
    Sample,
    evaluate_samples,
    read_yolo_labels,
    samples_from_zip,
)


def test_yolo_label_conversion() -> None:
    assert read_yolo_labels(b"0 0.5 0.5 0.2 0.4\n", 100, 50, "color") == [
        {"bbox": [40.0, 15.0, 60.0, 35.0], "class_key": "purple"}
    ]


def test_unknown_color_class_is_rejected() -> None:
    with pytest.raises(BenchmarkInputError, match="class 0 ถึง 1"):
        read_yolo_labels(b"2 0.5 0.5 0.2 0.2\n", 100, 100, "color")


def test_zip_keeps_valid_pairs_and_reports_missing_label() -> None:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("images/a.jpg", b"image")
        archive.writestr("labels/a.txt", b"0 0.5 0.5 0.2 0.2")
        archive.writestr("images/missing.png", b"image")
    samples, errors = samples_from_zip(buffer.getvalue(), "color")
    assert [sample.name for sample in samples] == ["a"]
    assert errors == [
        {"file": "images/missing.png", "reason": "ไม่พบ label ที่ชื่อเดียวกันสำหรับประเภทที่เลือก"}
    ]


class FakePipeline:
    def __init__(self) -> None:
        self.overlay_calls: list[tuple] = []

    def analyze(self, *_args, **_kwargs):
        return {
            "image": {"width": 100, "height": 100},
            "detections": [
                {
                    "bbox": [40, 40, 60, 60],
                    "detector_confidence": 0.9,
                    "color": {"key": "purple", "confidence": 0.91},
                    "defect": {"key": "normal", "confidence": 0.87},
                }
            ],
            "crop_success": {"successful": 1},
            "quality": {"status": "evaluated"},
            "runtime": {"device_actual": "cpu", "execution_provider": "CPUExecutionProvider"},
            "timing": {"total_ms": 10.0, "yolo_ms": 2.0, "crop_ms": 1.0, "convnext_ms": 7.0},
        }

    def render_detection_overlay(self, *args):
        self.overlay_calls.append(args)
        return "preview"


def test_color_confusion_matrix_reports_classifier_result() -> None:
    pipeline = FakePipeline()
    result = evaluate_samples(
        pipeline,
        [Sample("sample", b"image", color_label=b"0 0.5 0.5 0.2 0.2")],
        0.25,
        0.5,
        "cpu",
        "color",
    )
    assert result["classification"]["color"]["rows"] == [
        {"actual": "เมล็ดสีม่วง", "values": [1, 0, 0]},
        {"actual": "เมล็ดสีน้ำตาล", "values": [0, 0, 0]},
        {"actual": "ไม่สามารถจำแนกได้", "values": [0, 0, 0]},
    ]
    assert result["preview"]["predicted_base64"] == "preview"
    assert pipeline.overlay_calls[0][3] == ["Purple 91%"]
    assert pipeline.overlay_calls[0][4] == [(190, 60, 186)]
    assert pipeline.overlay_calls[1][3] == ["GT: Purple"]


def test_zip_both_uses_separate_color_and_defect_labels() -> None:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("images/a.jpg", b"image")
        archive.writestr("labels/color/a.txt", b"0 0.5 0.5 0.2 0.2")
        archive.writestr("labels/defect/a.txt", b"3 0.5 0.5 0.2 0.2")
    samples, errors = samples_from_zip(buffer.getvalue(), "both")
    assert samples[0].color_label and samples[0].defect_label
    assert errors == []
