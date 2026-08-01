"""Safe benchmark ingestion for detector boxes and crop-classifier labels."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from io import BytesIO
from pathlib import PurePosixPath
from typing import Literal
import zipfile

import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
BenchmarkTarget = Literal["color", "defect", "both"]
CLASS_KEYS = {
    "color": ("purple", "brown"),
    "defect": ("normal", "germinate", "slaty_hard_as_rock", "moldy"),
}
THAI_CLASS_NAMES = {
    "purple": "เมล็ดสีม่วง",
    "brown": "เมล็ดสีน้ำตาล",
    "normal": "เมล็ดปกติ",
    "germinate": "เมล็ดงอก",
    "slaty_hard_as_rock": "เมล็ดสีเทาหินชนวน",
    "moldy": "เมล็ดขึ้นรา",
    "unclassified": "ไม่สามารถจำแนกได้",
}
BOX_COLORS = {
    "purple": (190, 60, 186),
    "brown": (64, 140, 228),
    "normal": (75, 210, 80),
    "germinate": (40, 190, 245),
    "slaty_hard_as_rock": (210, 190, 65),
    "moldy": (55, 55, 235),
    "unclassified": (120, 120, 120),
}


class BenchmarkInputError(ValueError):
    pass


@dataclass(frozen=True)
class Sample:
    name: str
    image: bytes
    color_label: bytes | None = None
    defect_label: bytes | None = None


def read_yolo_labels(
    raw: bytes, width: int, height: int, target: Literal["color", "defect"]
) -> list[dict]:
    annotations: list[dict] = []
    class_keys = CLASS_KEYS[target]
    for line_number, line in enumerate(raw.decode("utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise BenchmarkInputError(f"label บรรทัด {line_number} ต้องมี 5 ค่าใน YOLO format")
        try:
            class_id = int(parts[0])
            x_center, y_center, box_width, box_height = (float(value) for value in parts[1:])
        except ValueError as error:
            raise BenchmarkInputError(f"label บรรทัด {line_number} มีค่าไม่ถูกต้อง") from error
        if class_id not in range(len(class_keys)):
            raise BenchmarkInputError(
                f"label {target} บรรทัด {line_number} รองรับ class 0 ถึง {len(class_keys) - 1} เท่านั้น"
            )
        if not all(0 <= value <= 1 for value in (x_center, y_center, box_width, box_height)):
            raise BenchmarkInputError(f"label บรรทัด {line_number} ต้องอยู่ระหว่าง 0 ถึง 1")
        x1, y1 = (x_center - box_width / 2) * width, (y_center - box_height / 2) * height
        x2, y2 = (x_center + box_width / 2) * width, (y_center + box_height / 2) * height
        if x2 <= x1 or y2 <= y1:
            raise BenchmarkInputError(f"label บรรทัด {line_number} มี bounding box ว่าง")
        annotations.append({"bbox": [x1, y1, x2, y2], "class_key": class_keys[class_id]})
    return annotations


def samples_from_zip(raw_zip: bytes, target: BenchmarkTarget) -> tuple[list[Sample], list[dict]]:
    errors: list[dict] = []
    try:
        archive = zipfile.ZipFile(BytesIO(raw_zip))
    except zipfile.BadZipFile as error:
        raise BenchmarkInputError("ไฟล์ที่อัปโหลดไม่ใช่ ZIP ที่ถูกต้อง") from error
    with archive:
        files = [info for info in archive.infolist() if not info.is_dir()]
        if not files:
            raise BenchmarkInputError("ZIP ไม่มีไฟล์ข้อมูล")
        if any(
            PurePosixPath(info.filename).is_absolute() or ".." in PurePosixPath(info.filename).parts
            for info in files
        ):
            raise BenchmarkInputError("ZIP มี path ที่ไม่ปลอดภัย")
        images = {
            PurePosixPath(info.filename).stem: info
            for info in files
            if PurePosixPath(info.filename).parts[:1] == ("images",)
            and PurePosixPath(info.filename).suffix.lower() in IMAGE_EXTENSIONS
        }
        if not images:
            raise BenchmarkInputError("ZIP ต้องมี images/ ที่รองรับ PNG, JPEG หรือ WebP")

        def labels_for(kind: Literal["color", "defect"]) -> dict:
            expected_parts = ("labels", kind) if target == "both" else ("labels",)
            return {
                PurePosixPath(info.filename).stem: info
                for info in files
                if PurePosixPath(info.filename).parts[:-1] == expected_parts
                and PurePosixPath(info.filename).suffix.lower() == ".txt"
            }

        color_files = labels_for("color") if target in {"color", "both"} else {}
        defect_files = labels_for("defect") if target in {"defect", "both"} else {}
        samples: list[Sample] = []
        for stem, image_info in images.items():
            color_info, defect_info = color_files.get(stem), defect_files.get(stem)
            missing = (target in {"color", "both"} and not color_info) or (
                target in {"defect", "both"} and not defect_info
            )
            if missing:
                errors.append(
                    {"file": image_info.filename, "reason": "ไม่พบ label ที่ชื่อเดียวกันสำหรับประเภทที่เลือก"}
                )
                continue
            samples.append(
                Sample(
                    stem,
                    archive.read(image_info),
                    archive.read(color_info) if color_info else None,
                    archive.read(defect_info) if defect_info else None,
                )
            )
    if not samples:
        required = "labels/color/ และ labels/defect/" if target == "both" else "labels/"
        raise BenchmarkInputError(f"ไม่มีคู่ภาพและ label ใน {required} ที่พร้อมประเมิน")
    return samples, errors


def _iou(first: list[float], second: list[float]) -> float:
    left, top, right, bottom = (
        max(first[0], second[0]),
        max(first[1], second[1]),
        min(first[2], second[2]),
        min(first[3], second[3]),
    )
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = (
        (first[2] - first[0]) * (first[3] - first[1])
        + (second[2] - second[0]) * (second[3] - second[1])
        - intersection
    )
    return intersection / union if union else 0.0


def _match(
    predictions: list[dict], ground_truth: list[dict], threshold: float
) -> tuple[list[bool], int, list[tuple[int, int]]]:
    matched_gt: set[int] = set()
    outcomes: list[bool] = []
    pairs: list[tuple[int, int]] = []
    for prediction_index, prediction in sorted(
        enumerate(predictions), key=lambda item: item[1]["detector_confidence"], reverse=True
    ):
        index, score = max(
            (
                (index, _iou(prediction["bbox"], gt["bbox"]))
                for index, gt in enumerate(ground_truth)
                if index not in matched_gt
            ),
            key=lambda item: item[1],
            default=(-1, 0.0),
        )
        is_true_positive = score >= threshold
        outcomes.append(is_true_positive)
        if is_true_positive:
            matched_gt.add(index)
            pairs.append((prediction_index, index))
    return outcomes, len(matched_gt), pairs


def _average_precision(records: list[tuple[float, bool]], ground_truth_count: int) -> float:
    if not ground_truth_count:
        return 0.0
    records.sort(reverse=True, key=lambda item: item[0])
    true_positive = np.cumsum([match for _, match in records], dtype=float)
    false_positive = np.cumsum([not match for _, match in records], dtype=float)
    recall = true_positive / ground_truth_count
    precision = true_positive / np.maximum(true_positive + false_positive, 1)
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    recall, precision = (
        np.concatenate(([0.0], recall, [1.0])),
        np.concatenate(([0.0], precision, [0.0])),
    )
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def _classification_result(
    target: Literal["color", "defect"],
    predictions: list[dict],
    annotations: list[dict],
    pairs: list[tuple[int, int]],
) -> dict:
    keys = (*CLASS_KEYS[target], "unclassified")
    counts = [[0 for _ in keys] for _ in keys]
    for prediction_index, gt_index in pairs:
        actual = annotations[gt_index]["class_key"]
        predicted = predictions[prediction_index].get(target) or {"key": "unclassified"}
        counts[keys.index(actual)][keys.index(predicted["key"])] += 1
    correct = sum(counts[index][index] for index in range(len(CLASS_KEYS[target])))
    return {
        "title": f"{target.title()} confusion matrix",
        "labels": [THAI_CLASS_NAMES[key] for key in keys],
        "rows": [
            {"actual": THAI_CLASS_NAMES[key], "values": counts[index]}
            for index, key in enumerate(keys)
        ],
        "evaluated": len(pairs),
        "accuracy": correct / len(pairs) if pairs else 0.0,
    }


def _prediction_label(prediction: dict, target: BenchmarkTarget) -> str:
    def label_for(kind: Literal["color", "defect"]) -> str:
        value = prediction.get(kind)
        if not value:
            return "Crop failed"
        return f"{value['key'].replace('_', ' ').title()} {value['confidence']:.0%}"

    if target == "both":
        return f"{label_for('color')} / {label_for('defect')}"
    return label_for(target)


def _ground_truth_label(
    annotations: dict[str, list[dict]], index: int, target: BenchmarkTarget
) -> str:
    if target == "both":
        return (
            f"GT: {annotations['color'][index]['class_key'].title()} / "
            f"{annotations['defect'][index]['class_key'].replace('_', ' ').title()}"
        )
    return f"GT: {annotations[target][index]['class_key'].replace('_', ' ').title()}"


def _preview_class_key(prediction: dict, target: BenchmarkTarget) -> str:
    """A single outline can have one color; use color class first in combined mode."""
    kind = "color" if target in {"color", "both"} else "defect"
    return (prediction.get(kind) or {"key": "unclassified"})["key"]


def _ground_truth_class_key(
    annotations: dict[str, list[dict]], index: int, target: BenchmarkTarget
) -> str:
    kind = "color" if target in {"color", "both"} else "defect"
    return annotations[kind][index]["class_key"]


def evaluate_samples(
    pipeline,
    samples: list[Sample],
    confidence: float,
    iou_threshold: float,
    device: str,
    target: BenchmarkTarget,
) -> dict:
    per_image: list[dict] = []
    all_predictions: dict[float, list[tuple[float, bool]]] = defaultdict(list)
    total_gt = total_predictions = total_matches = total_crops = 0
    timing = defaultdict(float)
    preview: dict | None = None
    runtime: dict | None = None
    classification_pairs = {"color": [], "defect": []}
    for sample_index, sample in enumerate(samples):
        analysis = pipeline.analyze(
            sample.image, confidence, iou_threshold, device, include_annotations=False
        )
        runtime = analysis["runtime"]
        image_info = analysis["image"]
        annotations = {
            "color": read_yolo_labels(
                sample.color_label, image_info["width"], image_info["height"], "color"
            )
            if sample.color_label
            else [],
            "defect": read_yolo_labels(
                sample.defect_label, image_info["width"], image_info["height"], "defect"
            )
            if sample.defect_label
            else [],
        }
        ground_truth = (
            annotations["color"] if target in {"color", "both"} else annotations["defect"]
        )
        if target == "both" and (
            len(annotations["color"]) != len(annotations["defect"])
            or any(
                _iou(first["bbox"], second["bbox"]) < 0.999
                for first, second in zip(annotations["color"], annotations["defect"], strict=True)
            )
        ):
            raise BenchmarkInputError(
                f"{sample.name}: color.txt และ defect.txt ต้องมี Bounding Box เดียวกันและเรียงลำดับเดียวกัน"
            )
        predictions = analysis["detections"]
        outcomes, matches, pairs = _match(predictions, ground_truth, iou_threshold)
        total_gt += len(ground_truth)
        total_predictions += len(predictions)
        total_matches += matches
        total_crops += analysis["crop_success"]["successful"]
        for key, value in analysis["timing"].items():
            timing[key] += value
        for threshold in np.arange(0.5, 1.0, 0.05):
            threshold_outcomes, _, _ = _match(predictions, ground_truth, float(threshold))
            all_predictions[float(threshold)].extend(
                (prediction["detector_confidence"], matched)
                for prediction, matched in zip(
                    sorted(predictions, key=lambda item: item["detector_confidence"], reverse=True),
                    threshold_outcomes,
                    strict=True,
                )
            )
        for kind in ["color", "defect"] if target == "both" else [target]:
            classification_pairs[kind].append((predictions, annotations[kind], pairs))
        precision, recall = (
            (matches / len(predictions) if predictions else 0.0),
            (matches / len(ground_truth) if ground_truth else 0.0),
        )
        per_image.append(
            {
                "image": sample.name,
                "ground_truth_boxes": len(ground_truth),
                "predicted_boxes": len(predictions),
                "matched_boxes": matches,
                "crop_successful": analysis["crop_success"]["successful"],
                "precision": precision,
                "recall": recall,
                "f1": (2 * precision * recall / (precision + recall))
                if precision + recall
                else 0.0,
                "quality": analysis["quality"],
                "timing": analysis["timing"],
            }
        )
        if sample_index == 0:
            preview = {
                "image": sample.name,
                "predicted_base64": pipeline.render_detection_overlay(
                    sample.image,
                    [prediction["bbox"] for prediction in predictions],
                    (70, 220, 80),
                    [_prediction_label(prediction, target) for prediction in predictions],
                    [
                        BOX_COLORS[_preview_class_key(prediction, target)]
                        for prediction in predictions
                    ],
                ),
                "ground_truth_base64": pipeline.render_detection_overlay(
                    sample.image,
                    [annotation["bbox"] for annotation in ground_truth],
                    (245, 170, 45),
                    [
                        _ground_truth_label(annotations, index, target)
                        for index in range(len(ground_truth))
                    ],
                    [
                        BOX_COLORS[_ground_truth_class_key(annotations, index, target)]
                        for index in range(len(ground_truth))
                    ],
                ),
            }
    classification = {}
    for kind in ["color", "defect"] if target == "both" else [target]:
        predictions, annotations, pairs = [], [], []
        for sample_predictions, sample_annotations, sample_pairs in classification_pairs[kind]:
            offset_prediction, offset_annotation = len(predictions), len(annotations)
            predictions.extend(sample_predictions)
            annotations.extend(sample_annotations)
            pairs.extend(
                (prediction + offset_prediction, annotation + offset_annotation)
                for prediction, annotation in sample_pairs
            )
        classification[kind] = _classification_result(kind, predictions, annotations, pairs)
    precision, recall = (
        (total_matches / total_predictions if total_predictions else 0.0),
        (total_matches / total_gt if total_gt else 0.0),
    )
    return {
        "target": target,
        "metrics": {
            "precision": precision,
            "recall": recall,
            "f1": (2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
            "map_50": _average_precision(all_predictions[0.5], total_gt),
            "map_50_95": float(
                np.mean(
                    [_average_precision(records, total_gt) for records in all_predictions.values()]
                )
            ),
            "ground_truth_boxes": total_gt,
            "predicted_boxes": total_predictions,
            "matched_boxes": total_matches,
            "crop_success_percent_of_ground_truth": (total_crops / total_gt * 100)
            if total_gt
            else 0.0,
        },
        "images": per_image,
        "preview": preview,
        "timing": dict(timing),
        "runtime": runtime,
        "classification": classification,
    }
