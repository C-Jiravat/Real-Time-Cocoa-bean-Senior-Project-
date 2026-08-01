"""The one backend source of truth for the user-approved cocoa standards."""

from __future__ import annotations

from collections.abc import Iterable


QUALITY_STANDARD_VERSION = "cocoa-quality-mvp-2026-08"


def _percent(numerator: int, denominator: int) -> float:
    return (numerator / denominator) * 100 if denominator else 0.0


def _pick_grade(values: tuple[float, float, float], thresholds: tuple[tuple[str, float, float, float], ...]) -> str:
    for label, moldy, purple_or_slaty, germinate in thresholds:
        if all(actual < limit for actual, limit in zip(values, (moldy, purple_or_slaty, germinate), strict=True)):
            return label
    return "ไม่ผ่านเกณฑ์"


def calculate_quality(beans: Iterable[dict], detected_total: int) -> dict:
    """Calculate two quality tables without double-counting purple/slaty beans."""
    beans = list(beans)
    if detected_total == 0:
        return {
            "status": "not_evaluable",
            "standard_version": QUALITY_STANDARD_VERSION,
            "reason": "ไม่พบเมล็ดโกโก้ในภาพ",
        }
    if any(not bean.get("valid_crop") for bean in beans):
        return {
            "status": "incomplete",
            "standard_version": QUALITY_STANDARD_VERSION,
            "reason": "ไม่สามารถ crop หรือจำแนกเมล็ดได้ครบทุก bounding box",
        }

    moldy = sum(bean["defect"]["key"] == "moldy" for bean in beans)
    purple_or_slaty = sum(
        bean["color"]["key"] == "purple" or bean["defect"]["key"] == "slaty_hard_as_rock"
        for bean in beans
    )
    germinate = sum(bean["defect"]["key"] == "germinate" for bean in beans)
    percentages = {
        "moldy": _percent(moldy, detected_total),
        "purple_or_slaty": _percent(purple_or_slaty, detected_total),
        "germinate": _percent(germinate, detected_total),
    }
    values = (percentages["moldy"], percentages["purple_or_slaty"], percentages["germinate"])
    bean_grade = _pick_grade(values, (
        ("พิเศษ", 3.0, 3.0, 2.5),
        ("ชั้น 1", 3.0, 5.0, 3.0),
        ("ชั้น 2", 4.0, 8.0, 5.0),
    ))
    fermentation_grade = _pick_grade(values, (
        ("การหมักชั้นดี", 5.0, 3.0, 5.0),
        ("การหมักขึ้นพอใช้", 10.0, 5.0, 10.0),
    ))
    return {
        "status": "evaluated",
        "standard_version": QUALITY_STANDARD_VERSION,
        "percentages": percentages,
        "bean_quality": bean_grade,
        "fermentation_quality": fermentation_grade,
        "counts": {
            "moldy": moldy,
            "purple_or_slaty": purple_or_slaty,
            "germinate": germinate,
            "detected_total": detected_total,
        },
    }
