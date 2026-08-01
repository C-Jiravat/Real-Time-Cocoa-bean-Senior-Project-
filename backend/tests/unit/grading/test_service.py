from cocoa_platform.grading import calculate_quality


def bean(color: str, defect: str) -> dict:
    return {"valid_crop": True, "color": {"key": color}, "defect": {"key": defect}}


def test_special_requires_strictly_less_than_thresholds() -> None:
    beans = [bean("brown", "normal") for _ in range(39)] + [bean("purple", "normal")]
    result = calculate_quality(beans, 40)
    assert result["bean_quality"] == "พิเศษ"
    assert result["fermentation_quality"] == "การหมักชั้นดี"


def test_purple_and_slaty_are_counted_once() -> None:
    result = calculate_quality([bean("purple", "slaty_hard_as_rock"), bean("brown", "normal")], 2)
    assert result["counts"]["purple_or_slaty"] == 1


def test_invalid_crop_never_returns_a_provisional_grade() -> None:
    result = calculate_quality([{"valid_crop": False}], 1)
    assert result["status"] == "incomplete"
    assert "bean_quality" not in result


def test_no_detections_cannot_be_evaluated() -> None:
    assert calculate_quality([], 0)["status"] == "not_evaluable"
