from datetime import datetime, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from cocoa_platform.contracts import (
    AnalysisResult,
    ApiEnvelope,
    BeanClassification,
    BoundingBox,
    ColorClass,
    ColorPrediction,
    DefectClass,
    DefectPrediction,
    Detection,
    DeviceActual,
    DeviceRequested,
    ExecutionProvider,
    Grade,
    GradeResult,
    GradeStatus,
    ModelFormat,
    ModelRole,
    OperationalLimits,
    RuntimeSelection,
    ErrorCode,
    ErrorEnvelope,
    ApiError,
    HealthResponse,
    RuntimeActiveResponse,
    RuntimeCapabilitiesResponse,
    ModelListResponse,
    ModelBundleListResponse,
    DatasetListResponse,
    BenchmarkListResponse,
    AnalysisHistoryResponse,
    BenchmarkHistoryResponse,
    JobResponse,
    OperationalLimitsResponse,
)


NOW = datetime(2026, 7, 28, tzinfo=timezone.utc)
HASH = "a" * 64


def runtime() -> RuntimeSelection:
    return RuntimeSelection(
        format=ModelFormat.ONNX,
        device_requested=DeviceRequested.CPU,
        device_actual=DeviceActual.CPU,
        execution_provider=ExecutionProvider.ONNX_CPU,
    )


def classification(detection_id):
    return BeanClassification(
        detection_id=detection_id,
        valid_crop=True,
        color=ColorPrediction(class_id=0, class_key=ColorClass.PURPLE, confidence=0.9),
        defect=DefectPrediction(class_id=3, class_key=DefectClass.MOLDY, confidence=0.8),
    )


def test_analysis_round_trip_and_strict_extra_fields() -> None:
    detection = Detection(id=uuid4(), bbox=BoundingBox(x_min=1, y_min=2, x_max=3, y_max=4), confidence=0.7)
    result = AnalysisResult(
        id=uuid4(), status=GradeStatus.EVALUATED, runtime=runtime(), detections=[detection],
        classifications=[classification(detection.id)],
        grade=GradeResult(status=GradeStatus.EVALUATED, grade=Grade.SPECIAL,
                          percentages={"moldy": 1, "purple_or_slaty": 2, "germinate": 1},
                          grade_standard_version="mvp-1"),
        model_hashes={ModelRole.DETECTOR: HASH, ModelRole.COLOR: HASH, ModelRole.DEFECT: HASH}, created_at=NOW,
    )
    assert AnalysisResult.model_validate_json(result.model_dump_json()) == result
    with pytest.raises(ValidationError):
        BoundingBox(x_min=0, y_min=0, x_max=1, y_max=1, unexpected=True)


@pytest.mark.parametrize("kwargs", [
    {"x_min": 1, "y_min": 1, "x_max": 1, "y_max": 2},
    {"x_min": 0, "y_min": 0, "x_max": float("inf"), "y_max": 2},
])
def test_invalid_bounding_boxes_rejected(kwargs) -> None:
    with pytest.raises(ValidationError):
        BoundingBox(**kwargs)


def test_invalid_confidence_mapping_and_provider_rejected() -> None:
    with pytest.raises(ValidationError):
        ColorPrediction(class_id=1, class_key=ColorClass.PURPLE, confidence=0.5)
    with pytest.raises(ValidationError):
        Detection(id=uuid4(), bbox=BoundingBox(x_min=0, y_min=0, x_max=1, y_max=1), confidence=1.01)
    with pytest.raises(ValidationError):
        RuntimeSelection(format=ModelFormat.ONNX, device_requested=DeviceRequested.CPU,
                         device_actual=DeviceActual.CPU, execution_provider=ExecutionProvider.PYTORCH_CPU)


@pytest.mark.parametrize("requested,actual", [(DeviceRequested.CPU, DeviceActual.GPU), (DeviceRequested.GPU, DeviceActual.CPU)])
def test_explicit_device_request_must_match_actual(requested, actual) -> None:
    with pytest.raises(ValidationError):
        RuntimeSelection(format=ModelFormat.ONNX, device_requested=requested, device_actual=actual,
                         execution_provider=ExecutionProvider.ONNX_CPU if actual is DeviceActual.CPU else ExecutionProvider.ONNX_CUDA)


def test_valid_bean_requires_both_top_one_predictions() -> None:
    with pytest.raises(ValidationError):
        BeanClassification(detection_id=uuid4(), valid_crop=True,
                           color=ColorPrediction(class_id=0, class_key=ColorClass.PURPLE, confidence=0.5))


def test_invalid_bean_requires_incomplete_analysis_and_null_grade() -> None:
    detection = Detection(id=uuid4(), bbox=BoundingBox(x_min=0, y_min=0, x_max=1, y_max=1), confidence=0.5)
    invalid = BeanClassification(detection_id=detection.id, valid_crop=False, failure_reason="crop failed")
    base = dict(id=uuid4(), runtime=runtime(), detections=[detection], classifications=[invalid],
                model_hashes={ModelRole.DETECTOR: HASH}, created_at=NOW)
    with pytest.raises(ValidationError):
        AnalysisResult(status=GradeStatus.EVALUATED, grade=GradeResult(status=GradeStatus.EVALUATED, grade=Grade.SPECIAL,
                       percentages={"moldy": 1, "purple_or_slaty": 1, "germinate": 1}, grade_standard_version="mvp"), **base)
    result = AnalysisResult(status=GradeStatus.INCOMPLETE, grade=GradeResult(status=GradeStatus.INCOMPLETE,
                            grade_standard_version="mvp", reason="crop failed"), **base)
    assert result.grade.grade is None and result.grade.percentages is None


def test_operational_limits_are_frozen_and_inclusive() -> None:
    limits = OperationalLimits()
    assert limits.inherited_normal_image_upload_limit == "50 MB"
    assert limits.allows_inclusive(536870912, limits.dataset_zip_bytes)
    assert not limits.allows_inclusive(536870913, limits.dataset_zip_bytes)
    assert limits.allows_inclusive(20000, limits.dataset_archive_files)
    assert not limits.allows_inclusive(20001, limits.dataset_archive_files)
    with pytest.raises(ValidationError):
        OperationalLimits(dataset_zip_bytes=1)


def test_api_envelope_json_schema_and_example_shape() -> None:
    schema = ApiEnvelope[RuntimeSelection].model_json_schema()
    assert "request_id" in schema["properties"]
    envelope = ApiEnvelope(data=runtime(), request_id=uuid4())
    assert ApiEnvelope[RuntimeSelection].model_validate_json(envelope.model_dump_json()) == envelope


@pytest.mark.parametrize("model", [
    HealthResponse, RuntimeActiveResponse, RuntimeCapabilitiesResponse, ModelListResponse, ModelBundleListResponse,
    DatasetListResponse, BenchmarkListResponse, AnalysisHistoryResponse, BenchmarkHistoryResponse, JobResponse,
    OperationalLimitsResponse, ErrorEnvelope,
])
def test_every_public_result_and_error_has_json_schema(model) -> None:
    schema = model.model_json_schema()
    assert schema["type"] == "object"
    assert schema["examples"]


def test_error_envelope_round_trip() -> None:
    value = ErrorEnvelope(error=ApiError(code=ErrorCode.LIMIT_EXCEEDED, message="too large", request_id=uuid4()))
    assert ErrorEnvelope.model_validate_json(value.model_dump_json()) == value
