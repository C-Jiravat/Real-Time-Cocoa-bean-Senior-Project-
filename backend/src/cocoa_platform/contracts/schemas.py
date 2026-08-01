"""Canonical strict Pydantic v2 contracts for the Cocoa Bean AI Platform."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from math import isfinite
from typing import Annotated, Generic, Literal, TypeVar
from uuid import UUID
import re

from pydantic import BaseModel, ConfigDict, Field, model_validator


SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
NonEmpty = Annotated[str, Field(min_length=1)]
Probability = Annotated[float, Field(ge=0.0, le=1.0)]
NonNegative = Annotated[float, Field(ge=0.0)]


class ContractModel(BaseModel):
    """Shared API-model policy: reject unknown fields and validate mutations."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, json_schema_extra={"examples": [{}]})


class ModelFormat(str, Enum):
    PTH = "pth"
    ONNX = "onnx"


class ModelRole(str, Enum):
    DETECTOR = "detector"
    COLOR = "color"
    DEFECT = "defect"


class ColorClass(str, Enum):
    PURPLE = "purple"
    BROWN = "brown"


class DefectClass(str, Enum):
    NORMAL = "normal"
    GERMINATE = "germinate"
    SLATY_HARD_AS_ROCK = "slaty_hard_as_rock"
    MOLDY = "moldy"


COLOR_CLASS_IDS: dict[ColorClass, int] = {ColorClass.PURPLE: 0, ColorClass.BROWN: 1}
DEFECT_CLASS_IDS: dict[DefectClass, int] = {
    DefectClass.NORMAL: 0,
    DefectClass.GERMINATE: 1,
    DefectClass.SLATY_HARD_AS_ROCK: 2,
    DefectClass.MOLDY: 3,
}


class DeviceRequested(str, Enum):
    AUTO = "auto"
    CPU = "cpu"
    GPU = "gpu"


class DeviceActual(str, Enum):
    CPU = "cpu"
    GPU = "gpu"


class ExecutionProvider(str, Enum):
    PYTORCH_CPU = "PyTorchCPU"
    PYTORCH_CUDA = "PyTorchCUDA"
    ONNX_CPU = "CPUExecutionProvider"
    ONNX_CUDA = "CUDAExecutionProvider"


class GradeStatus(str, Enum):
    EVALUATED = "evaluated"
    INCOMPLETE = "incomplete"
    NOT_EVALUABLE = "not_evaluable"


class Grade(str, Enum):
    SPECIAL = "special"
    GRADE_1 = "grade_1"
    GRADE_2 = "grade_2"
    REJECTED = "rejected"


class DatasetStatus(str, Enum):
    VALIDATING = "validating"
    VALID = "valid"
    INVALID = "invalid"
    SOFT_DELETED = "soft_deleted"


class BenchmarkStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobStatus(str, Enum):
    IDLE = "idle"
    LOADING = "loading"
    READY = "ready"
    PROCESSING = "processing"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SyncStatus(str, Enum):
    LOCAL_ONLY = "local_only"
    PENDING_SYNC = "pending_sync"
    SYNCED = "synced"
    FAILED = "failed"


class ErrorCode(str, Enum):
    VALIDATION_ERROR = "validation_error"
    NOT_FOUND = "not_found"
    CONFLICT = "conflict"
    RUNTIME_UNAVAILABLE = "runtime_unavailable"
    LIMIT_EXCEEDED = "limit_exceeded"
    INCOMPLETE_RESULT = "incomplete_result"
    INTERNAL_ERROR = "internal_error"


class BoundingBox(ContractModel):
    x_min: NonNegative
    y_min: NonNegative
    x_max: NonNegative
    y_max: NonNegative

    @model_validator(mode="after")
    def validate_order_and_finiteness(self) -> "BoundingBox":
        if not all(isfinite(value) for value in (self.x_min, self.y_min, self.x_max, self.y_max)):
            raise ValueError("bounding-box coordinates must be finite")
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("bounding box requires x_max > x_min and y_max > y_min")
        return self


class Detection(ContractModel):
    id: UUID
    bbox: BoundingBox
    confidence: Probability
    class_id: Literal[0] = 0
    class_key: Literal["cocoa_bean"] = "cocoa_bean"

    @model_validator(mode="after")
    def validate_finite_confidence(self) -> "Detection":
        if not isfinite(self.confidence):
            raise ValueError("confidence must be finite")
        return self


class ColorPrediction(ContractModel):
    class_id: int
    class_key: ColorClass
    confidence: Probability

    @model_validator(mode="after")
    def validate_mapping_and_confidence(self) -> "ColorPrediction":
        if COLOR_CLASS_IDS[self.class_key] != self.class_id:
            raise ValueError("color class_id does not match class_key")
        if not isfinite(self.confidence):
            raise ValueError("confidence must be finite")
        return self


class DefectPrediction(ContractModel):
    class_id: int
    class_key: DefectClass
    confidence: Probability

    @model_validator(mode="after")
    def validate_mapping_and_confidence(self) -> "DefectPrediction":
        if DEFECT_CLASS_IDS[self.class_key] != self.class_id:
            raise ValueError("defect class_id does not match class_key")
        if not isfinite(self.confidence):
            raise ValueError("confidence must be finite")
        return self


class BeanClassification(ContractModel):
    detection_id: UUID
    valid_crop: bool
    color: ColorPrediction | None = None
    defect: DefectPrediction | None = None
    failure_reason: str | None = None

    @model_validator(mode="after")
    def validate_complete_valid_crop(self) -> "BeanClassification":
        if self.valid_crop and (self.color is None or self.defect is None):
            raise ValueError("every valid bean requires one color and one defect top-1 prediction")
        if not self.valid_crop and (self.color is not None or self.defect is not None):
            raise ValueError("invalid crops cannot carry classifier predictions")
        if not self.valid_crop and not self.failure_reason:
            raise ValueError("invalid crops require a failure_reason")
        return self


class GradePercentages(ContractModel):
    moldy: NonNegative
    purple_or_slaty: NonNegative
    germinate: NonNegative


class GradeResult(ContractModel):
    status: GradeStatus
    grade: Grade | None = None
    percentages: GradePercentages | None = None
    grade_standard_version: NonEmpty
    reason: str | None = None

    @model_validator(mode="after")
    def validate_status(self) -> "GradeResult":
        if self.status is GradeStatus.EVALUATED and (self.grade is None or self.percentages is None):
            raise ValueError("evaluated grade requires grade and percentages")
        if self.status is not GradeStatus.EVALUATED and self.grade is not None:
            raise ValueError("incomplete or not_evaluable grade must be null")
        if self.status is not GradeStatus.EVALUATED and not self.reason:
            raise ValueError("non-evaluated grade requires a reason")
        return self


class RuntimeSelection(ContractModel):
    format: ModelFormat
    device_requested: DeviceRequested
    device_actual: DeviceActual
    execution_provider: ExecutionProvider

    @model_validator(mode="after")
    def validate_provider(self) -> "RuntimeSelection":
        if self.device_requested is not DeviceRequested.AUTO and self.device_requested.value != self.device_actual.value:
            raise ValueError("explicit cpu/gpu request must match device_actual")
        valid = {
            (ModelFormat.PTH, DeviceActual.CPU): ExecutionProvider.PYTORCH_CPU,
            (ModelFormat.PTH, DeviceActual.GPU): ExecutionProvider.PYTORCH_CUDA,
            (ModelFormat.ONNX, DeviceActual.CPU): ExecutionProvider.ONNX_CPU,
            (ModelFormat.ONNX, DeviceActual.GPU): ExecutionProvider.ONNX_CUDA,
        }
        if valid[(self.format, self.device_actual)] is not self.execution_provider:
            raise ValueError("execution_provider is incompatible with runtime format and actual device")
        return self


class ModelArtifact(ContractModel):
    id: UUID
    role: ModelRole
    format: ModelFormat
    sha256: str
    file_name: NonEmpty
    registry_key: NonEmpty
    input_shape: tuple[int | None, ...]
    output_shape: tuple[int | None, ...]
    created_at: datetime

    @model_validator(mode="after")
    def validate_hash_and_time(self) -> "ModelArtifact":
        if not SHA256_RE.fullmatch(self.sha256):
            raise ValueError("sha256 must be a 64-character hexadecimal digest")
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        return self


class ModelBundleMember(ContractModel):
    role: ModelRole
    artifact_id: UUID
    sha256: str

    @model_validator(mode="after")
    def validate_hash(self) -> "ModelBundleMember":
        if not SHA256_RE.fullmatch(self.sha256):
            raise ValueError("bundle member sha256 must be a 64-character hexadecimal digest")
        return self


class ModelBundle(ContractModel):
    id: UUID
    format: ModelFormat
    members: list[ModelBundleMember]
    active: bool = False
    created_at: datetime

    @model_validator(mode="after")
    def validate_members(self) -> "ModelBundle":
        roles = {member.role for member in self.members}
        if roles != {ModelRole.DETECTOR, ModelRole.COLOR, ModelRole.DEFECT} or len(self.members) != 3:
            raise ValueError("model bundle requires exactly one detector, color, and defect member")
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        return self


class DatasetTask(str, Enum):
    DETECTOR = "detector"
    COLOR = "color"
    DEFECT = "defect"


class DatasetProfile(ContractModel):
    id: UUID
    name: NonEmpty
    version: NonEmpty
    task: DatasetTask
    sha256: str
    image_count: Annotated[int, Field(ge=0)]
    label_count: Annotated[int, Field(ge=0)]
    status: DatasetStatus
    created_at: datetime

    @model_validator(mode="after")
    def validate_hash_and_time(self) -> "DatasetProfile":
        if not SHA256_RE.fullmatch(self.sha256):
            raise ValueError("dataset sha256 must be a 64-character hexadecimal digest")
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        return self


class DatasetValidationSummary(ContractModel):
    valid: bool
    archive_file_count: Annotated[int, Field(ge=0)]
    extracted_bytes: Annotated[int, Field(ge=0)]
    errors: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_result(self) -> "DatasetValidationSummary":
        if self.valid and self.errors:
            raise ValueError("valid dataset summary cannot contain validation errors")
        if not self.valid and not self.errors:
            raise ValueError("invalid dataset summary requires validation errors")
        return self


class OperationalLimits(ContractModel):
    """Approved SP-005 settings; image size intentionally remains the PRD unit."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    dataset_zip_bytes: Literal[536870912] = 536870912
    dataset_extracted_bytes: Literal[2147483648] = 2147483648
    dataset_archive_files: Literal[20000] = 20000
    pth_upload_bytes: Literal[268435456] = 268435456
    onnx_upload_bytes: Literal[268435456] = 268435456
    onnx_bundle_bytes: Literal[536870912] = 536870912
    deletion_mode: Literal["soft_delete"] = "soft_delete"
    deletion_recovery_seconds: Literal[604800] = 604800
    compare_min_top1_agreement: Literal[0.95] = 0.95
    compare_max_mean_confidence_delta: Literal[0.10] = 0.10
    compare_min_mean_bbox_iou: Literal[0.75] = 0.75
    compare_max_metric_delta: Literal[0.02] = 0.02
    compare_max_p95_latency_ratio_delta: Literal[0.20] = 0.20
    compare_max_peak_memory_ratio_delta: Literal[0.20] = 0.20
    image_analysis_p95_latency_ms: Literal[2000] = 2000
    live_target_fps: Literal[5] = 5
    inherited_normal_image_upload_limit: Literal["50 MB"] = "50 MB"

    @classmethod
    def allows_inclusive(cls, observed: int, maximum: int) -> bool:
        return observed <= maximum


class TimingMetrics(ContractModel):
    mean_ms: NonNegative
    median_ms: NonNegative
    p95_ms: NonNegative
    fps: NonNegative


class BenchmarkMetric(ContractModel):
    class_key: str
    precision: Probability
    recall: Probability
    f1: Probability
    support: Annotated[int, Field(ge=0)]


class BenchmarkRun(ContractModel):
    id: UUID
    idempotency_key: UUID
    task: DatasetTask
    format: ModelFormat
    status: BenchmarkStatus
    dataset_id: UUID
    dataset_hash: str
    runtime: RuntimeSelection
    model_bundle_id: UUID
    metric_implementation_version: NonEmpty
    timing: TimingMetrics | None = None
    class_metrics: list[BenchmarkMetric] = Field(default_factory=list)
    created_at: datetime

    @model_validator(mode="after")
    def validate_hash_time_and_completion(self) -> "BenchmarkRun":
        if not SHA256_RE.fullmatch(self.dataset_hash):
            raise ValueError("dataset_hash must be a 64-character hexadecimal digest")
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        if self.status is BenchmarkStatus.COMPLETED and self.timing is None:
            raise ValueError("completed benchmark requires timing metrics")
        return self


class BenchmarkReport(ContractModel):
    benchmark_run_id: UUID
    html_report_path: NonEmpty
    artifact_paths: list[str] = Field(default_factory=list)
    created_at: datetime

    @model_validator(mode="after")
    def validate_time(self) -> "BenchmarkReport":
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        return self


class ComparisonWarning(ContractModel):
    code: NonEmpty
    message: NonEmpty
    observed: float | None = None
    threshold: float | None = None


class BenchmarkComparison(ContractModel):
    left_run_id: UUID
    right_run_id: UUID
    compatible: bool
    warnings: list[ComparisonWarning] = Field(default_factory=list)
    informational_no_match: bool = False

    @model_validator(mode="after")
    def validate_diagnostic_only(self) -> "BenchmarkComparison":
        if not self.compatible and not self.warnings:
            raise ValueError("incompatible comparison requires actionable warning(s)")
        return self


class JobState(ContractModel):
    id: UUID
    generation: Annotated[int, Field(ge=1)]
    status: JobStatus
    created_at: datetime
    updated_at: datetime

    @model_validator(mode="after")
    def validate_times(self) -> "JobState":
        if any(value.tzinfo is None or value.utcoffset() is None for value in (self.created_at, self.updated_at)):
            raise ValueError("job timestamps must be timezone-aware")
        if self.updated_at < self.created_at:
            raise ValueError("updated_at must not precede created_at")
        return self


class PersistenceReference(ContractModel):
    id: UUID
    idempotency_key: UUID
    sync_status: SyncStatus
    local_record_key: NonEmpty
    source_dataset_unavailable: bool = False
    created_at: datetime

    @model_validator(mode="after")
    def validate_time(self) -> "PersistenceReference":
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        return self


class PersistenceArtifact(ContractModel):
    id: UUID
    run_id: UUID
    storage_key: NonEmpty
    content_type: NonEmpty
    sync_status: SyncStatus
    deleted_at: datetime | None = None

    @model_validator(mode="after")
    def validate_deleted_time(self) -> "PersistenceArtifact":
        if self.deleted_at is not None and (self.deleted_at.tzinfo is None or self.deleted_at.utcoffset() is None):
            raise ValueError("deleted_at must be timezone-aware")
        return self


class AnalysisResult(ContractModel):
    id: UUID
    status: GradeStatus
    runtime: RuntimeSelection
    detections: list[Detection]
    classifications: list[BeanClassification]
    grade: GradeResult
    model_hashes: dict[ModelRole, str]
    created_at: datetime

    @model_validator(mode="after")
    def validate_analysis_invariants(self) -> "AnalysisResult":
        detection_ids = {detection.id for detection in self.detections}
        classification_ids = [item.detection_id for item in self.classifications]
        if len(detection_ids) != len(self.detections):
            raise ValueError("detection IDs must be unique")
        if set(classification_ids) != detection_ids or len(classification_ids) != len(detection_ids):
            raise ValueError("each detection requires exactly one classification record")
        if any(not SHA256_RE.fullmatch(value) for value in self.model_hashes.values()):
            raise ValueError("model_hashes must contain SHA-256 digests")
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        incomplete_beans = any(not item.valid_crop for item in self.classifications)
        if incomplete_beans and self.status is not GradeStatus.INCOMPLETE:
            raise ValueError("invalid bean classification requires incomplete analysis status")
        if incomplete_beans and (self.grade.grade is not None or self.grade.percentages is not None):
            raise ValueError("incomplete bean classification requires null grade and percentages")
        if self.status is GradeStatus.EVALUATED and self.grade.status is not GradeStatus.EVALUATED:
            raise ValueError("evaluated analysis requires evaluated grade")
        if self.status is not GradeStatus.EVALUATED and self.grade.grade is not None:
            raise ValueError("incomplete analysis cannot expose a grade")
        return self


class ApiError(ContractModel):
    code: ErrorCode
    message: NonEmpty
    request_id: UUID
    details: dict[str, str] = Field(default_factory=dict)


T = TypeVar("T")


class ApiEnvelope(ContractModel, Generic[T]):
    data: T
    request_id: UUID


class ErrorEnvelope(ContractModel):
    error: ApiError


class HealthComponent(ContractModel):
    name: NonEmpty
    status: Literal["ready", "degraded", "unavailable"]
    detail: str | None = None


class HealthResponse(ContractModel):
    status: Literal["ready", "degraded", "unavailable"]
    components: list[HealthComponent]


class RuntimeCapabilitiesResponse(ContractModel):
    available: list[RuntimeSelection]


class ModelListResponse(ContractModel):
    items: list[ModelArtifact]


class ModelBundleListResponse(ContractModel):
    items: list[ModelBundle]


class DatasetListResponse(ContractModel):
    items: list[DatasetProfile]


class BenchmarkListResponse(ContractModel):
    items: list[BenchmarkRun]


class AnalysisHistoryResponse(ContractModel):
    items: list[AnalysisResult]


class BenchmarkHistoryResponse(ContractModel):
    items: list[BenchmarkRun]


class JobResponse(ContractModel):
    job: JobState


class OperationalLimitsResponse(ContractModel):
    limits: OperationalLimits


class RuntimeActiveResponse(ContractModel):
    active_bundle_id: UUID | None = None
    runtime: RuntimeSelection | None = None

    @model_validator(mode="after")
    def validate_pair(self) -> "RuntimeActiveResponse":
        if (self.active_bundle_id is None) != (self.runtime is None):
            raise ValueError("active_bundle_id and runtime must be supplied together")
        return self
