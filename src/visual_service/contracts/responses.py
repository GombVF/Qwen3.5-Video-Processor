from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from visual_service.settings import SCHEMA_VERSION


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    ready: bool
    model: str | None = None
    device: str | None = None
    dtype: str | None = None


class KernelDiagnosticsResponse(BaseModel):
    flashLinearAttentionAvailable: bool = False
    flashLinearAttentionActive: bool = False
    flashLinearAttentionBinding: str = "reference"
    flashLinearAttentionExecutionVerified: bool = False
    flashLinearAttentionChunkCalls: int = 0
    flashLinearAttentionRecurrentCalls: int = 0
    causalConv1dAvailable: bool = False
    causalConv1dActive: bool = False
    linearAttentionBackend: str = "reference"
    fullAttentionBackend: str = "sdpa"
    fallbackReasons: list[str] = Field(default_factory=list)


class ModelInfoResponse(BaseModel):
    modelId: str
    modelRevision: str
    dtype: str
    actualDtypeDistribution: dict[str, int] = Field(default_factory=dict)
    device: str
    attentionImplementation: str
    thinkingEnabled: bool
    torchVersion: str
    hipVersion: str
    transformersVersion: str
    toolkitVersion: str
    videoBackend: str
    promptVersion: str
    schemaVersion: str
    loadedAt: str
    kernelDiagnostics: KernelDiagnosticsResponse = Field(
        default_factory=KernelDiagnosticsResponse,
    )


class WindowResponse(BaseModel):
    sequence: int
    startMs: int
    endMs: int


class SamplingResponse(BaseModel):
    fps: float
    minFrames: int
    maxFrames: int
    maxPixels: int


class PlanResponse(BaseModel):
    profile: str
    strategy: str
    windows: list[WindowResponse]
    sampling: SamplingResponse


class ErrorDetail(BaseModel):
    code: str
    message: str
    retryable: bool


class ErrorResponse(BaseModel):
    error: ErrorDetail


class ModelDetail(BaseModel):
    id: str
    revision: str
    dtype: str
    attentionImplementation: str


class VersionsDetail(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    torch: str
    transformers: str
    qwenVlUtils: str
    prompt: str
    schema_version: str = Field(alias="schema")


class ObservationResponse(BaseModel):
    order: int
    type: str
    description: str


class PassStatus(str, Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    FAILED = "failed"
    SKIPPED = "skipped"


class PassMetrics(BaseModel):
    decodeMs: int = 0
    preprocessMs: int = 0
    inferenceMs: int = 0
    generatedTokens: int = 0
    visualTokens: int | None = None
    peakVramBytes: int = 0
    inferenceCount: int = 0
    selectedFrames: int = 0
    finishReason: str = "stop"
    isPartial: bool = False
    status: PassStatus = PassStatus.COMPLETE


class WindowAccounting(BaseModel):
    plannedWindows: int = 0
    attemptedWindows: int = 0
    successfulWindows: int = 0
    partialWindows: int = 0
    failedWindows: int = 0


class VisualMetricsResponse(BaseModel):
    totalMs: int
    peakVramBytes: int
    windowCount: int
    analysisMode: str = "SEMANTIC_AND_OCR"
    globalOverviewPass: PassMetrics
    semanticPass: PassMetrics
    ocrPass: PassMetrics
    windowAccounting: WindowAccounting = Field(default_factory=WindowAccounting)


class VisualContent(BaseModel):
    schemaVersion: str = SCHEMA_VERSION
    summary: str
    observations: list[ObservationResponse] = Field(default_factory=list)
    visibleTexts: list[str] = Field(default_factory=list)
    communicatedIdeas: list[str] = Field(default_factory=list)
    themes: list[str] = Field(default_factory=list)
    objects: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)


class VisualAnalysisResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    runId: str
    contentId: str
    model: ModelDetail
    versions: VersionsDetail
    visual: VisualContent
    metrics: VisualMetricsResponse
    validationWarnings: list[str] = Field(default_factory=list)
    consolidationDiagnostics: list[str] = Field(default_factory=list)
