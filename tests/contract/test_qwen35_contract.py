from visual_service.contracts.requests import VideoAnalyzeRequest
from visual_service.contracts.responses import (
    KernelDiagnosticsResponse,
    ModelDetail,
    ModelInfoResponse,
    ObservationResponse,
    PassMetrics,
    PassStatus,
    VersionsDetail,
    VisualAnalysisResponse,
    VisualContent,
    VisualMetricsResponse,
    WindowAccounting,
)
from visual_service.settings import AnalysisMode


def test_public_semantic_contract_has_no_temporal_fields():
    response = VisualAnalysisResponse(
        runId="run-1",
        contentId="content-1",
        model=ModelDetail(
            id="Qwen/Qwen3.5-4B",
            revision="abc",
            dtype="bfloat16",
            attentionImplementation="sdpa",
        ),
        versions=VersionsDetail(
            torch="2.9.1", transformers="5.11.0", qwenVlUtils="0.0.14",
            prompt="visual-analysis-qwen35-v3", schema="5.0",
        ),
        visual=VisualContent(
            summary="Una persona prepara comida.",
            observations=[ObservationResponse(order=1, type="action", description="Mezcla ingredientes.")],
            visibleTexts=["RECETA"], objects=["tazón"], topics=["cocina"],
        ),
        metrics=VisualMetricsResponse(
            totalMs=1, peakVramBytes=1, windowCount=1,
            globalOverviewPass=PassMetrics(), semanticPass=PassMetrics(), ocrPass=PassMetrics(),
        ),
    )
    payload = response.model_dump(by_alias=True)
    semantic = payload["visual"]
    forbidden = {"startMs", "endMs", "frameIndex", "frameIndices", "timestamp"}
    assert forbidden.isdisjoint(semantic.keys())
    assert forbidden.isdisjoint(semantic["observations"][0].keys())
    assert semantic["schemaVersion"] == "5.0"


def test_pass_metrics_has_status_field():
    metrics = PassMetrics()
    assert metrics.status == PassStatus.COMPLETE
    assert metrics.model_dump()["status"] == "complete"


def test_pass_status_skipped():
    metrics = PassMetrics(status=PassStatus.SKIPPED)
    assert metrics.status == PassStatus.SKIPPED
    assert metrics.model_dump()["status"] == "skipped"


def test_metrics_response_includes_analysis_mode():
    metrics = VisualMetricsResponse(
        totalMs=1, peakVramBytes=1, windowCount=1,
        analysisMode="SEMANTIC_ONLY",
        globalOverviewPass=PassMetrics(),
        semanticPass=PassMetrics(),
        ocrPass=PassMetrics(status=PassStatus.SKIPPED),
    )
    payload = metrics.model_dump()
    assert payload["analysisMode"] == "SEMANTIC_ONLY"
    assert payload["ocrPass"]["status"] == "skipped"


def test_request_accepts_analysis_mode():
    req = VideoAnalyzeRequest(
        runId="r1",
        contentId="c1",
        mediaPath="/path/to/video.mp4",
        durationMs=12000,
        analysisMode=AnalysisMode.SEMANTIC_ONLY,
    )
    assert req.analysisMode == AnalysisMode.SEMANTIC_ONLY


def test_request_analysis_mode_defaults_to_none():
    req = VideoAnalyzeRequest(
        runId="r1",
        contentId="c1",
        mediaPath="/path/to/video.mp4",
        durationMs=12000,
    )
    assert req.analysisMode is None


def test_kernel_diagnostics_response_structure():
    kd = KernelDiagnosticsResponse()
    payload = kd.model_dump()
    assert payload["flashLinearAttentionAvailable"] is False
    assert payload["linearAttentionBackend"] == "reference"
    assert payload["fullAttentionBackend"] == "sdpa"
    assert payload["fallbackReasons"] == []


def test_window_accounting_fields():
    acct = WindowAccounting(
        plannedWindows=7,
        attemptedWindows=7,
        successfulWindows=6,
        partialWindows=1,
        failedWindows=1,
    )
    payload = acct.model_dump()
    assert payload["plannedWindows"] == 7
    assert payload["attemptedWindows"] == 7
    assert payload["successfulWindows"] == 6
    assert payload["partialWindows"] == 1
    assert payload["failedWindows"] == 1


def test_metrics_response_includes_window_accounting():
    metrics = VisualMetricsResponse(
        totalMs=1, peakVramBytes=1, windowCount=3,
        globalOverviewPass=PassMetrics(),
        semanticPass=PassMetrics(),
        ocrPass=PassMetrics(),
        windowAccounting=WindowAccounting(plannedWindows=3, attemptedWindows=3, successfulWindows=3),
    )
    payload = metrics.model_dump()
    assert payload["windowAccounting"]["plannedWindows"] == 3


def test_model_info_includes_kernel_diagnostics():
    info = ModelInfoResponse(
        modelId="test",
        modelRevision="abc",
        dtype="bfloat16",
        device="AMD",
        attentionImplementation="sdpa",
        thinkingEnabled=False,
        torchVersion="2.9.1",
        hipVersion="7.2",
        transformersVersion="5.18.0",
        toolkitVersion="0.0.14",
        videoBackend="pyav",
        promptVersion="visual-analysis-qwen35-v3",
        schemaVersion="5.0",
        loadedAt="2026-01-01T00:00:00",
    )
    payload = info.model_dump()
    assert "kernelDiagnostics" in payload
    assert payload["kernelDiagnostics"]["linearAttentionBackend"] == "reference"
