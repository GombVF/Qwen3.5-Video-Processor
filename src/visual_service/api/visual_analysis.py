import logging
from pathlib import Path

from fastapi import APIRouter, Request

from visual_service.contracts.requests import PlanRequest, VideoAnalyzeRequest
from visual_service.contracts.responses import (
    ModelDetail,
    ObservationResponse,
    PlanResponse,
    SamplingResponse,
    VersionsDetail,
    VisualAnalysisResponse,
    VisualContent,
    WindowResponse,
)
from visual_service.errors import ModelNotReady
from visual_service.logging import correlation
from visual_service.model.pipeline import run_visual_pipeline
from visual_service.settings import settings
from visual_service.validation.path_security import resolve_media_path
from visual_service.video.planner import create_windows
from visual_service.video.profiles import semantic_profile

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/visual-analysis")


@router.post("/plan", response_model=PlanResponse)
async def plan(body: PlanRequest) -> PlanResponse:
    profile = semantic_profile(settings)
    windows, strategy = create_windows(body.durationMs, profile)
    return PlanResponse(
        profile=profile.name,
        strategy=strategy,
        windows=[
            WindowResponse(sequence=item.sequence, startMs=item.start_ms, endMs=item.end_ms)
            for item in windows
        ],
        sampling=SamplingResponse(
            fps=profile.fps,
            minFrames=profile.min_frames,
            maxFrames=profile.max_frames,
            maxPixels=profile.max_pixels,
        ),
    )


@router.post(
    "/analyze",
    response_model=VisualAnalysisResponse,
    response_model_by_alias=True,
)
async def analyze_video(body: VideoAnalyzeRequest, request: Request) -> VisualAnalysisResponse:
    correlation.set({"runId": body.runId, "contentId": body.contentId})
    runtime = getattr(request.app.state, "qwen_runtime", None)
    if runtime is None:
        raise ModelNotReady("Visual model is not loaded")

    media_path = resolve_media_path(Path(body.mediaPath), settings.media_root)
    result = await run_visual_pipeline(
        runtime,
        media_path,
        body.durationMs,
        settings,
        analysis_mode=body.analysisMode,
    )
    content = result.consolidated
    return VisualAnalysisResponse(
        runId=body.runId,
        contentId=body.contentId,
        model=ModelDetail(
            id=runtime.model_id,
            revision=runtime.model_revision,
            dtype=runtime.configured_dtype,
            attentionImplementation=runtime.attention_implementation,
        ),
        versions=VersionsDetail(
            torch=runtime.torch_version,
            transformers=runtime.transformers_version,
            qwenVlUtils=runtime.toolkit_version,
            prompt=runtime.prompt_version,
            schema=settings.qwen_schema_version,
        ),
        visual=VisualContent(
            schemaVersion=settings.qwen_schema_version,
            summary=content.summary,
            observations=[ObservationResponse(**item) for item in content.observations],
            visibleTexts=content.visible_texts,
            objects=content.objects,
            topics=content.topics,
        ),
        metrics=result.metrics,
        validationWarnings=result.warnings,
        consolidationDiagnostics=content.diagnostics,
    )
