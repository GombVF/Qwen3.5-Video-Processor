from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from visual_service.contracts.responses import (
    ErrorResponse,
    ErrorDetail,
    HealthResponse,
    KernelDiagnosticsResponse,
    ModelInfoResponse,
    ReadyResponse,
)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="UP")


@router.get("/ready", response_model=ReadyResponse)
async def ready(request: Request):
    runtime = getattr(request.app.state, "qwen_runtime", None)
    if runtime is None:
        return JSONResponse(
            status_code=503,
            content=ErrorResponse(
                error=ErrorDetail(
                    code="MODEL_NOT_READY",
                    message="Visual model is not ready",
                    retryable=True,
                )
            ).model_dump(),
        )
    return ReadyResponse(
        ready=True,
        model=runtime.model_id,
        device=runtime.device_name,
        dtype=runtime.configured_dtype,
    )


@router.get("/model", response_model=ModelInfoResponse)
async def model_info(request: Request):
    runtime = getattr(request.app.state, "qwen_runtime", None)
    if runtime is None:
        return JSONResponse(
            status_code=503,
            content=ErrorResponse(
                error=ErrorDetail(
                    code="MODEL_NOT_READY",
                    message="Visual model is not ready",
                    retryable=True,
                )
            ).model_dump(),
        )

    from visual_service.settings import settings

    kd = runtime.kernel_diagnostics
    return ModelInfoResponse(
        modelId=runtime.model_id,
        modelRevision=runtime.model_revision,
        dtype=runtime.configured_dtype,
        actualDtypeDistribution=runtime.actual_dtype_distribution,
        device=runtime.device_name,
        attentionImplementation=runtime.attention_implementation,
        thinkingEnabled=runtime.thinking_enabled,
        torchVersion=runtime.torch_version,
        hipVersion=runtime.hip_version,
        transformersVersion=runtime.transformers_version,
        toolkitVersion=runtime.toolkit_version,
        videoBackend=runtime.video_backend,
        promptVersion=runtime.prompt_version,
        schemaVersion=settings.qwen_schema_version,
        loadedAt=runtime.loaded_at.isoformat(),
        kernelDiagnostics=KernelDiagnosticsResponse(
            flashLinearAttentionAvailable=kd.flash_linear_attention_available,
            flashLinearAttentionActive=kd.flash_linear_attention_active,
            flashLinearAttentionBinding=kd.flash_linear_attention_binding,
            flashLinearAttentionExecutionVerified=(
                kd.flash_linear_attention_chunk_calls
                + kd.flash_linear_attention_recurrent_calls
            ) > 0,
            flashLinearAttentionChunkCalls=kd.flash_linear_attention_chunk_calls,
            flashLinearAttentionRecurrentCalls=kd.flash_linear_attention_recurrent_calls,
            causalConv1dAvailable=kd.causal_conv1d_available,
            causalConv1dActive=kd.causal_conv1d_active,
            linearAttentionBackend=kd.linear_attention_backend,
            fullAttentionBackend=kd.full_attention_backend,
            fallbackReasons=list(kd.fallback_reasons),
        ),
    )
