from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from visual_service.contracts.responses import ErrorDetail, ErrorResponse

logger = logging.getLogger(__name__)


class ServiceError(Exception):
    status_code: int = 500
    error_code: str = "INFERENCE_FAILED"
    retryable: bool = True


class ModelNotReady(ServiceError):
    status_code = 503
    error_code = "MODEL_NOT_READY"
    retryable = True


class InvalidRequest(ServiceError):
    status_code = 400
    error_code = "INVALID_REQUEST"
    retryable = False


class WindowMismatch(ServiceError):
    status_code = 409
    error_code = "WINDOW_MISMATCH"
    retryable = False


def _error_response(status_code: int, code: str, message: str, retryable: bool) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(
            error=ErrorDetail(code=code, message=message, retryable=retryable)
        ).model_dump(),
    )


def _is_oom(exc: BaseException) -> bool:
    if type(exc).__name__ == "OutOfMemoryError":
        return True
    if isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower():
        return True
    if exc.__cause__ is not None:
        return _is_oom(exc.__cause__)
    return False


def register_exception_handlers(app: FastAPI) -> None:
    from visual_service.model.qwen_analyzer import InferenceError
    from visual_service.validation.output_validator import ModelOutputInvalid, ModelOutputTruncated
    from visual_service.validation.path_security import (
        InvalidMediaFile,
        MediaNotFound,
        PathOutsideMediaRoot,
    )

    @app.exception_handler(ServiceError)
    async def handle_service_error(request, exc: ServiceError):
        return _error_response(exc.status_code, exc.error_code, str(exc), exc.retryable)

    @app.exception_handler(PathOutsideMediaRoot)
    async def handle_path_outside_root(request, exc):
        return _error_response(403, "PATH_OUTSIDE_MEDIA_ROOT", str(exc), retryable=False)

    @app.exception_handler(MediaNotFound)
    async def handle_media_not_found(request, exc):
        return _error_response(404, "MEDIA_NOT_FOUND", str(exc), retryable=False)

    @app.exception_handler(InvalidMediaFile)
    async def handle_invalid_media(request, exc):
        return _error_response(400, "INVALID_REQUEST", str(exc), retryable=False)

    @app.exception_handler(ModelOutputTruncated)
    async def handle_output_truncated(request, exc):
        return _error_response(422, "MODEL_OUTPUT_TRUNCATED", str(exc), retryable=True)

    @app.exception_handler(ModelOutputInvalid)
    async def handle_output_invalid(request, exc):
        return _error_response(422, "MODEL_OUTPUT_INVALID", str(exc), retryable=True)

    @app.exception_handler(InferenceError)
    async def handle_inference_error(request, exc):
        if _is_oom(exc):
            return _error_response(507, "GPU_OUT_OF_MEMORY", "GPU ran out of memory during inference", retryable=True)
        return _error_response(500, "INFERENCE_FAILED", str(exc), retryable=True)

    @app.exception_handler(Exception)
    async def handle_unhandled(request, exc):
        if _is_oom(exc):
            return _error_response(507, "GPU_OUT_OF_MEMORY", "GPU ran out of memory", retryable=True)
        logger.exception("unhandled_error", extra={"data": {
            "errorType": type(exc).__name__,
            "path": str(request.url.path),
        }})
        return _error_response(500, "INFERENCE_FAILED", "Internal server error", retryable=True)
