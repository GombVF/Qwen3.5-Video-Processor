import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from visual_service.api.health import router as health_router
from visual_service.api.visual_analysis import router as visual_analysis_router
from visual_service.errors import register_exception_handlers
from visual_service.logging import configure_logging
from visual_service.model.lifecycle import load_qwen_runtime
from visual_service.model.qwen_analyzer import wait_for_inference_complete
from visual_service.settings import settings

configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        runtime = await load_qwen_runtime(settings)
        app.state.qwen_runtime = runtime
        logger.info("model_ready")
    except Exception:
        logger.exception("model_load_failed")
        app.state.qwen_runtime = None
    yield
    await wait_for_inference_complete()
    runtime = getattr(app.state, "qwen_runtime", None)
    if runtime is not None:
        await runtime.close()


app = FastAPI(
    title="Qwen3.5 Visual Analysis Service",
    version="1.0.0",
    lifespan=lifespan,
)
register_exception_handlers(app)
app.include_router(health_router)
app.include_router(visual_analysis_router)
