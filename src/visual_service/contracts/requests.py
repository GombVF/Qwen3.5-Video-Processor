from pydantic import BaseModel, Field

from visual_service.settings import AnalysisMode


class PlanRequest(BaseModel):
    contentId: str
    mediaPath: str
    durationMs: int = Field(gt=0)


class VideoAnalyzeRequest(BaseModel):
    runId: str
    contentId: str
    mediaPath: str
    durationMs: int = Field(gt=0)
    analysisMode: AnalysisMode | None = None
