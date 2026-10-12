import os
import platform
import re
from enum import Enum
from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PROMPT_VERSION = "visual-analysis-qwen35-v6"
SCHEMA_VERSION = "6.0"


class FramePolicy(str, Enum):
    ADAPTIVE = "ADAPTIVE"
    FIXED_MAX = "FIXED_MAX"

_WINDOWS_DRIVE_PATH = re.compile(r"^(?P<drive>[A-Za-z]):[\\/](?P<rest>.*)$")


def is_wsl() -> bool:
    """Return whether the process is running inside WSL, without shell calls."""

    return platform.system() == "Linux" and (
        bool(os.getenv("WSL_INTEROP")) or "microsoft" in platform.release().lower()
    )


def windows_path_to_wsl(value: str) -> str:
    """Translate a local ``E:/...`` path to ``/mnt/e/...`` under WSL.

    Hugging Face repository IDs such as ``Qwen/Qwen3.5-4B`` do not match the
    drive-letter pattern and are returned unchanged.
    """

    match = _WINDOWS_DRIVE_PATH.match(value)
    if not match:
        return value
    drive = match.group("drive").lower()
    rest = match.group("rest").replace("\\", "/")
    return f"/mnt/{drive}/{rest}"


class AnalysisMode(str, Enum):
    SEMANTIC_ONLY = "SEMANTIC_ONLY"
    SEMANTIC_AND_OCR = "SEMANTIC_AND_OCR"


class Settings(BaseSettings):
    """Single supported Qwen3.5 configuration.

    Environment variables use the VISUAL_ prefix. Windows drive-letter paths
    are translated automatically when this process runs under WSL2.
    """

    model_config = SettingsConfigDict(
        env_prefix="VISUAL_",
        extra="ignore",
        validate_default=True,
    )

    host: str = "127.0.0.1"
    port: int = 8001
    media_root: Path = Path("E:/personal-media-search/data/media")

    qwen_model_id: str = "E:/Trabajo/Proyectos/Qwen/Qwen3.5-4B"
    qwen_dtype: str = "bfloat16"
    qwen_attn_implementation: str = "sdpa"
    qwen_local_files_only: bool = True
    qwen_enable_thinking: bool = False
    qwen_max_concurrency: int = 1
    qwen_force_fla_binding: bool = Field(default_factory=is_wsl)
    qwen_require_fla: bool = Field(default_factory=is_wsl)
    qwen_prompt_version: str = PROMPT_VERSION
    qwen_schema_version: str = SCHEMA_VERSION
    analysis_language: str = "Spanish"
    analysis_mode: AnalysisMode = AnalysisMode.SEMANTIC_AND_OCR
    grounded_text_understanding_enabled: bool = True

    semantic_window_duration_ms: int = 12_000
    semantic_window_overlap_ms: int = 2_000
    semantic_fps: float = 2.0
    semantic_min_frames: int = 4
    semantic_max_frames: int = 24
    semantic_max_pixels: int = 392 * 700
    semantic_max_new_tokens: int = 512
    semantic_frame_policy: FramePolicy = FramePolicy.FIXED_MAX

    global_overview_fps: float = 1.0
    global_overview_min_frames: int = 8
    global_overview_max_frames: int = 24
    global_overview_max_pixels: int = 392 * 700
    global_overview_max_new_tokens: int = 512
    global_overview_frame_policy: FramePolicy = FramePolicy.FIXED_MAX

    video_decode_max_attempts: int = 2
    video_decode_retry_delay_ms: int = 250

    ocr_fps: float = 1.0
    ocr_max_frames_total: int = 120
    ocr_max_pixels: int = 504 * 896
    ocr_batch_size: int = 4
    ocr_max_new_tokens: int = 384

    grounded_synthesis_max_new_tokens: int = 1024
    max_evidence_references_per_synthesis_item: int = 6

    max_summary_sentences: int = 3
    max_observations_per_window: int = 8
    max_visible_texts_per_image: int = 12
    max_objects_per_window: int = 12
    max_global_topics: int = 10
    max_final_communicated_ideas: int = 6
    max_final_themes: int = 6

    consolidation_adjacent_threshold: float = 0.78
    consolidation_nonadjacent_threshold: float = 0.92
    consolidation_visible_text_threshold: float = 0.94
    consolidation_fragment_neighborhood: int = 3

    @field_validator("media_root", mode="before")
    @classmethod
    def normalize_media_root(cls, value: str | Path) -> Path:
        raw = str(value)
        return Path(windows_path_to_wsl(raw) if is_wsl() else raw)

    @field_validator("qwen_model_id", mode="before")
    @classmethod
    def normalize_model_id(cls, value: str) -> str:
        return windows_path_to_wsl(value) if is_wsl() else value

    @field_validator("qwen_dtype")
    @classmethod
    def validate_dtype(cls, value: str) -> str:
        if value != "bfloat16":
            raise ValueError("Qwen3.5-4B is supported only with bfloat16 in this project")
        return value

    @field_validator("qwen_prompt_version")
    @classmethod
    def validate_prompt_version(cls, value: str) -> str:
        allowed = {PROMPT_VERSION, "visual-analysis-qwen35-v5", "visual-analysis-qwen35-v4", "visual-analysis-qwen35-v3", "visual-analysis-qwen35-v2"}
        if value not in allowed:
            raise ValueError(f"Only {sorted(allowed)!r} are supported")
        return value

    @field_validator("qwen_schema_version")
    @classmethod
    def validate_schema_version(cls, value: str) -> str:
        if value != SCHEMA_VERSION:
            raise ValueError(f"Only schema {SCHEMA_VERSION!r} is supported")
        return value

    @field_validator("analysis_mode", mode="before")
    @classmethod
    def validate_analysis_mode(cls, value: str | AnalysisMode) -> AnalysisMode:
        if isinstance(value, AnalysisMode):
            return value
        try:
            return AnalysisMode(value)
        except ValueError:
            allowed = ", ".join(m.value for m in AnalysisMode)
            raise ValueError(f"Unsupported analysis mode {value!r}. Allowed: {allowed}")

    @model_validator(mode="after")
    def validate_runtime_budget(self) -> "Settings":
        if self.qwen_enable_thinking:
            raise ValueError("Thinking must remain disabled for deterministic JSON extraction")
        if self.qwen_max_concurrency != 1:
            raise ValueError("GPU inference concurrency must remain exactly 1")
        if self.semantic_window_overlap_ms >= self.semantic_window_duration_ms:
            raise ValueError("Semantic overlap must be smaller than the window duration")
        if self.ocr_batch_size < 1:
            raise ValueError("OCR batch size must be positive")
        if self.ocr_max_frames_total < self.ocr_batch_size:
            raise ValueError("OCR frame budget must be at least one batch")
        if self.video_decode_max_attempts < 1:
            raise ValueError("Video decode attempts must be at least one")
        if self.video_decode_retry_delay_ms < 0:
            raise ValueError("Video decode retry delay cannot be negative")
        if self.semantic_frame_policy == FramePolicy.FIXED_MAX and self.semantic_max_frames < 2:
            raise ValueError("FIXED_MAX semantic policy requires max_frames >= 2")
        if self.global_overview_frame_policy == FramePolicy.FIXED_MAX and self.global_overview_max_frames < 2:
            raise ValueError("FIXED_MAX global overview policy requires max_frames >= 2")
        if self.grounded_synthesis_max_new_tokens < 1:
            raise ValueError("Grounded synthesis max_new_tokens must be positive")
        if self.max_evidence_references_per_synthesis_item < 1:
            raise ValueError("Max evidence references per synthesis item must be positive")
        return self


settings = Settings()
