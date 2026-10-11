from __future__ import annotations

import importlib.metadata
import json
import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any

from visual_service.settings import PROMPT_VERSION, Settings

logger = logging.getLogger(__name__)


@dataclass
class KernelDiagnostics:
    flash_linear_attention_available: bool = False
    flash_linear_attention_active: bool = False
    flash_linear_attention_binding: str = "reference"
    flash_linear_attention_chunk_calls: int = 0
    flash_linear_attention_recurrent_calls: int = 0
    causal_conv1d_available: bool = False
    causal_conv1d_active: bool = False
    linear_attention_backend: str = "reference"
    full_attention_backend: str = "sdpa"
    fallback_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "flashLinearAttentionAvailable": self.flash_linear_attention_available,
            "flashLinearAttentionActive": self.flash_linear_attention_active,
            "flashLinearAttentionBinding": self.flash_linear_attention_binding,
            "flashLinearAttentionExecutionVerified": (
                self.flash_linear_attention_chunk_calls
                + self.flash_linear_attention_recurrent_calls
            ) > 0,
            "flashLinearAttentionChunkCalls": self.flash_linear_attention_chunk_calls,
            "flashLinearAttentionRecurrentCalls": self.flash_linear_attention_recurrent_calls,
            "causalConv1dAvailable": self.causal_conv1d_available,
            "causalConv1dActive": self.causal_conv1d_active,
            "linearAttentionBackend": self.linear_attention_backend,
            "fullAttentionBackend": self.full_attention_backend,
            "fallbackReasons": list(self.fallback_reasons),
        }


@dataclass
class QwenRuntime:
    model: Any
    processor: Any
    device_name: str
    model_id: str
    model_revision: str
    prompt_version: str
    configured_dtype: str
    actual_dtype_distribution: dict[str, int]
    attention_implementation: str
    thinking_enabled: bool
    torch_version: str
    hip_version: str
    transformers_version: str
    toolkit_version: str
    video_backend: str
    loaded_at: datetime
    kernel_diagnostics: KernelDiagnostics = field(default_factory=KernelDiagnostics)
    _closed: bool = field(default=False, repr=False)

    async def close(self) -> None:
        import torch

        if self._closed:
            return
        self._closed = True
        del self.model
        del self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("qwen35_runtime_closed")


async def load_qwen_runtime(config: Settings) -> QwenRuntime:
    import torch

    try:
        from transformers import AutoModelForMultimodalLM, AutoProcessor
        import transformers
    except ImportError as exc:
        raise RuntimeError(
            "Qwen3.5 requires a recent Transformers build exposing "
            "AutoModelForMultimodalLM. Run scripts/setup-wsl.sh."
        ) from exc

    allowed_prompts = {PROMPT_VERSION, "visual-analysis-qwen35-v3", "visual-analysis-qwen35-v2"}
    if config.qwen_prompt_version not in allowed_prompts:
        raise ValueError(f"Only prompts {sorted(allowed_prompts)!r} are supported")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm is unavailable; Qwen3.5-4B must load on the GPU")

    logger.info("qwen35_load_start", extra={"data": {
        "modelId": config.qwen_model_id,
        "dtype": config.qwen_dtype,
        "attentionImplementation": config.qwen_attn_implementation,
        "thinkingEnabled": config.qwen_enable_thinking,
    }})
    started = time.perf_counter()
    dtype = getattr(torch, config.qwen_dtype, None)
    if dtype is None:
        raise ValueError(f"Unknown torch dtype: {config.qwen_dtype!r}")

    # Fail before loading several GiB of weights when the explicit PyAV
    # sampling path is unavailable.
    from visual_service.video.decoder import validate_video_backend_compatibility

    video_backend = validate_video_backend_compatibility()
    kernel_diag = _configure_qwen35_kernels(config, config.qwen_attn_implementation)

    processor = AutoProcessor.from_pretrained(
        config.qwen_model_id,
        local_files_only=config.qwen_local_files_only,
    )
    model = AutoModelForMultimodalLM.from_pretrained(
        config.qwen_model_id,
        dtype=dtype,
        device_map={"": "cuda:0"},
        attn_implementation=config.qwen_attn_implementation,
        low_cpu_mem_usage=True,
        local_files_only=config.qwen_local_files_only,
    ).eval()

    model_type = getattr(model.config, "model_type", "")
    if model_type != "qwen3_5":
        raise RuntimeError(
            f"Configured checkpoint is {model_type!r}, expected the Qwen3.5 architecture"
        )
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True

    devices = {parameter.device.type for parameter in model.parameters()}
    if devices != {"cuda"}:
        raise RuntimeError(f"Model was not loaded entirely on the ROCm GPU: {sorted(devices)}")

    actual_dtypes = dict(Counter(str(parameter.dtype) for parameter in model.parameters()))
    device_name = torch.cuda.get_device_name(0)
    hip_version = getattr(torch.version, "hip", None) or "N/A"
    toolkit_version = _package_version("qwen-vl-utils")
    elapsed = time.perf_counter() - started
    allocated = torch.cuda.memory_allocated(0)

    logger.info("qwen35_load_complete", extra={"data": {
        "elapsedS": round(elapsed, 2),
        "allocatedVramBytes": allocated,
        "device": device_name,
        "parameterDtypes": actual_dtypes,
        "kernelDiagnostics": kernel_diag.to_dict(),
    }})

    return QwenRuntime(
        model=model,
        processor=processor,
        device_name=device_name,
        model_id=config.qwen_model_id,
        model_revision=_resolve_revision(config.qwen_model_id),
        prompt_version=config.qwen_prompt_version,
        configured_dtype=config.qwen_dtype,
        actual_dtype_distribution=actual_dtypes,
        attention_implementation=config.qwen_attn_implementation,
        thinking_enabled=config.qwen_enable_thinking,
        torch_version=torch.__version__,
        hip_version=hip_version,
        transformers_version=transformers.__version__,
        toolkit_version=toolkit_version,
        video_backend=video_backend,
        loaded_at=datetime.now(timezone.utc),
        kernel_diagnostics=kernel_diag,
    )


def _configure_qwen35_kernels(config: Settings, attn_impl: str) -> KernelDiagnostics:
    """Select and instrument Qwen3.5's optional optimized kernels.

    Current Transformers releases decorate the gated-delta functions with a
    package fallback. Some releases can import ``fla`` but fail to resolve its
    nested operators and silently retain the slow Torch reference functions.
    WSL can therefore request an explicit, API-compatible binding. The wrappers
    below also count real calls so benchmark output proves execution rather
    than merely reporting package availability.
    """

    diagnostics = KernelDiagnostics(full_attention_backend=attn_impl)

    try:
        from fla.ops.gated_delta_rule import (
            chunk_gated_delta_rule,
            fused_recurrent_gated_delta_rule,
        )

        diagnostics.flash_linear_attention_available = True
    except ImportError as exc:
        diagnostics.fallback_reasons.append(
            f"flash-linear-attention package or gated-delta operators unavailable: {exc}"
        )
        if config.qwen_require_fla:
            raise RuntimeError(diagnostics.fallback_reasons[-1]) from exc
        chunk_gated_delta_rule = None
        fused_recurrent_gated_delta_rule = None
    except Exception as exc:
        diagnostics.fallback_reasons.append(
            f"flash-linear-attention installed but operators are unusable: {exc}"
        )
        if config.qwen_require_fla:
            raise RuntimeError(diagnostics.fallback_reasons[-1]) from exc
        chunk_gated_delta_rule = None
        fused_recurrent_gated_delta_rule = None

    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5
    except Exception as exc:
        diagnostics.fallback_reasons.append(
            f"could not inspect Qwen3.5 kernel bindings: {exc}"
        )
        if config.qwen_require_fla:
            raise RuntimeError(diagnostics.fallback_reasons[-1]) from exc
        return diagnostics

    if diagnostics.flash_linear_attention_available:
        if config.qwen_force_fla_binding:
            modeling_qwen3_5.torch_chunk_gated_delta_rule = _tracked_fla_call(
                chunk_gated_delta_rule,
                diagnostics,
                recurrent=False,
            )
            modeling_qwen3_5.torch_recurrent_gated_delta_rule = _tracked_fla_call(
                fused_recurrent_gated_delta_rule,
                diagnostics,
                recurrent=True,
            )
            diagnostics.flash_linear_attention_active = True
            diagnostics.flash_linear_attention_binding = "explicit"
            diagnostics.linear_attention_backend = "fla"
        else:
            chunk_modules = _callable_modules(
                getattr(modeling_qwen3_5, "torch_chunk_gated_delta_rule", None)
            )
            recurrent_modules = _callable_modules(
                getattr(modeling_qwen3_5, "torch_recurrent_gated_delta_rule", None)
            )
            auto_selected = any(name.startswith("fla.") for name in chunk_modules) and any(
                name.startswith("fla.") for name in recurrent_modules
            )
            diagnostics.flash_linear_attention_active = auto_selected
            if auto_selected:
                diagnostics.flash_linear_attention_binding = "transformers-auto"
                diagnostics.linear_attention_backend = "fla"
            else:
                diagnostics.fallback_reasons.append(
                    "FLA imports successfully but Transformers selected its Torch reference "
                    "gated-delta functions; set VISUAL_QWEN_FORCE_FLA_BINDING=true under WSL"
                )

    try:
        import causal_conv1d  # noqa: F401

        diagnostics.causal_conv1d_available = True
        conv_modules = _callable_modules(
            getattr(modeling_qwen3_5, "causal_conv1d_fn", None)
        )
        diagnostics.causal_conv1d_active = any(
            name.startswith("causal_conv1d") for name in conv_modules
        )
        if not diagnostics.causal_conv1d_active:
            diagnostics.fallback_reasons.append(
                "causal_conv1d is installed but Transformers selected the Torch fallback"
            )
    except ImportError:
        diagnostics.fallback_reasons.append(
            "causal_conv1d is unavailable on this ROCm stack; Torch convolution fallback is active"
        )

    if config.qwen_require_fla and not diagnostics.flash_linear_attention_active:
        raise RuntimeError(
            "VISUAL_QWEN_REQUIRE_FLA=true but Qwen3.5 did not select FLA gated-delta kernels"
        )
    return diagnostics


def _tracked_fla_call(function: Any, diagnostics: KernelDiagnostics, *, recurrent: bool):
    @wraps(function)
    def tracked(*args, **kwargs):
        if recurrent:
            diagnostics.flash_linear_attention_recurrent_calls += 1
        else:
            diagnostics.flash_linear_attention_chunk_calls += 1
        return function(*args, **kwargs)

    return tracked


def _callable_modules(function: Any) -> set[str]:
    """Collect callable module names reachable through decorator closures."""

    modules: set[str] = set()
    pending = [function]
    seen: set[int] = set()
    while pending:
        candidate = pending.pop()
        if not callable(candidate) or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        module_name = getattr(candidate, "__module__", "")
        if module_name:
            modules.add(module_name)
        for cell in getattr(candidate, "__closure__", None) or ():
            try:
                value = cell.cell_contents
            except ValueError:
                continue
            if callable(value):
                pending.append(value)
    return modules


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _resolve_revision(model_id: str) -> str:
    model_path = Path(model_id)
    ref_path = model_path / "refs" / "main"
    if ref_path.is_file():
        return ref_path.read_text(encoding="utf-8").strip()
    config_path = model_path / "config.json"
    if config_path.is_file():
        config_data = json.loads(config_path.read_text(encoding="utf-8"))
        return str(config_data.get("_commit_hash", "local"))
    return "remote-or-local-unresolved"
