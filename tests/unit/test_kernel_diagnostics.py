"""Tests for WSL/ROCm kernel diagnostics and execution accounting."""

from visual_service.model.lifecycle import (
    KernelDiagnostics,
    _callable_modules,
    _tracked_fla_call,
)


def test_default_diagnostics():
    diagnostics = KernelDiagnostics()
    assert diagnostics.flash_linear_attention_available is False
    assert diagnostics.flash_linear_attention_active is False
    assert diagnostics.flash_linear_attention_binding == "reference"
    assert diagnostics.causal_conv1d_available is False
    assert diagnostics.causal_conv1d_active is False
    assert diagnostics.linear_attention_backend == "reference"
    assert diagnostics.full_attention_backend == "sdpa"
    assert diagnostics.fallback_reasons == []


def test_to_dict_distinguishes_selection_from_execution():
    diagnostics = KernelDiagnostics(
        flash_linear_attention_available=True,
        flash_linear_attention_active=True,
        flash_linear_attention_binding="explicit",
    )
    data = diagnostics.to_dict()
    assert data["flashLinearAttentionActive"] is True
    assert data["flashLinearAttentionExecutionVerified"] is False

    diagnostics.flash_linear_attention_chunk_calls = 2
    data = diagnostics.to_dict()
    assert data["flashLinearAttentionExecutionVerified"] is True
    assert data["flashLinearAttentionChunkCalls"] == 2


def test_to_dict_keys():
    expected_keys = {
        "flashLinearAttentionAvailable",
        "flashLinearAttentionActive",
        "flashLinearAttentionBinding",
        "flashLinearAttentionExecutionVerified",
        "flashLinearAttentionChunkCalls",
        "flashLinearAttentionRecurrentCalls",
        "causalConv1dAvailable",
        "causalConv1dActive",
        "linearAttentionBackend",
        "fullAttentionBackend",
        "fallbackReasons",
    }
    assert set(KernelDiagnostics().to_dict()) == expected_keys


def test_tracked_fla_call_records_real_invocation():
    diagnostics = KernelDiagnostics()

    def fake_kernel(value, *, scale=1):
        return value * scale

    wrapped = _tracked_fla_call(fake_kernel, diagnostics, recurrent=False)
    assert wrapped(3, scale=4) == 12
    assert diagnostics.flash_linear_attention_chunk_calls == 1
    assert diagnostics.flash_linear_attention_recurrent_calls == 0


def test_tracked_recurrent_call_is_counted_separately():
    diagnostics = KernelDiagnostics()

    def fake_kernel(value):
        return value

    wrapped = _tracked_fla_call(fake_kernel, diagnostics, recurrent=True)
    assert wrapped("ok") == "ok"
    assert diagnostics.flash_linear_attention_recurrent_calls == 1


def test_callable_modules_inspects_decorator_closures():
    def fla_kernel():
        return "fla"

    fla_kernel.__module__ = "fla.ops.gated_delta_rule"

    def wrapper():
        return fla_kernel()

    modules = _callable_modules(wrapper)
    assert "fla.ops.gated_delta_rule" in modules

