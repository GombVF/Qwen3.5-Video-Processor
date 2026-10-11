"""Fail-fast verification for the supported WSL2/ROCm execution stack.

This intentionally executes a small FLA gated-delta kernel. Importing ``fla``
is not enough: Triton compilation or the GPU architecture may still fail on
the first real call.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def main() -> None:
    import av
    import torch
    import triton
    from fla.ops.gated_delta_rule import (
        chunk_gated_delta_rule,
        fused_recurrent_gated_delta_rule,
    )

    if platform.system() != "Linux":
        raise SystemExit("This verifier must be executed inside WSL/Linux")
    if not (os.getenv("WSL_INTEROP") or "microsoft" in platform.release().lower()):
        raise SystemExit("WSL2 was not detected")
    if not Path("/dev/dxg").exists():
        raise SystemExit("/dev/dxg is missing; WSL GPU passthrough is unavailable")
    if not torch.cuda.is_available():
        raise SystemExit("PyTorch cannot access the ROCm GPU")
    if not torch.version.hip:
        raise SystemExit("The installed PyTorch build does not expose a HIP runtime")
    if not torch.cuda.is_bf16_supported():
        raise SystemExit("The GPU/PyTorch stack does not report BF16 support")

    if av.__version__ != "18.1.0":
        raise SystemExit(
            "PyAV does not match the tested uniform-sampler version: "
            f"version={av.__version__}, expected=18.1.0"
        )
    media_probe = Path(__file__).resolve().parents[1] / "data" / "test_videos" / "video_12_sec.mp4"
    if media_probe.is_file():
        container = av.open(str(media_probe))
        try:
            frame = next(container.decode(video=0), None)
            if frame is None:
                raise SystemExit(f"PyAV decoded no frames from {media_probe}")
        finally:
            container.close()

    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    shape = (1, 64, 2, 64)  # batch, sequence, heads, head dimension
    q = torch.randn(shape, device=device, dtype=dtype)
    k = torch.randn(shape, device=device, dtype=dtype)
    v = torch.randn(shape, device=device, dtype=dtype)
    g = -torch.rand(shape[:-1], device=device, dtype=torch.float32)
    beta = torch.sigmoid(torch.randn(shape[:-1], device=device, dtype=torch.float32))

    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        output, final_state = chunk_gated_delta_rule(
            q,
            k,
            v,
            g=g,
            beta=beta,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
    torch.cuda.synchronize()
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)

    if output.shape != q.shape:
        raise SystemExit(f"Unexpected FLA output shape: {tuple(output.shape)}")
    if final_state is None:
        raise SystemExit("FLA did not return the requested final state")
    if not torch.isfinite(output).all().item():
        raise SystemExit("FLA produced non-finite output")

    recurrent_shape = (1, 1, 2, 64)
    rq = torch.randn(recurrent_shape, device=device, dtype=dtype)
    rk = torch.randn(recurrent_shape, device=device, dtype=dtype)
    rv = torch.randn(recurrent_shape, device=device, dtype=dtype)
    rg = -torch.rand(recurrent_shape[:-1], device=device, dtype=torch.float32)
    rbeta = torch.sigmoid(
        torch.randn(recurrent_shape[:-1], device=device, dtype=torch.float32)
    )
    torch.cuda.synchronize()
    recurrent_started = time.perf_counter()
    with torch.inference_mode():
        recurrent_output, recurrent_state = fused_recurrent_gated_delta_rule(
            rq,
            rk,
            rv,
            g=rg,
            beta=rbeta,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
    torch.cuda.synchronize()
    recurrent_ms = round((time.perf_counter() - recurrent_started) * 1000, 2)
    if recurrent_output.shape != rq.shape or recurrent_state is None:
        raise SystemExit("FLA recurrent gated-delta smoke test returned invalid output")
    if not torch.isfinite(recurrent_output).all().item():
        raise SystemExit("FLA recurrent gated-delta kernel produced non-finite output")

    report = {
        "wsl": True,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "triton": triton.__version__,
        "flashLinearAttention": package_version("flash-linear-attention"),
        "pyav": av.__version__,
        "bf16": True,
        "flaKernelExecuted": True,
        "flaChunkFirstCallMs": elapsed_ms,
        "flaRecurrentFirstCallMs": recurrent_ms,
        "outputShape": list(output.shape),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
