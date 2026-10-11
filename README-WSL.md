# Qwen 3.5 visual service on WSL2/ROCm

This is the supported high-performance environment for the AMD Radeon RX 9070 XT.
The source, test videos, and model may remain on the Windows `E:` drive. The
Python virtual environment is intentionally stored on WSL's native filesystem.

## Setup

From Ubuntu 24.04 under WSL2:

```bash
cd /mnt/e/Trabajo/Proyectos/Python/Qwen3.5
cp .env-wsl.example .env-wsl
bash scripts/setup-wsl.sh
```

The default virtual environment is:

```text
~/.venvs/Qwen3.5/.venv-wsl
```

## Activate and verify

```bash
cd /mnt/e/Trabajo/Proyectos/Python/Qwen3.5

source /home/ville/.venvs/Qwen3.5/.venv-wsl/bin/activate

set -a
source .env-wsl
set +a
```

Verify that the environment is correct:

```bash
which python
python -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available())"
```

The verifier script checks ROCm GPU visibility, BF16 support, the pinned PyAV API,
Triton, and an actual FLA gated-delta GPU call:

```bash
python scripts/verify_wsl_acceleration.py
```

Import success alone is not treated as proof that the optimized kernel can execute.

A convenience activation script is provided. It derives the default environment
from `$HOME` and allows an override via `VENV_DIR`:

```bash
source scripts/activate-wsl.sh
```

## Run

```bash
cd /mnt/e/Trabajo/Proyectos/Python/Qwen3.5

source /home/ville/.venvs/Qwen3.5/.venv-wsl/bin/activate

set -a
source .env-wsl
set +a

python -m uvicorn visual_service.main:app --host 127.0.0.1 --port 8001
```

## Benchmark

```bash
cd /mnt/e/Trabajo/Proyectos/Python/Qwen3.5

source /home/ville/.venvs/Qwen3.5/.venv-wsl/bin/activate

set -a
source .env-wsl
set +a

python scripts/benchmark_e2e.py \
  --output data/benchmark-qwen35-4b-fixed-frames.json
```

The benchmark records whether FLA was selected and how many optimized chunk
and recurrent calls were actually observed. A package being installed or
importable is not sufficient evidence of activation. By default it first
warms the global-overview, semantic-video, and OCR-image shapes. This separates
one-time Triton compilation from the measured long-running service latency.

The first warm-up after changing PyTorch, Triton, FLA, Transformers, shapes, or
pixel budgets can take several minutes. That cost is recorded under `warmup`
but excluded from each video's `metrics.totalMs`.

## Video sampling strategy

All semantic inputs use the same explicit `pyav-uniform` strategy. The service
plans uniformly spaced timestamps first, seeks to each logical window, decodes
only until those frames have been collected, and keeps the selected PIL images
in RAM. It then gives that in-memory frame list to `qwen-vl-utils`; the toolkit
may resize and tensorize it, but it may not choose a different frame count.

### Frame count policy

The `video_frame_policy` setting controls how many frames are selected per pass:

- **FIXED_MAX** (default for semantic and global-overview): always selects the
  configured `max_frames` for that pass, regardless of window duration. This
  eliminates shape-dependent Triton/FLA compilation spikes during measured runs.
  For short windows, the configured frame count is distributed throughout the
  available duration. Repeated decoded frames are acceptable when the source
  genuinely does not provide enough distinct frames.

- **ADAPTIVE**: retains the original behavior where frame count scales with
  duration and configured FPS, clamped between min and max. Required for
  controlled A/B benchmarks comparing different frame counts.

In `ADAPTIVE` mode, `fps` determines the frame count. In `FIXED_MAX` mode,
`max_frames` is authoritative and `fps` is ignored for count selection.

The global overview uses the same path with a sparse, capped budget. Long
videos are not written as temporary clips: they remain one file and are
analyzed as overlapping 12-second logical windows. OCR is still a separate
high-resolution PyAV pass. This preserves the two-pass semantic/OCR design
while preventing the 65-second overview from decoding the entire video through
the deprecated torchvision video reader.

`av==18.1.0` remains pinned as the known-good WSL baseline. Any decoder or PyAV
upgrade must be benchmarked against the same videos before changing the pin.
