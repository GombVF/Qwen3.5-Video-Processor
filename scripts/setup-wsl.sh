#!/usr/bin/env bash
# setup-wsl.sh — Create the Qwen 3.5-4B environment inside WSL2.
#
# Supported host distributions for the pinned ROCm 7.2.1 wheel set:
#   - Ubuntu 24.04 + Python 3.12 (cp312)
#   - Ubuntu 22.04 + Python 3.10 (cp310)
#
# Usage from the project root:
#   bash scripts/setup-wsl.sh
#
# Optional overrides:
#   VENV_DIR=/path/to/venv bash scripts/setup-wsl.sh
#   TRANSFORMERS_SPEC='git+https://github.com/huggingface/transformers.git@<commit>' \
#     bash scripts/setup-wsl.sh

set -Eeuo pipefail

readonly SCRIPT_NAME="$(basename "$0")"

log()  { printf '\n[%s] %s\n' "$1" "$2"; }
info() { log INFO "$1"; }
ok()   { log OK "$1"; }
warn() { log WARN "$1" >&2; }
die()  { log ERROR "$1" >&2; exit 1; }

on_error() {
    local exit_code=$?
    printf '\n[ERROR] %s failed at line %s (exit %s).\n' \
        "$SCRIPT_NAME" "${BASH_LINENO[0]}" "$exit_code" >&2
    printf '[ERROR] Command: %s\n' "$BASH_COMMAND" >&2
    printf '[ERROR] Re-run after correcting the reported error; completed steps are reusable.\n' >&2
    exit "$exit_code"
}
trap on_error ERR

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SCRIPT_DIR/../pyproject.toml" ]]; then
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
elif [[ -f "$SCRIPT_DIR/pyproject.toml" ]]; then
    PROJECT_ROOT="$SCRIPT_DIR"
else
    die "Could not find pyproject.toml beside this script or in its parent directory. Put the script in <project>/scripts or <project>."
fi

TRANSFORMERS_SPEC="${TRANSFORMERS_SPEC:-git+https://github.com/huggingface/transformers.git@main}"
ROCM_VERSION="7.2.1"
ROCM_WHEEL_BASE="https://repo.radeon.com/rocm/manylinux/rocm-rel-${ROCM_VERSION}"
AMDGPU_INSTALL_RELEASE="7.2"
AMDGPU_INSTALL_VERSION="7.2.70200"

# ── 1. Validate the WSL/Linux host ─────────────────────────────────────────

[[ -r /etc/os-release ]] || die "Cannot read /etc/os-release; this installer expects Ubuntu under WSL2."
# shellcheck disable=SC1091
source /etc/os-release

[[ "${ID:-}" == "ubuntu" ]] || die "Unsupported distribution '${ID:-unknown}'. Use Ubuntu 22.04 or 24.04 under WSL2."

case "${VERSION_ID:-}" in
    24.04)
        PYTHON_VERSION="3.12"
        PYTHON_TAG="cp312"
        UBUNTU_CODENAME="noble"
        ;;
    22.04)
        PYTHON_VERSION="3.10"
        PYTHON_TAG="cp310"
        UBUNTU_CODENAME="jammy"
        ;;
    *)
        die "Ubuntu ${VERSION_ID:-unknown} is not supported by this pinned ROCm ${ROCM_VERSION} setup. Install an Ubuntu 24.04 WSL distro (recommended) or Ubuntu 22.04."
        ;;
esac

PYTHON_BIN="python${PYTHON_VERSION}"
ok "Detected Ubuntu ${VERSION_ID}; using ${PYTHON_BIN} (${PYTHON_TAG})."

# Python environments perform poorly and can fail while creating ensurepip on
# Windows-mounted DrvFS paths such as /mnt/c or /mnt/e. Keep source, media, and
# models on the Windows drive, but default the Linux venv to WSL's ext4 disk.
if [[ "$PROJECT_ROOT" == /mnt/* ]]; then
    PROJECT_KEY="$(printf '%s' "$(basename "$PROJECT_ROOT")" | tr -c 'A-Za-z0-9._-' '_')"
    DEFAULT_VENV_DIR="$HOME/.venvs/$PROJECT_KEY/.venv-wsl"
else
    DEFAULT_VENV_DIR="$PROJECT_ROOT/.venv-wsl"
fi
VENV_DIR="${VENV_DIR:-$DEFAULT_VENV_DIR}"
LEGACY_PROJECT_VENV="$PROJECT_ROOT/.venv-wsl"

info "Qwen 3.5-4B WSL2 setup"
printf 'Project root: %s\n' "$PROJECT_ROOT"
printf 'Virtual env:  %s\n' "$VENV_DIR"

if [[ "$VENV_DIR" != "$LEGACY_PROJECT_VENV" && -e "$LEGACY_PROJECT_VENV" ]]; then
    warn "Ignoring the project-local environment at $LEGACY_PROJECT_VENV. It is on DrvFS and may be partial. The native WSL environment above will be used instead."
fi

# WSL exposes the AMD GPU through DirectX GPU paravirtualization. /dev/kfd is
# a native-Linux ROCm device and is not the correct WSL gate.
[[ -e /dev/dxg ]] || die "/dev/dxg is missing. Windows-to-WSL GPU passthrough is not active. Run 'wsl --shutdown' from PowerShell after installing a compatible AMD Windows driver, then retry."
ok "WSL GPU device /dev/dxg is available."

# ── 2. Ensure the distro's supported Python is present ─────────────────────

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1 \
    || ! dpkg-query -W -f='${Status}' "${PYTHON_BIN}-venv" 2>/dev/null | grep -q 'install ok installed'; then
    info "Installing ${PYTHON_BIN} and its venv package from Ubuntu repositories."
    sudo apt-get update
    sudo apt-get install -y "${PYTHON_BIN}" "${PYTHON_BIN}-venv" "${PYTHON_BIN}-dev" git
else
    ok "$($PYTHON_BIN --version) and ${PYTHON_BIN}-venv are available."
fi

if ! command -v git >/dev/null 2>&1; then
    info "Installing git, required by the default Transformers source."
    sudo apt-get update
    sudo apt-get install -y git
fi

# PyTorch's Linux wheel loads the GCC OpenMP runtime during `import torch`.
# Minimal WSL images do not always include it.
if ! dpkg-query -W -f='${Status}' libgomp1 2>/dev/null | grep -q 'install ok installed'; then
    info "Installing libgomp1, required by PyTorch's native libraries."
    sudo apt-get update
    sudo apt-get install -y libgomp1
else
    ok "libgomp1 is available."
fi

LD_CONFIG_CACHE="$(ldconfig -p)"
grep -F 'libgomp.so.1' <<<"$LD_CONFIG_CACHE" >/dev/null \
    || die "libgomp1 is installed but libgomp.so.1 is not visible to the dynamic linker. Run 'sudo ldconfig' and retry."

# /dev/dxg is only the WSL GPU transport. PyTorch also needs the matching ROCm
# user-space stack (including libroctx64.so.4) inside this Ubuntu distribution.
rocm_userspace_ready() {
    local linker_cache
    linker_cache="$(ldconfig -p)"
    command -v rocminfo >/dev/null 2>&1 \
        && grep -F 'libroctx64.so.4' <<<"$linker_cache" >/dev/null
}

if rocm_userspace_ready; then
    ok "ROCm WSL user-space libraries are available."
else
    info "Installing the official ROCm ${AMDGPU_INSTALL_RELEASE} WSL user-space stack. This can take several minutes."
    sudo apt-get update
    sudo apt-get install -y wget ca-certificates

    if ! command -v amdgpu-install >/dev/null 2>&1; then
        AMDGPU_DEB_DIR="$(mktemp -d)"
        AMDGPU_DEB="$AMDGPU_DEB_DIR/amdgpu-install.deb"
        AMDGPU_DEB_URL="https://repo.radeon.com/amdgpu-install/${AMDGPU_INSTALL_RELEASE}/ubuntu/${UBUNTU_CODENAME}/amdgpu-install_${AMDGPU_INSTALL_VERSION}-1_all.deb"

        wget --progress=bar:force:noscroll -O "$AMDGPU_DEB" "$AMDGPU_DEB_URL"
        sudo apt-get install -y "$AMDGPU_DEB"
    fi

    sudo amdgpu-install -y --usecase=wsl,rocm --no-dkms
    sudo ldconfig
fi

rocm_userspace_ready \
    || die "The ROCm WSL installation completed but rocminfo or libroctx64.so.4 is still unavailable. Run 'sudo amdgpu-install -y --usecase=wsl,rocm --no-dkms', then 'sudo ldconfig', and retry."

ROCMINFO_OUTPUT="$(rocminfo 2>/dev/null)"
GPU_AGENT="$(awk '/^[[:space:]]*Name:[[:space:]]*gfx/{print $2; exit}' <<<"$ROCMINFO_OUTPUT")"
[[ -n "$GPU_AGENT" ]] || die "rocminfo did not report a gfx GPU agent after the ROCm WSL installation."
ok "rocminfo sees GPU agent ${GPU_AGENT}."

# ── 3. Create or validate .venv-wsl ────────────────────────────────────────

if [[ -e "$VENV_DIR" && ! -x "$VENV_DIR/bin/python" ]]; then
    die "$VENV_DIR exists but is not a valid Linux virtual environment. Rename or remove only that directory, then rerun this script."
fi

if [[ ! -d "$VENV_DIR" ]]; then
    VENV_PARENT="$(dirname "$VENV_DIR")"
    mkdir -p "$VENV_PARENT"

    "$PYTHON_BIN" -m ensurepip --version >/dev/null \
        || die "${PYTHON_BIN} cannot run ensurepip. Reinstall ${PYTHON_BIN}-venv and rerun."

    # Python virtual environments are not relocatable. Creating the venv under
    # a temporary name and moving it leaves absolute paths in bin/activate and
    # generated entry points pointing at the removed temporary directory.
    info "Creating virtual environment directly at $VENV_DIR."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
else
    ok "Reusing existing virtual environment at $VENV_DIR."
fi

# Repair environments created by the old create-then-move installer. Running
# venv again in place rewrites activation scripts while preserving packages.
ACTIVATE_PATH="$VENV_DIR/bin/activate"
if [[ -f "$ACTIVATE_PATH" ]] && ! grep -Fq "VIRTUAL_ENV=\"$VENV_DIR\"" "$ACTIVATE_PATH"; then
    warn "The existing environment contains relocated activation paths; repairing it in place."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

PYTHON="$VENV_DIR/bin/python"
PIP=("$PYTHON" -m pip)

VENV_TAG="$($PYTHON -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')"
[[ "$VENV_TAG" == "$PYTHON_TAG" ]] || die "The existing venv uses ${VENV_TAG}, but Ubuntu ${VERSION_ID} requires ${PYTHON_TAG} for this wheel set. Rename/remove $VENV_DIR and rerun."

info "Upgrading pip, setuptools, and wheel. Download progress is intentionally visible."
"${PIP[@]}" install --upgrade pip setuptools wheel

# ── 4. Install the exact AMD ROCm wheel set ────────────────────────────────

TORCH_FILE="torch-2.9.1%2Brocm7.2.1.lw.gitff65f5bc-${PYTHON_TAG}-${PYTHON_TAG}-linux_x86_64.whl"
VISION_FILE="torchvision-0.24.0%2Brocm7.2.1.gitb919bd0c-${PYTHON_TAG}-${PYTHON_TAG}-linux_x86_64.whl"
AUDIO_FILE="torchaudio-2.9.0%2Brocm7.2.1.gite3c6ee2b-${PYTHON_TAG}-${PYTHON_TAG}-linux_x86_64.whl"
TRITON_FILE="triton-3.5.1%2Brocm7.2.1.gita272dfa8-${PYTHON_TAG}-${PYTHON_TAG}-linux_x86_64.whl"

stack_matches() {
    "$PYTHON" - <<'PY'
import torch
import torchaudio
import torchvision
import triton

expected = {
    "torch": (torch.__version__, "2.9.1+rocm7.2.1"),
    "torchvision": (torchvision.__version__, "0.24.0+rocm7.2.1"),
    "torchaudio": (torchaudio.__version__, "2.9.0+rocm7.2.1"),
    "triton": (triton.__version__, "3.5.1"),
}
bad = [f"{name}={actual}" for name, (actual, prefix) in expected.items()
       if not actual.startswith(prefix)]
if bad:
    raise SystemExit("Unexpected versions: " + ", ".join(bad))
PY
}

if stack_matches >/dev/null 2>&1; then
    ok "The pinned PyTorch/ROCm/Triton stack is already installed."
else
    info "Installing the exact AMD ROCm ${ROCM_VERSION} wheels. This is the longest download step."
    # Remove incompatible remnants from an interrupted or earlier attempt.
    "${PIP[@]}" uninstall -y torch torchvision torchaudio triton pytorch-triton-rocm 2>/dev/null || true
    "${PIP[@]}" install --no-cache-dir \
        "${ROCM_WHEEL_BASE}/${TORCH_FILE}" \
        "${ROCM_WHEEL_BASE}/${VISION_FILE}" \
        "${ROCM_WHEEL_BASE}/${AUDIO_FILE}" \
        "${ROCM_WHEEL_BASE}/${TRITON_FILE}"
fi

stack_matches
ok "Pinned PyTorch, torchvision, torchaudio, and AMD Triton versions match."

# Verify GPU access now, before installing the application stack.
info "Verifying PyTorch GPU access."
"$PYTHON" - <<'PY'
import torch

print(f"PyTorch:      {torch.__version__}")
print(f"HIP runtime:  {torch.version.hip}")
print(f"GPU visible:  {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit(
        "PyTorch cannot access the AMD GPU. The Python environment is valid, "
        "but WSL/driver passthrough is not ready."
    )
print(f"GPU:          {torch.cuda.get_device_name(0)}")
print(f"BF16:         {torch.cuda.is_bf16_supported()}")
PY

# ── 5. Install FLA, Transformers, and the project ──────────────────────────

if "$PYTHON" -c 'from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule' >/dev/null 2>&1; then
    ok "flash-linear-attention ROCm operators are already importable."
else
    info "Installing flash-linear-attention 0.5.2 with its ROCm dependencies."
    "${PIP[@]}" install --no-cache-dir "flash-linear-attention[rocm]==0.5.2"
fi

"$PYTHON" -c 'from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule; print("FLA gated-delta operators import: OK")'

info "Installing the project and development dependencies."
"${PIP[@]}" install -e "${PROJECT_ROOT}[dev]"

info "Installing the configured Transformers build: $TRANSFORMERS_SPEC"
"${PIP[@]}" install --upgrade "$TRANSFORMERS_SPEC"

# Reassert the decoder version after dependency resolution. The application
# opens media with PyAV itself and passes preselected in-memory frames to
# qwen-vl-utils, so torchvision/TorchCodec are not part of semantic sampling.
info "Enforcing the tested PyAV uniform-sampler version."
"${PIP[@]}" install --upgrade --force-reinstall --no-cache-dir "av==18.1.0"

"$PYTHON" - <<'PY'
import av

if av.__version__ != "18.1.0":
    raise SystemExit(f"Unexpected PyAV version: {av.__version__}")

print(f"PyAV:         {av.__version__} (explicit uniform sampler)")
PY

# Project dependency resolution must not silently replace the pinned GPU stack.
stack_matches
"${PIP[@]}" check

# ── 6. Final verification and next commands ────────────────────────────────

info "Running final import and accelerator verification."
"$PYTHON" - <<'PY'
import torch
import transformers
import triton
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule

print(f"PyTorch:       {torch.__version__}")
print(f"Transformers:  {transformers.__version__}")
print(f"Triton:        {triton.__version__}")
print(f"HIP runtime:   {torch.version.hip}")
print(f"GPU:           {torch.cuda.get_device_name(0)}")
print(f"BF16:          {torch.cuda.is_bf16_supported()}")
print(f"FLA operator:  {chunk_gated_delta_rule.__module__}")
print(f"FLA recurrent: {fused_recurrent_gated_delta_rule.__module__}")
PY

info "Executing the WSL/ROCm acceleration smoke test."
"$PYTHON" "$PROJECT_ROOT/scripts/verify_wsl_acceleration.py"

if [[ -f "$PROJECT_ROOT/.env-wsl" ]]; then
    ok "Found $PROJECT_ROOT/.env-wsl."
else
    warn "$PROJECT_ROOT/.env-wsl is missing. Copy .env-wsl.example to .env-wsl before starting the service."
fi

printf '\n=== Setup complete ===\n\n'
printf 'Activate and verify:\n'
printf '  cd %q\n' "$PROJECT_ROOT"
printf '  source %q/bin/activate\n' "$VENV_DIR"
printf '  python -c %q\n' 'import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available(), torch.cuda.get_device_name(0))'
printf '\nStart the service:\n'
printf '  set -a; source .env-wsl; set +a\n'
printf '  python -m uvicorn visual_service.main:app --host 127.0.0.1 --port 8001\n'
printf '\nRun the benchmark after the service is healthy:\n'
printf '  python scripts/benchmark_e2e.py\n'
