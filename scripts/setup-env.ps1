#Requires -Version 5.1
<#
.SYNOPSIS
    Creates the Qwen3.5 Python 3.12 environment with ROCm/PyTorch.
.DESCRIPTION
    1. Creates .venv with Python 3.12
    2. Installs AMD ROCm SDK wheels (Windows, gfx1201)
    3. Installs PyTorch + torchvision + torchaudio (ROCm 7.2.1)
    4. Installs recent Transformers and project dependencies
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvDir = Join-Path $ProjectRoot ".venv"

# ── Step 1: Locate Python 3.12 ──────────────────────────────────────────────

$python = & py -3.12 -c "import sys; print(sys.executable)" 2>$null
if (-not $python) {
    Write-Error "Python 3.12 not found. Install it and ensure 'py -3.12' works."
    exit 1
}
Write-Host "Using Python: $python"

# ── Step 2: Create venv ─────────────────────────────────────────────────────

if (Test-Path $VenvDir) {
    Write-Host ".venv already exists at $VenvDir - skipping creation."
} else {
    Write-Host "Creating venv at $VenvDir ..."
    & py -3.12 -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) { Write-Error "Failed to create venv."; exit 1 }
}

$pip = Join-Path $VenvDir "Scripts\pip.exe"
$pythonVenv = Join-Path $VenvDir "Scripts\python.exe"

# Upgrade pip first
Write-Host "Upgrading pip ..."
& $pythonVenv -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { Write-Error "pip upgrade failed."; exit 1 }

# ── Step 3: Install ROCm SDK wheels ─────────────────────────────────────────

$rocmBase = "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1"

$rocmWheels = @(
    "$rocmBase/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl"
    "$rocmBase/rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl"
    "$rocmBase/rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl"
    "$rocmBase/rocm-7.2.1.tar.gz"
)

Write-Host "Installing ROCm SDK 7.2.1 ..."
& $pip install --no-cache-dir @rocmWheels
if ($LASTEXITCODE -ne 0) { Write-Error "ROCm SDK installation failed."; exit 1 }

# ── Step 4: Install PyTorch + torchvision + torchaudio ───────────────────────

$torchWheels = @(
    "$rocmBase/torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl"
    "$rocmBase/torchaudio-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl"
    "$rocmBase/torchvision-0.24.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl"
)

Write-Host "Installing PyTorch 2.9.1+rocm7.2.1 ..."
& $pip install --no-cache-dir @torchWheels
if ($LASTEXITCODE -ne 0) { Write-Error "PyTorch installation failed."; exit 1 }

# ── Step 5: Install project dependencies ─────────────────────────────────────

Write-Host "Installing project dependencies (including dev extras) ..."
& $pip install -e "$ProjectRoot[dev]"
if ($LASTEXITCODE -ne 0) { Write-Error "Project dependency installation failed."; exit 1 }

Write-Host ""
Write-Host "Setup complete. Activate with: .venv\Scripts\Activate.ps1"
Write-Host "Verify GPU with:              .\scripts\verify-gpu.ps1"
Write-Host "Download Qwen3.5-4B with:     .\scripts\download-model.ps1"
