#Requires -Version 5.1
<#
.SYNOPSIS
    Verifies that PyTorch can see the AMD GPU via ROCm.
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$pythonVenv = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $pythonVenv)) {
    Write-Error ".venv not found. Run scripts\setup-env.ps1 first."
    exit 1
}

& $pythonVenv -c @'
import torch, sys

print(f"Python:          {sys.version}")
print(f"PyTorch:         {torch.__version__}")
print(f"CUDA available:  {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("\nERROR: No GPU detected. Check AMD driver and ROCm installation.")
    sys.exit(1)

print(f"Device count:    {torch.cuda.device_count()}")
print(f"Device name:     {torch.cuda.get_device_name(0)}")
print(f"HIP version:     {torch.version.hip}")

mem = torch.cuda.get_device_properties(0).total_mem
print(f"VRAM:            {mem / (1024**3):.1f} GB")

t = torch.randn(256, 256, device="cuda")
result = (t @ t).sum().item()
print(f"\nGPU compute OK   (matmul checksum: {result:.2f})")
'@

if ($LASTEXITCODE -ne 0) { exit 1 }
