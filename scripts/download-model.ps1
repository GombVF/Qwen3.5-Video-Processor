#Requires -Version 5.1
param(
    [string]$Destination = "E:\Trabajo\Proyectos\Qwen\Qwen3.5-4B"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$pythonVenv = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$hfCli = Join-Path $ProjectRoot ".venv\Scripts\hf.exe"
if (-not (Test-Path $pythonVenv)) {
    Write-Error ".venv not found. Run scripts\setup-env.ps1 first."
    exit 1
}

if (-not (Test-Path $hfCli)) {
    Write-Error "hf.exe not found. Re-run scripts\setup-env.ps1."
    exit 1
}

& $hfCli download `
    Qwen/Qwen3.5-4B `
    --local-dir $Destination

if ($LASTEXITCODE -ne 0) {
    Write-Error "Model download failed."
    exit 1
}

Write-Host "Qwen3.5-4B downloaded to $Destination"
