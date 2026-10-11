#Requires -Version 5.1
<#
.SYNOPSIS
    Launches the visual analysis service via uvicorn.
.DESCRIPTION
    Starts uvicorn with a single worker bound to 127.0.0.1:8001.
    Override host/port with VISUAL_HOST / VISUAL_PORT env vars.
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$pythonVenv = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $pythonVenv)) {
    Write-Error ".venv not found. Run scripts\setup-env.ps1 first."
    exit 1
}

$host_ = if ($env:VISUAL_HOST) { $env:VISUAL_HOST } else { "127.0.0.1" }
$port  = if ($env:VISUAL_PORT) { $env:VISUAL_PORT } else { "8001" }

Write-Host "Starting visual-service on ${host_}:${port} ..."

Set-Location $ProjectRoot
& $pythonVenv -m uvicorn visual_service.main:app --host $host_ --port $port
