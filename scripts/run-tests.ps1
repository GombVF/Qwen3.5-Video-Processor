#Requires -Version 5.1
<#
.SYNOPSIS
    Runs the project test suite via pytest.
.DESCRIPTION
    Accepts an optional test path/marker filter as arguments.
    Examples:
        .\scripts\run-tests.ps1                          # all tests
        .\scripts\run-tests.ps1 tests/unit               # unit only
        .\scripts\run-tests.ps1 tests/contract            # contract only
        .\scripts\run-tests.ps1 -k "test_planner"        # by name
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$pythonVenv = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $pythonVenv)) {
    Write-Error ".venv not found. Run scripts\setup-env.ps1 first."
    exit 1
}

Set-Location $ProjectRoot
& $pythonVenv -m pytest @args

exit $LASTEXITCODE
