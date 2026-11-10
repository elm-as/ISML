# PowerShell 1-Click Runner for ISML
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $ScriptDir

$PythonExe = "C:\Users\elmas\AppData\Local\Programs\Python\Python310\python.exe"
if (-not (Test-Path $PythonExe)) {
    $PythonExe = "python"
}

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "ISML: Impact of Social Media on Life (4,500 Cohort)" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan

& $PythonExe main.py @args
