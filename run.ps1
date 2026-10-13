$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $ProjectDir

$VenvPython = Join-Path $ProjectDir ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $VenvPython)) {
    Write-Host "Creating a small local environment (.venv)..." -ForegroundColor Cyan
    py -3.11 -m venv .venv
}

$PreviousErrorPreference = $ErrorActionPreference
$ErrorActionPreference = "SilentlyContinue"
& $VenvPython -c "import numpy" *> $null
$HasNumpy = ($LASTEXITCODE -eq 0)
$ErrorActionPreference = $PreviousErrorPreference

if (-not $HasNumpy) {
    Write-Host "Installing NumPy (first run only)..." -ForegroundColor Cyan
    & $VenvPython -m pip install --disable-pip-version-check -r requirements.txt
}

$Port = 8501
while (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
    $Port++
}

Write-Host "Opening the web at http://127.0.0.1:$Port" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop." -ForegroundColor DarkGray
& $VenvPython server.py --port $Port
