# Create the tested local environment. No data acquisition or model training.
[CmdletBinding()]
param([string]$PythonExecutable = "python")

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$temporaryRoot = Join-Path $projectRoot "tmp"
New-Item -ItemType Directory -Path $temporaryRoot -Force | Out-Null
$env:TEMP = $temporaryRoot
$env:TMP = $temporaryRoot
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:PIP_DISABLE_PIP_VERSION_CHECK = "1"

try {
    & $PythonExecutable -B -c "import sys; assert sys.version_info[:2] == (3, 14), 'PitWall setup is verified with Python 3.14'; print(sys.version.split()[0])"
    if ($LASTEXITCODE -ne 0) { throw "Python version check failed." }
} catch {
    throw "Supply a working Python 3.14 executable using -PythonExecutable. $($_.Exception.Message)"
}

$environmentRoot = Join-Path $projectRoot ".venv"
$environmentPython = Join-Path $environmentRoot "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $environmentPython)) {
    & $PythonExecutable -B -m venv $environmentRoot
    if ($LASTEXITCODE -ne 0) { throw "Virtual environment creation failed." }
}
& $environmentPython -B -c "import sys; assert sys.version_info[:2] == (3, 14), 'Existing .venv must use Python 3.14'"
if ($LASTEXITCODE -ne 0) { throw "Existing virtual environment has an incompatible Python version." }

$dependencyFile = Join-Path $projectRoot "requirements-lock.txt"
& $environmentPython -B -m pip install --no-cache-dir -r $dependencyFile
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }
& $environmentPython -B -m pip check
if ($LASTEXITCODE -ne 0) { throw "Dependency verification failed." }
Write-Host "Setup complete. From the project directory, run: .\.venv\Scripts\python.exe -m streamlit run app.py"
