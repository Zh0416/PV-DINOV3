$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$workspaceRoot = Split-Path -Parent $repoRoot
$pythonPath = Join-Path $workspaceRoot '.venv-kaggle\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Kaggle CLI environment is missing. See README.md to install it."
}

& $pythonPath -m kaggle @args
exit $LASTEXITCODE
