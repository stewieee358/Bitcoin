param([switch]$Smoke, [switch]$Full)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if ($Smoke -and $Full) { throw 'Choose either -Smoke or -Full.' }
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw 'uv is required. Run: python -m pip install uv, then reopen the terminal.'
}
if (-not (Test-Path -LiteralPath '.\.venv\Scripts\python.exe')) {
    uv python install 3.12
    if ($LASTEXITCODE -ne 0) { throw 'Python installation failed.' }
    uv venv --python 3.12
    if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed.' }
}
$treePython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
& $treePython -c 'import sys; assert sys.version_info[:2] in ((3,12),(3,13)), "Use Python 3.12 or 3.13 for the pinned project dependencies."'
if ($LASTEXITCODE -ne 0) { throw 'Use a compatible .venv; see README_Tree_Study_CN.md.' }
uv pip install --python $treePython -r requirements_tree_study.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:LOKY_MAX_CPU_COUNT = '4'
if ($Full) {
    & $treePython run_tree_study.py
} else {
    # A short run is the default; pass -Full for the entire 2025 evaluation.
    & $treePython run_tree_study.py --evaluation-end '2025-01-04T00:00:00+00:00'
}
if ($LASTEXITCODE -ne 0) { throw 'Tree study failed. Inspect the terminal and run.json status.' }
