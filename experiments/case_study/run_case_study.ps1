$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $ScriptDir ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    throw "Expected virtual environment at $ScriptDir\.venv. Run 'uv sync' in this directory first."
}

Set-Location $ScriptDir

function Require-File {
    param([string]$PathValue)

    if (-not (Test-Path $PathValue -PathType Leaf)) {
        throw "Expected file not found: $PathValue"
    }
}

function Cleanup-OldOutputs {
    Get-ChildItem -Force ".coverage*" -ErrorAction SilentlyContinue |
        Where-Object { -not $_.PSIsContainer } |
        Remove-Item -Force
    Remove-Item "case_study_summary.json" -Force -ErrorAction SilentlyContinue
}

function Run-TestWithCoverage {
    param(
        [string]$Setting,
        [string]$Grammar,
        [string]$Source
    )

    $coverageFile = ".coverage.$Setting.$Grammar"
    $coverageJsonFile = "$coverageFile.json"
    $testFile = "tests/test_$Source.py"
    $htmlDir = "htmlcov_${Setting}_${Grammar}"
    $datasetFilename = "$Setting-$Grammar.json"

    Require-File $testFile
    Require-File "data/$datasetFilename"

    Write-Host "Running tests for $Source with $Setting test cases..."
    $env:TEST_DATA_FILENAME = $datasetFilename
    & $VenvPython -m coverage run --data-file="$coverageFile" --source "$Source" -m pytest "$testFile" -qq
    & $VenvPython -m coverage report --data-file="$coverageFile"
    & $VenvPython -m coverage html -d "$htmlDir" --data-file="$coverageFile"
    & $VenvPython -m coverage json --data-file="$coverageFile" -o "$coverageJsonFile" | Out-Null
    Remove-Item Env:TEST_DATA_FILENAME -ErrorAction SilentlyContinue
}

Cleanup-OldOutputs

Run-TestWithCoverage -Setting "baseline" -Grammar "email" -Source "email_validator"
Run-TestWithCoverage -Setting "diverse" -Grammar "email" -Source "email_validator"
Run-TestWithCoverage -Setting "baseline" -Grammar "css-color" -Source "webcolors"
Run-TestWithCoverage -Setting "diverse" -Grammar "css-color" -Source "webcolors"

& $VenvPython "report_case_study.py"
Write-Host "Case study summary written to case_study_summary.json."
