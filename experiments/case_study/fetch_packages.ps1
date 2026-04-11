$ErrorActionPreference = "Stop"

$EmailValidatorUrl = "https://github.com/JoshData/python-email-validator.git"
$EmailValidatorCommit = "936aead3bf5c608f8561954e0d2955b7f97bfdad"
$EmailValidatorSubdir = "email_validator"
$WebcolorsUrl = "https://github.com/ubernostrum/webcolors.git"
$WebcolorsCommit = "834f77b381fad6eb31634d583894c3bc16a7ff99"
$WebcolorsSubdir = "src/webcolors"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$EmailValidatorTargetDir = Join-Path $ScriptDir "email_validator"
$WebcolorsTargetDir = Join-Path $ScriptDir "webcolors"
$TempDir = Join-Path ([System.IO.Path]::GetTempPath()) ("case-study-" + [System.Guid]::NewGuid().ToString("N"))

function Require-Command {
    param([string]$Name)

    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Missing required command: $Name"
    }
}

function Export-Snapshot {
    param(
        [string]$RepoUrl,
        [string]$Commit,
        [string]$SparseSubdir,
        [string]$CheckoutDir,
        [string]$TargetDir
    )

    & git clone --filter=blob:none --no-checkout $RepoUrl $CheckoutDir
    & git -C $CheckoutDir sparse-checkout init --cone
    & git -C $CheckoutDir sparse-checkout set $SparseSubdir
    & git -C $CheckoutDir checkout $Commit

    if (Test-Path $TargetDir) {
        Remove-Item $TargetDir -Recurse -Force
    }

    Copy-Item (Join-Path $CheckoutDir $SparseSubdir) $TargetDir -Recurse
}

Require-Command git
New-Item -ItemType Directory -Path $TempDir | Out-Null

try {
    Write-Host "Using temporary directory: $TempDir"

    Export-Snapshot `
        -RepoUrl $EmailValidatorUrl `
        -Commit $EmailValidatorCommit `
        -SparseSubdir $EmailValidatorSubdir `
        -CheckoutDir (Join-Path $TempDir "python-email-validator") `
        -TargetDir $EmailValidatorTargetDir

    Write-Host "Done."
    Write-Host "Exported $EmailValidatorSubdir at EMAIL_VALIDATOR_COMMIT $EmailValidatorCommit to:"
    Write-Host "  $EmailValidatorTargetDir"

    Export-Snapshot `
        -RepoUrl $WebcolorsUrl `
        -Commit $WebcolorsCommit `
        -SparseSubdir $WebcolorsSubdir `
        -CheckoutDir (Join-Path $TempDir "webcolors") `
        -TargetDir $WebcolorsTargetDir

    Write-Host "Done."
    Write-Host "Exported $WebcolorsSubdir at WEBCOLORS_COMMIT $WebcolorsCommit to:"
    Write-Host "  $WebcolorsTargetDir"
    Write-Host "Pinned package snapshots refreshed successfully."
}
finally {
    if (Test-Path $TempDir) {
        Remove-Item $TempDir -Recurse -Force
    }
}
