$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
Set-Location $RepoRoot

$script:DefaultModel = "Qwen/Qwen2.5-1.5B-Instruct"
$script:DefaultPplModel = "microsoft/Phi-4-mini-instruct"
$script:Grammars = @("email", "css-color", "json", "no-bomb", "ipv4", "ipv6", "threefold")

function Invoke-Poe {
    param(
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$Arguments
    )

    & uv run poe @Arguments
}

function Get-GrammarExtraArgs {
    param([string]$Grammar)

    switch ($Grammar) {
        "json" { return @("--max-tokens", "54") }
        default { return @() }
    }
}

function Get-ResultSettingDir {
    param([switch]$Baseline)

    if ($Baseline) {
        return "baseline"
    }
    return "diverse"
}

function Get-MetricResultPath {
    param(
        [string]$Experiment,
        [switch]$Baseline,
        [string]$Grammar,
        [string]$ResultSuffix = ""
    )

    $setting = Get-ResultSettingDir -Baseline:$Baseline
    if ($ResultSuffix) {
        return "results/$Experiment/$setting/$ResultSuffix/$Grammar.json"
    }
    return "results/$Experiment/$setting/$Grammar.json"
}

function Get-RuntimeResultPath {
    param(
        [switch]$Baseline,
        [string]$Grammar
    )

    $setting = Get-ResultSettingDir -Baseline:$Baseline
    return "results/runtime/$setting/$Grammar.json"
}

function Invoke-GenerationSuite {
    param(
        [switch]$Baseline,
        [string[]]$ExtraArgs = @()
    )

    foreach ($grammar in $script:Grammars) {
        $cmd = @("gen", $grammar, "--model", $script:DefaultModel, "-n", "1000")
        $cmd += Get-GrammarExtraArgs -Grammar $grammar
        $cmd += $ExtraArgs
        if ($Baseline) {
            $cmd += "--baseline"
        }
        Invoke-Poe @cmd
    }
}

function Invoke-EvalSuite {
    param(
        [switch]$Baseline,
        [string]$Experiment = "diversity",
        [string]$ResultSuffix = "",
        [string[]]$ExtraArgs = @()
    )

    foreach ($grammar in $script:Grammars) {
        $outputPath = Get-MetricResultPath -Experiment $Experiment -Baseline:$Baseline -Grammar $grammar -ResultSuffix $ResultSuffix
        $cmd = @("eval", $grammar, "--model", $script:DefaultModel, "--experiment", $Experiment, "--output", $outputPath)
        $cmd += $ExtraArgs
        if ($Baseline) {
            $cmd += "--baseline"
        }
        Invoke-Poe @cmd
    }
}

function Invoke-RuntimeSuite {
    param([switch]$Baseline)

    foreach ($grammar in $script:Grammars) {
        $outputPath = Get-RuntimeResultPath -Baseline:$Baseline -Grammar $grammar
        $cmd = @("eval-runtime", $grammar, "--model", $script:DefaultModel, "--output", $outputPath)
        if ($Baseline) {
            $cmd += "--baseline"
        }
        Invoke-Poe @cmd
    }
}
