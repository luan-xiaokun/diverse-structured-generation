. "$PSScriptRoot/common.ps1"

$timeoutSeconds = if ($env:TIMEOUT_SECONDS) { [int]$env:TIMEOUT_SECONDS } else { 1800 }
$ablationComponents = @("reward", "penalty", "range_scaling")

function Invoke-GenerationWithTimeout {
    param([string]$Component)

    $job = Start-Job -ScriptBlock {
        param($RepoRoot, $Model, $AblationComponent, $Seed)
        Set-Location $RepoRoot
        $seedArgs = @()
        if ($Seed) {
            $seedArgs = @("--seed", $Seed)
        }
        & uv run poe gen css-color --model $Model -n 1000 --ablation-component $AblationComponent @seedArgs
    } -ArgumentList $RepoRoot, $script:DefaultModel, $Component, $script:DefaultSeed

    if (-not (Wait-Job $job -Timeout $timeoutSeconds)) {
        Stop-Job $job
        Remove-Job $job
        throw "Timed out after $timeoutSeconds seconds while generating component '$Component'."
    }

    Receive-Job $job
    Remove-Job $job
}

# generate samples with ablation of reward, penalty, and range scaling components
foreach ($component in $ablationComponents) {
    Invoke-GenerationWithTimeout -Component $component
}

# evaluate the default and ablated runs
Invoke-Poe eval css-color `
    --model $script:DefaultModel `
    --experiment component_ablation `
    --output results/component_ablation/default/css-color.json
foreach ($component in $ablationComponents) {
    Invoke-Poe eval css-color `
        --model $script:DefaultModel `
        --experiment component_ablation `
        --ablation-component $component `
        --output "results/component_ablation/$component/css-color.json"
}
