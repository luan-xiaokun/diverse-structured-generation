. "$PSScriptRoot/common.ps1"

$temperature = "1.5"
$commonAblationArgs = @("--temperature", $temperature)
$pplArgs = @(
    "--temperature", $temperature,
    "--ppl-model", $script:DefaultPplModel,
    "--ppl"
)

# diverse generation with temperature 1.5
Invoke-GenerationSuite -ExtraArgs $commonAblationArgs

# baseline generation with temperature 1.5
Invoke-GenerationSuite -Baseline -ExtraArgs $commonAblationArgs

# evaluate diverse samples with diversity metrics and perplexity
Invoke-EvalSuite -ExtraArgs $pplArgs

# evaluate baseline samples with diversity metrics and perplexity
Invoke-EvalSuite -Baseline -ExtraArgs $pplArgs
