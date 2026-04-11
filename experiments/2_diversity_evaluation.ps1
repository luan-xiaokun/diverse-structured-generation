. "$PSScriptRoot/common.ps1"

# evaluate diverse samples with diversity metrics
Invoke-EvalSuite

# evaluate baseline samples with diversity metrics
Invoke-EvalSuite -Baseline
