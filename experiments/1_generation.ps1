. "$PSScriptRoot/common.ps1"

# diverse generation with 1000 samples per grammar
Invoke-GenerationSuite

# baseline generation with 1000 samples per grammar
Invoke-GenerationSuite -Baseline
