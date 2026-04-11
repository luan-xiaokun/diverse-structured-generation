. "$PSScriptRoot/common.ps1"

# runtime evaluation for diverse generation
Invoke-RuntimeSuite

# runtime evaluation for the baseline
Invoke-RuntimeSuite -Baseline
