#!/usr/bin/env pwsh
$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
python "$scriptDir/build_wd_kernel.py" @args
