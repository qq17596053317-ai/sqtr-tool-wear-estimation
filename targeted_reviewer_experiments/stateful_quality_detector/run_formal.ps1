$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$rawFile = Get-ChildItem -Path 'C:\Users\66485\Desktop' -Recurse `
    -Filter 'nuaa_orthogonal_bundle_high_resolution.csv' | Select-Object -First 1
if (-not $rawFile) { throw 'High-resolution NUAA data file was not found.' }
$env:SQTR_RAW_DATA_DIR = Split-Path -Parent $rawFile.FullName
$env:SQTR_STATEFUL_TRAIN_REPEATS = '3'
$env:SQTR_STATEFUL_TEST_REPEATS = '30'
$env:SQTR_STATEFUL_RESULT_DIR = Join-Path $PSScriptRoot 'results'

Set-Location -LiteralPath $repo
matlab -batch "run('step42_stateful_quality_detector.m')"
