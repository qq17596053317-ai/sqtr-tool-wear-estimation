$ErrorActionPreference = 'Stop'

$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$dataRoot = $env:SQTR_GRID_DATA_ROOT
if ([string]::IsNullOrWhiteSpace($dataRoot)) {
    throw 'SQTR_GRID_DATA_ROOT must point to the folder containing the high-resolution CSV files.'
}
$matlab = 'E:\Program Files\MATLAB\R2024b\bin\matlab.exe'
$script = Join-Path $repo 'step28_randomized_fault_repeats.m'
$root = $PSScriptRoot
$masterLog = Join-Path $root 'threshold_grid_runner.log'

"Threshold grid started: $(Get-Date -Format o)" | Set-Content -LiteralPath $masterLog

$floors = @(3, 4, 5)
$quantiles = @(97.5, 99, 99.5)
foreach ($floor in $floors) {
    foreach ($quantile in $quantiles) {
        $qLabel = ($quantile.ToString('0.0', [Globalization.CultureInfo]::InvariantCulture)).Replace('.', 'p')
        $out = Join-Path $root ("floor{0}_q{1}" -f $floor, $qLabel)
        New-Item -ItemType Directory -Force -Path $out | Out-Null

        $env:SQTR_RANDOM_FAULT_RESULT_DIR = $out
        $env:SQTR_RAW_DATA_DIR = $dataRoot
        $env:SQTR_FAULT_SET = 'original'
        $env:SQTR_DETECTOR_FLOOR = [string]$floor
        $env:SQTR_DETECTOR_QUANTILE = [string]$quantile
        $env:SQTR_EXPORT_RECORD_AUDIT = '0'
        $env:SQTR_RANDOM_FAULT_REPEAT_COUNT = '30'

        $stdout = Join-Path $out 'matlab.log'
        $stderr = Join-Path $out 'matlab.err'
        $batch = "run('$($script.Replace("'", "''"))')"
        "Starting floor=$floor quantile=$quantile at $(Get-Date -Format o)" | Add-Content -LiteralPath $masterLog
        $process = Start-Process -FilePath $matlab -ArgumentList '-batch', $batch `
            -RedirectStandardOutput $stdout -RedirectStandardError $stderr `
            -WindowStyle Hidden -Wait -PassThru
        "Finished floor=$floor quantile=$quantile exit=$($process.ExitCode) at $(Get-Date -Format o)" | Add-Content -LiteralPath $masterLog
        if ($process.ExitCode -ne 0) {
            throw "MATLAB failed for floor=$floor quantile=$quantile. See $stderr"
        }
    }
}

"Threshold grid completed: $(Get-Date -Format o)" | Add-Content -LiteralPath $masterLog
