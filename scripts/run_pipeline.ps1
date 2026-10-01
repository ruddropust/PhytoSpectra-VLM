[CmdletBinding()]
param(
    [ValidateSet('prepare','stage42d-parity-check','phytocore','phytocore-evaluate','evidence-alignment','evidence-outer','vqa','vlm','remediation','cultivar-ranking','fixed-interpolation','contract-optimization','vlm-evaluate','ablation','statistics','report','figures','aggregate','campaign-figures','source-freeze','all')]
    [string]$Stage = 'all',
    [ValidateSet(2026, 2027, 2028)][int]$Seed = 2026,
    [ValidateRange(0, 4)][int]$Fold = 0,
    [string]$RunId = 'final_3x5_v1',
    [string]$RawData = '',
    [string]$Python = 'E:\Habib\.ruddro\Scripts\python.exe',
    [switch]$Smoke,
    [switch]$Resume,
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not $RawData) { $RawData = Join-Path $Root 'data\raw' }
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Configured research virtual-environment Python was not found: $Python"
}
$PreviousPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = Join-Path $Root 'src'
try {
    $Arguments = @('-m', 'phytospectra', $Stage, '--root', $Root, '--raw-data', $RawData, '--run-id', $RunId, '--seed', "$Seed", '--fold', "$Fold")
    if ($Smoke) { $Arguments += '--smoke' }
    if ($Resume) { $Arguments += '--resume' }
    if ($Force) { $Arguments += '--force' }
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Pipeline stage '$Stage' failed." }
}
finally { $env:PYTHONPATH = $PreviousPythonPath }
