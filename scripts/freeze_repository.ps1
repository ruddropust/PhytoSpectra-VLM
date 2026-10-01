[CmdletBinding()]
param(
    [string]$RunId = 'final_3x5_v1',
    [string]$Python = 'E:\Habib\.ruddro\Scripts\python.exe'
)
$ErrorActionPreference = 'Stop'
$Runner = Join-Path $PSScriptRoot 'run_pipeline.ps1'
& $Runner -Stage source-freeze -RunId $RunId -Seed 2026 -Fold 0 -Python $Python
if ($LASTEXITCODE -ne 0) { throw 'Source freeze failed.' }
