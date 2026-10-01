[CmdletBinding()]
param(
    [string]$RunId = 'final_3x5_v1',
    [string]$RawData = 'E:\Habib\PhytoChat\data\raw',
    [string]$Python = 'E:\Habib\.ruddro\Scripts\python.exe',
    [switch]$Resume,
    [switch]$Smoke,
    [switch]$ConfirmFullCampaign
)

$ErrorActionPreference = 'Stop'
$Runner = Join-Path $PSScriptRoot 'run_pipeline.ps1'
if (-not $Smoke -and -not $ConfirmFullCampaign) {
    throw 'Full 15-run execution requires -ConfirmFullCampaign.'
}
$Seeds = @(2026, 2027, 2028)
$Folds = @(0, 1, 2, 3, 4)
$Total = $Seeds.Count * $Folds.Count
$Position = 0
foreach ($Seed in $Seeds) {
    foreach ($Fold in $Folds) {
        $Position += 1
        Write-Host "[$Position/$Total] seed=$Seed fold=$Fold"
        # PowerShell array splatting is positional for script invocation, so a
        # token such as "-Stage" can become the Stage value. Use a hashtable to
        # preserve the named-parameter contract across PowerShell hosts.
        $PipelineParams = @{
            Stage = 'all'
            Seed = $Seed
            Fold = $Fold
            RunId = $RunId
            RawData = $RawData
            Python = $Python
            Resume = $Resume
            Smoke = $Smoke
        }
        & $Runner @PipelineParams
        if ($LASTEXITCODE -ne 0) { throw "Stopped at seed=$Seed fold=$Fold. Rerun unchanged with -Resume." }
    }
}
& $Runner -Stage aggregate -Seed 2026 -Fold 0 -RunId $RunId -RawData $RawData -Python $Python
& $Runner -Stage campaign-figures -Seed 2026 -Fold 0 -RunId $RunId -RawData $RawData -Python $Python
Write-Host "Complete: reports\$RunId\campaign"
