Param(
    [string]$OutputDir = "artifacts",
    [int]$Dimension = 256,
    [int]$Samples = 256,
    [int]$DatasetSize = 4096,
    [int]$Queries = 128,
    [int]$TopK = 10,
    [string]$Bits = "1,2,3,4",
    [string]$RecallCurveKs = "1,2,4,8,16,32,64",
    [UInt64]$Seed = 7
)

$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

function Invoke-CargoCsv {
    Param(
        [string[]]$CargoArgs,
        [string]$Path
    )

    & cargo $CargoArgs | Out-File -Encoding ascii $Path
    if ($LASTEXITCODE -ne 0) {
        throw "cargo $($CargoArgs -join ' ') failed"
    }
}

$msePath = Join-Path $OutputDir "mse_validation.csv"
$prodPath = Join-Path $OutputDir "prod_validation.csv"
$recallPath = Join-Path $OutputDir "recall_validation.csv"
$recallCurvePath = Join-Path $OutputDir "recall_curve_validation.csv"
$boundsPath = Join-Path $OutputDir "bounds_validation.csv"
$filteredRecallCurveKs = ($RecallCurveKs -split '[,\s]+' | Where-Object { $_ -ne "" } | ForEach-Object { [int]$_ } | Where-Object { $_ -le $DatasetSize }) -join ","
if ([string]::IsNullOrWhiteSpace($filteredRecallCurveKs)) {
    throw "RecallCurveKs must contain at least one value less than or equal to DatasetSize"
}

Invoke-CargoCsv -CargoArgs @(
    "run", "--quiet", "--release", "--bin", "paper_validation", "--",
    "mse",
    "--dimension", "$Dimension",
    "--samples", "$Samples",
    "--bits", "$Bits",
    "--seed", "$Seed"
) -Path $msePath

Invoke-CargoCsv -CargoArgs @(
    "run", "--quiet", "--release", "--bin", "paper_validation", "--",
    "prod",
    "--dimension", "$Dimension",
    "--samples", "$Samples",
    "--bits", "$Bits",
    "--seed", "$Seed"
) -Path $prodPath

Invoke-CargoCsv -CargoArgs @(
    "run", "--quiet", "--release", "--bin", "paper_validation", "--",
    "recall",
    "--dimension", "$Dimension",
    "--dataset-size", "$DatasetSize",
    "--queries", "$Queries",
    "--top-k", "$TopK",
    "--bits", "$Bits",
    "--seed", "$Seed"
) -Path $recallPath

Invoke-CargoCsv -CargoArgs @(
    "run", "--quiet", "--release", "--bin", "paper_validation", "--",
    "recall_curve",
    "--dimension", "$Dimension",
    "--dataset-size", "$DatasetSize",
    "--queries", "$Queries",
    "--ks", "$filteredRecallCurveKs",
    "--bits", "$Bits",
    "--seed", "$Seed"
) -Path $recallCurvePath

Invoke-CargoCsv -CargoArgs @(
    "run", "--quiet", "--release", "--bin", "paper_validation", "--",
    "bounds",
    "--metric", "both",
    "--dimension", "$Dimension",
    "--samples", "$Samples",
    "--bits", "$Bits",
    "--seed", "$Seed"
) -Path $boundsPath

Write-Host "Wrote:"
Write-Host "  $msePath"
Write-Host "  $prodPath"
Write-Host "  $recallPath"
Write-Host "  $recallCurvePath"
Write-Host "  $boundsPath"
