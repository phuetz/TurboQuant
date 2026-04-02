Param(
    [Parameter(Mandatory = $true)]
    [string]$Data,
    [string]$Queries = "",
    [string]$OutputDir = "artifacts",
    [int]$TrainSize = 100000,
    [int]$QuerySize = 1000,
    [string]$Bits = "2,4",
    [string]$Ks = "1,2,4,8,16,32,64",
    [int]$PqSubspaces = 24,
    [int]$PqIters = 12,
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

$turboPath = Join-Path $OutputDir "ann_turboquant.csv"
$pqPath = Join-Path $OutputDir "ann_pq.csv"
$rabitqPath = Join-Path $OutputDir "ann_rabitq.csv"

$queryArgs = @()
if ($Queries -ne "") {
    $queryArgs += "--queries"
    $queryArgs += $Queries
}

$turboArgs = @(
    "run", "--quiet", "--release", "--bin", "ann_benchmark", "--",
    "--method", "turboquant",
    "--data", $Data
) + $queryArgs + @(
    "--train-size", "$TrainSize",
    "--query-size", "$QuerySize",
    "--bits", "$Bits",
    "--ks", "$Ks",
    "--seed", "$Seed"
)
Invoke-CargoCsv -CargoArgs $turboArgs -Path $turboPath

$pqArgs = @(
    "run", "--quiet", "--release", "--bin", "ann_benchmark", "--",
    "--method", "pq",
    "--data", $Data
) + $queryArgs + @(
    "--train-size", "$TrainSize",
    "--query-size", "$QuerySize",
    "--bits", "$Bits",
    "--subspaces", "$PqSubspaces",
    "--pq-iters", "$PqIters",
    "--ks", "$Ks",
    "--seed", "$Seed"
)
Invoke-CargoCsv -CargoArgs $pqArgs -Path $pqPath

$rabitqArgs = @(
    "run", "--quiet", "--release", "--bin", "ann_benchmark", "--",
    "--method", "rabitq",
    "--data", $Data
) + $queryArgs + @(
    "--train-size", "$TrainSize",
    "--query-size", "$QuerySize",
    "--bits", "$Bits",
    "--ks", "$Ks",
    "--seed", "$Seed"
)
Invoke-CargoCsv -CargoArgs $rabitqArgs -Path $rabitqPath

Write-Host "Wrote:"
Write-Host "  $turboPath"
Write-Host "  $pqPath"
Write-Host "  $rabitqPath"
