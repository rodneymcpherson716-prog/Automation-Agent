param(
    [switch]$Apply
)

$ErrorActionPreference = 'Stop'
$root = [System.IO.Path]::GetFullPath($PSScriptRoot).TrimEnd('\')
$archive = Join-Path $root 'archive'
$gitPrefix = Join-Path $root '.git'
$archivePrefix = $archive
$excludedNames = @(
    'execution_engine.py',
    'start_agent.cmd',
    'requirements.txt'
)

$files = @(
    Get-ChildItem -LiteralPath $root -Force -Recurse -File |
        Where-Object {
            $inGit = $_.FullName.StartsWith(
                $gitPrefix + '\',
                [System.StringComparison]::OrdinalIgnoreCase
            )
            $inArchive = $_.FullName.StartsWith(
                $archivePrefix + '\',
                [System.StringComparison]::OrdinalIgnoreCase
            )
            $matchesPattern = $_.Name -like '*.bak' -or
                $_.Name -like '*backup*' -or
                $_.Name -like 'before_*'

            -not $inGit -and
                -not $inArchive -and
                $_.Name -notin $excludedNames -and
                $matchesPattern
        } |
        Sort-Object FullName
)

Write-Output "Files matching cleanup patterns: $($files.Count)"
foreach ($file in $files) {
    $relativePath = $file.FullName.Substring($root.Length).TrimStart('\')
    Write-Output "$relativePath -> archive\$relativePath"
}

if (-not $Apply) {
    Write-Output 'Preview only. Re-run with -Apply to move these files.'
    return
}

$moves = @(
    foreach ($file in $files) {
        $relativePath = $file.FullName.Substring($root.Length).TrimStart('\')
        $destination = Join-Path $archive $relativePath
        if (Test-Path -LiteralPath $destination) {
            throw "Destination already exists; no files were moved: $destination"
        }
        [PSCustomObject]@{
            Source = $file.FullName
            Destination = $destination
        }
    }
)

foreach ($move in $moves) {
    $destinationDirectory = Split-Path -Parent $move.Destination
    New-Item -ItemType Directory -Path $destinationDirectory -Force | Out-Null
    Move-Item -LiteralPath $move.Source -Destination $move.Destination
}

Write-Output "Moved $($moves.Count) files into archive."
