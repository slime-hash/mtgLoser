param(
    [ValidatePattern('^[a-z0-9][a-z0-9_-]{0,95}$')]
    [string]$Player = 'unknown',
    [switch]$DeferLabeling
)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (-not (Test-Path '.venv/Scripts/python.exe')) {
    throw 'Python environment is missing. Run setup.ps1 first.'
}

$sessionArguments = @(
    'main.py', 'session',
    '--source', 'screen',
    '--window', 'Champions',
    '--player', $Player
)
if ($DeferLabeling) {
    $sessionArguments += '--defer-labeling'
}
& ./.venv/Scripts/python.exe @sessionArguments
