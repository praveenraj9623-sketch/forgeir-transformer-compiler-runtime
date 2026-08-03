[CmdletBinding()]
param(
    [ValidateSet('Debug', 'Release')]
    [string] $Configuration = 'Debug'
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'common.ps1')

$root = Get-ForgeIRRoot
$preset = Get-ForgeIRWindowsPreset -Configuration $Configuration
Enable-ForgeIRWindowsToolchain

Push-Location $root
try {
    & ctest --preset $preset --output-on-failure
    if ($LASTEXITCODE -ne 0) {
        throw "CTest failed with exit code $LASTEXITCODE."
    }
} finally {
    Pop-Location
}
