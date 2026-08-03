[CmdletBinding()]
param(
    [ValidateSet('Debug', 'Release')]
    [string] $Configuration = 'Debug'
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'common.ps1')

$root = Get-ForgeIRRoot
$preset = Get-ForgeIRWindowsPreset -Configuration $Configuration
$buildRoot = [IO.Path]::GetFullPath((Join-Path $root 'build'))
$target = [IO.Path]::GetFullPath((Join-Path $buildRoot $preset))
if (-not $target.StartsWith("$buildRoot$([IO.Path]::DirectorySeparatorChar)")) {
    throw "Refusing to clean a path outside $buildRoot."
}

if (Test-Path -LiteralPath $target) {
    Remove-Item -LiteralPath $target -Recurse -Force
    Write-Output "Removed $target"
} else {
    Write-Output "Nothing to clean at $target"
}
