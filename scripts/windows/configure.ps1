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
$python = Get-ForgeIRActivePython

Push-Location $root
try {
    $pybind11CmakeDirectory = & $python -m pybind11 --cmakedir
    if ($LASTEXITCODE -ne 0 -or -not $pybind11CmakeDirectory) {
        throw 'python -m pybind11 --cmakedir failed in the active virtual environment.'
    }
    & cmake --preset $preset "-Dpybind11_DIR=$pybind11CmakeDirectory"
    if ($LASTEXITCODE -ne 0) {
        throw "CMake configure failed with exit code $LASTEXITCODE."
    }
} finally {
    Pop-Location
}
