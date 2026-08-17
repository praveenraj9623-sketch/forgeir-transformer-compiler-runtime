[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'

$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path -LiteralPath $vswhere -PathType Leaf)) {
    throw 'Visual Studio Installer vswhere.exe was not found.'
}

$installationPath = & $vswhere -latest -products * `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
    -property installationPath
if ($LASTEXITCODE -ne 0 -or -not $installationPath) {
    throw 'Visual Studio 2022 with the MSVC x64 build tools was not found.'
}

$developerShell = Join-Path $installationPath 'Common7\Tools\Launch-VsDevShell.ps1'
if (-not (Test-Path -LiteralPath $developerShell -PathType Leaf)) {
    throw "Visual Studio developer shell was not found at $developerShell."
}

& $developerShell -Arch amd64 -HostArch amd64 -SkipAutomaticLocation

$resolvedTools = @{}
foreach ($tool in @('cl.exe', 'cmake.exe', 'ninja.exe')) {
    $command = Get-Command $tool -ErrorAction SilentlyContinue
    if (-not $command) {
        throw "$tool was not found after initializing the Visual Studio developer shell."
    }
    $resolvedTools[$tool] = $command.Source
}

$cmakeVersion = (& cmake --version | Select-Object -First 1)
if ($LASTEXITCODE -ne 0) {
    throw "cmake --version failed with exit code $LASTEXITCODE."
}
$ninjaVersion = & ninja --version
if ($LASTEXITCODE -ne 0) {
    throw "ninja --version failed with exit code $LASTEXITCODE."
}

Write-Output 'ForgeIR MSVC x64 environment initialized for this PowerShell process.'
Write-Output "Visual Studio: $installationPath"
Write-Output "cl.exe: $($resolvedTools['cl.exe'])"
Write-Output "cmake.exe: $($resolvedTools['cmake.exe']) ($cmakeVersion)"
Write-Output "ninja.exe: $($resolvedTools['ninja.exe']) (version $ninjaVersion)"
Write-Output "VCToolsInstallDir: $env:VCToolsInstallDir"
Write-Output "WindowsSdkDir: $env:WindowsSdkDir"
