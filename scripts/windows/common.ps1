Set-StrictMode -Version Latest

function Get-ForgeIRRoot {
    return (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
}

function Get-ForgeIRWindowsPreset {
    param(
        [Parameter(Mandatory = $true)]
        [ValidateSet('Debug', 'Release')]
        [string] $Configuration
    )

    return "windows-msvc-$($Configuration.ToLowerInvariant())"
}

function Enable-ForgeIRWindowsToolchain {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (-not (Test-Path -LiteralPath $vswhere -PathType Leaf)) {
        throw 'Visual Studio Installer vswhere.exe was not found.'
    }

    $installationPath = & $vswhere -latest -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath
    if (-not $installationPath) {
        throw 'Visual Studio with the MSVC x64 toolchain was not found.'
    }

    $vsDevCmd = Join-Path $installationPath 'Common7\Tools\VsDevCmd.bat'
    $environmentCommand = "`"$vsDevCmd`" -no_logo -arch=amd64 && set"
    $environmentLines = & $env:ComSpec /s /c $environmentCommand
    if ($LASTEXITCODE -ne 0) {
        throw "VsDevCmd.bat failed with exit code $LASTEXITCODE."
    }
    foreach ($line in $environmentLines) {
        $separator = $line.IndexOf('=')
        if ($separator -gt 0) {
            $name = $line.Substring(0, $separator)
            $value = $line.Substring($separator + 1)
            Set-Item -Path "Env:$name" -Value $value
        }
    }

    $cmakeDirectory = Join-Path $installationPath `
        'Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin'
    $ninjaDirectory = Join-Path $installationPath `
        'Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja'
    $env:Path = "$cmakeDirectory;$ninjaDirectory;$env:Path"

    foreach ($tool in @('cl.exe', 'cmake.exe', 'ninja.exe')) {
        if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
            throw "$tool was not found after initializing the Visual Studio toolchain."
        }
    }
}

function Get-ForgeIRActivePython {
    if (-not $env:VIRTUAL_ENV) {
        throw 'Activate a Python 3.11 virtual environment before running this script.'
    }

    $python = Join-Path $env:VIRTUAL_ENV 'Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
        throw "The active virtual environment has no Python executable at $python."
    }
    $env:Path = "$(Split-Path -Parent $python);$env:Path"
    return $python
}
