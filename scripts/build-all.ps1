<#
.SYNOPSIS
    配置并编译 YKCBCT 的全部模块和测试目标。

.DESCRIPTION
    默认开启螺旋算法、CVP、Yktest、GoogleTest 单元测试和 DLL 接口测试。
    所有 CMake 功能开关集中在本脚本参数区，调用者可按需关闭单项，例如：

      .\scripts\build-all.ps1 -BuildCvp:$false
      .\scripts\build-all.ps1 -Configuration Debug -Fresh

    默认只编译，不自动运行耗时较长的算法测试。-RunUnitTests 只运行 CTest
    中注册的轻量单元测试；Yktest 由用户按具体 case/config 显式运行。
#>

[CmdletBinding()]
param(
    # --------------------------- 构建工具与输出 ---------------------------
    [ValidateSet("Debug", "Release", "RelWithDebInfo", "MinSizeRel")]
    [string]$Configuration = "Release",
    [string]$BuildDirectory = "",
    [string]$Generator = "Ninja",
    [ValidateRange(1, 256)]
    [int]$Jobs = [Math]::Max(1, [Environment]::ProcessorCount),
    [string]$CudaArchitectures = "86",
    [string]$VcpkgRoot = $env:VCPKG_ROOT,
    [string]$TomlPlusPlusIncludeDirectory = "",

    # ----------------------------- 功能开关 -------------------------------
    [bool]$BuildHelical = $true,
    [bool]$BuildCvp = $true,
    [bool]$BuildManualTests = $true,
    [bool]$BuildUnitTests = $true,
    [bool]$BuildDllTest = $true,

    # ----------------------------- 执行动作 -------------------------------
    [switch]$Fresh,
    [switch]$Install,
    [switch]$RunUnitTests
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function ConvertTo-CMakeBool([bool]$Value) {
    if ($Value) { return "ON" }
    return "OFF"
}

function Invoke-Checked([string]$Program, [string[]]$Arguments) {
    Write-Host "`n> $Program $($Arguments -join ' ')" -ForegroundColor Cyan
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "命令执行失败，退出码 $LASTEXITCODE：$Program"
    }
}

function Import-VisualStudioEnvironment {
    if (Get-Command cl.exe -ErrorAction SilentlyContinue) { return }

    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio/Installer/vswhere.exe"
    $vsInstall = ""
    if (Test-Path -LiteralPath $vswhere) {
        $vsInstall = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
            -property installationPath | Select-Object -First 1
    }
    if ([string]::IsNullOrWhiteSpace($vsInstall)) {
        throw "未找到 Visual Studio C++ x64 工具链。请安装 Desktop development with C++，或从 VS Developer PowerShell 运行本脚本。"
    }
    $vcvars = Join-Path $vsInstall "VC/Auxiliary/Build/vcvars64.bat"
    if (-not (Test-Path -LiteralPath $vcvars -PathType Leaf)) {
        throw "未找到 vcvars64.bat：$vcvars"
    }

    # vcvars 脚本只会修改其 cmd 子进程环境；将 `set` 输出回灌到当前
    # PowerShell，确保 nvcc 和 CMake 都能找到 cl/link/rc。
    $lines = cmd.exe /s /c "call `"$vcvars`" >nul && set"
    foreach ($line in $lines) {
        if ($line -match '^([^=]+)=(.*)$') {
            [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2])
        }
    }
    if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
        throw "Visual Studio 环境导入失败，PATH 中仍未找到 cl.exe。"
    }
    Write-Host "已导入 Visual Studio x64 C++ 环境：$vsInstall" -ForegroundColor DarkGreen
}

$SourceDirectory = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($BuildDirectory)) {
    $BuildDirectory = Join-Path $SourceDirectory "out/build/all-$Configuration"
}
elseif (-not [IO.Path]::IsPathRooted($BuildDirectory)) {
    $BuildDirectory = Join-Path $SourceDirectory $BuildDirectory
}
$BuildDirectory = [IO.Path]::GetFullPath($BuildDirectory)

$CMake = (Get-Command cmake -ErrorAction Stop).Source
Import-VisualStudioEnvironment
$ConfigureArguments = @(
    "-S", $SourceDirectory,
    "-B", $BuildDirectory,
    "-G", $Generator,
    "-DCMAKE_CUDA_ARCHITECTURES=$CudaArchitectures",
    "-DYKCBCT_BUILD_HELICAL=$(ConvertTo-CMakeBool $BuildHelical)",
    "-DYKCBCT_BUILD_CVP=$(ConvertTo-CMakeBool $BuildCvp)",
    "-DBUILD_TESTS=$(ConvertTo-CMakeBool $BuildManualTests)",
    "-DBUILD_UNIT_TESTS=$(ConvertTo-CMakeBool $BuildUnitTests)",
    "-DBUILD_DLLTEST=$(ConvertTo-CMakeBool $BuildDllTest)",
    "-DVCPKG_MANIFEST_MODE=ON",
    "-DVCPKG_INSTALLED_DIR=$SourceDirectory/thirdparty/vcpkg_installed"
)

# Ninja 是单配置生成器，构建类型必须在 configure 阶段确定。Visual Studio
# 等多配置生成器则在 build/install 阶段通过 --config 选择配置。
$IsMultiConfig = $Generator -match "Visual Studio|Xcode|Ninja Multi-Config"
if (-not $IsMultiConfig) {
    $ConfigureArguments += "-DCMAKE_BUILD_TYPE=$Configuration"
}

if (-not [string]::IsNullOrWhiteSpace($VcpkgRoot)) {
    $Toolchain = Join-Path $VcpkgRoot "scripts/buildsystems/vcpkg.cmake"
    if (-not (Test-Path -LiteralPath $Toolchain -PathType Leaf)) {
        throw "VCPKG_ROOT 无效，未找到 toolchain：$Toolchain"
    }
    $ConfigureArguments += "-DCMAKE_TOOLCHAIN_FILE=$Toolchain"
}
else {
    throw "全开构建需要 gtest 和 toml++。请设置 VCPKG_ROOT，或通过 -VcpkgRoot 指定 vcpkg。"
}

if (-not [string]::IsNullOrWhiteSpace($TomlPlusPlusIncludeDirectory)) {
    $TomlPlusPlusIncludeDirectory = [IO.Path]::GetFullPath(
        $TomlPlusPlusIncludeDirectory)
    $InstalledHeader = Join-Path $TomlPlusPlusIncludeDirectory "toml++/toml.hpp"
    if (-not (Test-Path -LiteralPath $InstalledHeader)) {
        throw "指定目录中未找到标准布局的 toml++/toml.hpp：$TomlPlusPlusIncludeDirectory"
    }
    $ConfigureArguments +=
        "-DYKCBCT_TOMLPLUSPLUS_INCLUDE_DIR=$TomlPlusPlusIncludeDirectory"
}

if ($Fresh) {
    # cmake --fresh 仅重建 CMake cache，不直接递归删除用户指定目录。
    $ConfigureArguments = @("--fresh") + $ConfigureArguments
}

Write-Host "YKCBCT 全开构建" -ForegroundColor Green
Write-Host "  Source       : $SourceDirectory"
Write-Host "  Build        : $BuildDirectory"
Write-Host "  Configuration: $Configuration"
Write-Host "  Generator    : $Generator"
Write-Host "  CUDA arch    : $CudaArchitectures"
Write-Host "  Helical/CVP  : $BuildHelical / $BuildCvp"
Write-Host "  Tests        : manual=$BuildManualTests unit=$BuildUnitTests dll=$BuildDllTest"

Invoke-Checked $CMake $ConfigureArguments

$BuildArguments = @("--build", $BuildDirectory, "--parallel", "$Jobs")
if ($IsMultiConfig) { $BuildArguments += @("--config", $Configuration) }
Invoke-Checked $CMake $BuildArguments

if ($RunUnitTests) {
    if (-not $BuildUnitTests) {
        throw "-RunUnitTests 要求 -BuildUnitTests:`$true。"
    }
    $CTest = (Get-Command ctest -ErrorAction Stop).Source
    $TestArguments = @("--test-dir", $BuildDirectory, "--output-on-failure")
    if ($IsMultiConfig) { $TestArguments += @("-C", $Configuration) }
    Invoke-Checked $CTest $TestArguments
}

if ($Install) {
    $InstallArguments = @("--install", $BuildDirectory)
    if ($IsMultiConfig) { $InstallArguments += @("--config", $Configuration) }
    Invoke-Checked $CMake $InstallArguments
}

Write-Host "`n全开构建完成：$BuildDirectory" -ForegroundColor Green
