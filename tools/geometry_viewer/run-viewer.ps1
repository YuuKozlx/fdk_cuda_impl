[CmdletBinding()]
param(
    [ValidateSet('flat-circular','cyl-circular','flat-helical','cyl-helical','source-orbit','nikon-tilted')]
    [string]$Geometry = 'cyl-helical',
    [int]$Frame = 0,
    [switch]$GantryFixed,
    [string]$VtkRuntimeDirectory = $env:VTK_RUNTIME_DIR,
    [string]$Screenshot = '',
    [string]$ExportDirectory = '',
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$ViewerArguments
)
$ErrorActionPreference = 'Stop'
$viewerRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$viewerExecutable = Join-Path $viewerRoot 'out/build/vtk-viewer/geometry_viewer.exe'
if (!(Test-Path -LiteralPath $viewerExecutable)) {
    throw 'Build the standalone viewer first; see tools/geometry_viewer/README.zh-CN.md.'
}
if ([string]::IsNullOrWhiteSpace($VtkRuntimeDirectory)) {
    $viewerLocalRuntime = Join-Path $viewerRoot 'out/vtk-env/Lib/site-packages/vtk.libs'
    if (Test-Path -LiteralPath $viewerLocalRuntime) { $VtkRuntimeDirectory = $viewerLocalRuntime }
}
$viewerSavedPath = $env:PATH
try {
    if (![string]::IsNullOrWhiteSpace($VtkRuntimeDirectory)) {
        $viewerResolvedRuntime = (Resolve-Path -LiteralPath $VtkRuntimeDirectory).Path
        $env:PATH = $viewerResolvedRuntime + ';' + $env:PATH
    }
    $viewerArgs = @('--geometry', $Geometry, '--frame', "$Frame")
    if ($GantryFixed) { $viewerArgs += '--gantry-fixed' }
    if ($Screenshot) { $viewerArgs += @('--screenshot', $Screenshot) }
    if ($ExportDirectory) { $viewerArgs += @('--export', $ExportDirectory) }
    if ($ViewerArguments) { $viewerArgs += $ViewerArguments }
    & $viewerExecutable @viewerArgs
    if ($LASTEXITCODE -ne 0) { throw "Viewer exited with code $LASTEXITCODE" }
}
finally { $env:PATH = $viewerSavedPath }
