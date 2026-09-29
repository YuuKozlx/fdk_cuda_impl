param(
    [Parameter(Mandatory=$true)][string]$InputFile,
    [Parameter(Mandatory=$true)][string]$OutputFile
)

$ErrorActionPreference = 'Stop'
$content = Get-Content -LiteralPath $InputFile
$elements = @{}
$symbols = @{}
$inMaterials = $false
foreach ($line in $content) {
    if ($line -match '^\[Materials\]') { $inMaterials = $true; continue }
    if ($inMaterials) { continue }
    $m = [regex]::Match($line, '^([^:]+):\s*S=\s*([A-Za-z]+)\s*;\s*Z=\s*(\d+)\.?\s*;\s*A=\s*([0-9.]+)')
    if ($m.Success) {
        $elements[$m.Groups[1].Value.Trim()] = @([int]$m.Groups[3].Value, [double]$m.Groups[4].Value)
        $symbols[$m.Groups[2].Value.Trim()] = @([int]$m.Groups[3].Value, [double]$m.Groups[4].Value)
    }
}

$records = [System.Collections.Generic.List[string]]::new()
$records.Add('// Generated from thirdparty/nist_xcom/data/GateMaterials.db.')
$records.Add('// Regenerate with tools/generate_gate_materials.ps1; do not edit manually.')
$records.Add('')
$current = $null

function Emit-Record($record) {
    if (-not $record) { return }
    $parts = @()
    $massTotal = 0.0
    foreach ($part in $record.Parts) {
        if ($part.Mode -eq 'f') { $mass = $part.Value }
        else { $mass = $part.Value * $part.AtomicWeight }
        if ($mass -gt 0) {
            $parts += [pscustomobject]@{ Z=$part.Z; Mass=$mass }
            $massTotal += $mass
        }
    }
    if ($parts.Count -eq 0 -or $massTotal -le 0) { return }
    $composition = ($parts | ForEach-Object { '{0}:{1:G12}' -f $_.Z, ($_.Mass / $massTotal) }) -join ','
    $key = $record.Name.ToLowerInvariant() -replace '[^a-z0-9]+','_'
    $tissues = 'lung|body|muscle|spinebone|ribbone|adipose|epidermis|hypodermis|blood|heart|kidney|liver|lymph|pancreas|intestine|skull|cartilage|brain|spleen|testis|breast'
    $allCounts = -not ($record.Parts | Where-Object { $_.Mode -ne 'n' })
    if ($record.Name -match "^($tissues)$") { $category = 'human_tissue' }
    elseif ($parts.Count -eq 1) { $category = 'element' }
    elseif ($allCounts) { $category = 'compound' }
    else { $category = 'mixture' }
    $records.Add(('    {{"gate.{0}", "{1}", "Gate {2}", "gate_{0}", {3:G12}, "{4}"}},' -f $key,$category,$record.Name,$record.Density,$composition))
}

foreach ($line in $content) {
    if ($line -match '^\[Materials\]') { $inMaterials = $true; continue }
    if (-not $inMaterials) { continue }
    $header = [regex]::Match($line, '^([^\s#][^:]*):\s*d\s*=\s*([0-9.eE+-]+)\s*(mg|g)/cm3')
    if ($header.Success) {
        Emit-Record $current
        $density = [double]$header.Groups[2].Value
        if ($header.Groups[3].Value -eq 'mg') { $density /= 1000.0 }
        $current = [pscustomobject]@{ Name=$header.Groups[1].Value.Trim(); Density=$density; Parts=@() }
        continue
    }
    if (-not $current) { continue }
    $part = [regex]::Match($line, '^\s*\+el:\s*name=([^;]+);\s*([fn])=([0-9.eE+-]+)')
    if (-not $part.Success) { continue }
    $name = $part.Groups[1].Value.Trim()
    if ($name -eq 'auto') { $name = $current.Name }
    $element = if ($elements.ContainsKey($name)) { $elements[$name] } elseif ($symbols.ContainsKey($name)) { $symbols[$name] } else { $null }
    if (-not $element) { throw "Unknown Gate element '$name' in $($current.Name)" }
    $current.Parts += [pscustomobject]@{ Z=$element[0]; AtomicWeight=$element[1]; Mode=$part.Groups[2].Value; Value=[double]$part.Groups[3].Value }
}
Emit-Record $current

$parent = Split-Path -Parent $OutputFile
if ($parent) { New-Item -ItemType Directory -Force -Path $parent | Out-Null }
Set-Content -LiteralPath $OutputFile -Value $records -Encoding utf8
Write-Host "generated $($records.Count - 3) Gate material records: $OutputFile"
