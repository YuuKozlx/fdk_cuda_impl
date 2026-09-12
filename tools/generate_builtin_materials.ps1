param(
    [Parameter(Mandatory=$true)][string]$InputDirectory,
    [Parameter(Mandatory=$true)][string]$OutputFile
)

$ErrorActionPreference = 'Stop'
$files = Get-ChildItem -LiteralPath $InputDirectory -File |
    Where-Object { $_.Name -ne 'Copyright.txt' }
$lines = [System.Collections.Generic.List[string]]::new()
$lines.Add('// Generated from the authorized MC-GPU material directory.')
$lines.Add('// Regenerate with tools/generate_builtin_materials.ps1; do not edit manually.')
$lines.Add('')

foreach ($file in $files) {
    $content = Get-Content -LiteralPath $file.FullName
    $density = ''
    for ($index = 0; $index -lt $content.Count; ++$index) {
        if ($content[$index] -match '^# Density:') {
            if ($index + 1 -lt $content.Count) {
                $density = [regex]::Match($content[$index + 1], '([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)').Value
            }
            break
        }
    }
    if ([string]::IsNullOrWhiteSpace($density)) {
        foreach ($line in $content) {
            $candidate = $line.Trim()
            if ($candidate -match '^(?=.*(?:\.|[eE]))\d+(?:\.\d+)?(?:[eE][-+]?\d+)?\s*$') {
                $density = $candidate
                break
            }
        }
    }
    if ([string]::IsNullOrWhiteSpace($density)) { continue }
    $pairs = @()
    foreach ($line in $content) {
        $m = [regex]::Match($line, '^\s*(\d+)\s+([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)(?:\s+#.*)?\s*$')
        if ($m.Success -and [double]$m.Groups[2].Value -gt 0) {
            $pairs += ($m.Groups[1].Value + ':' + $m.Groups[2].Value)
        }
    }
    if ($pairs.Count -eq 0) { continue }
    $key = $file.BaseName.ToLowerInvariant() -replace '[^a-z0-9]+','_'
    $isElement = $content -match '^# Elemental material:'
    $isTissue = $file.BaseName -match '^(ICRU|CIRS|ncat_)' -or
        $file.BaseName -match '^(blood|bone|brain|breast|muscle|lung|liver|kidney|heart|adipose|cartilage|spine|skull|skin|spleen|pancreas|testis)'
    if ($isElement) { $category = 'element'; $prefix = 'element' }
    elseif ($isTissue) { $category = 'human_tissue'; $prefix = 'tissue' }
    elseif ($file.BaseName -match '^(air|water_.*|brass|SS304)$') { $category = 'mixture'; $prefix = 'mixture' }
    else { $category = 'compound'; $prefix = 'compound' }
    $id = "$prefix.$key"
    $display = $file.BaseName.Replace('_',' ')
    $aliases = $file.BaseName.ToLowerInvariant()
    $composition = ($pairs -join ',')
    $lines.Add(('    {{"{0}", "{1}", "{2}", "{3}", {4}, "{5}"}},' -f $id,$category,$display,$aliases,$density,$composition))
}

$parent = Split-Path -Parent $OutputFile
if ($parent) { New-Item -ItemType Directory -Force -Path $parent | Out-Null }
Set-Content -LiteralPath $OutputFile -Value $lines -Encoding utf8
Write-Host "generated $($lines.Count - 3) material records: $OutputFile"
