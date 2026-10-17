param(
    [string]$Source = (Join-Path $PSScriptRoot '../data/materials/icrp145/media.csv'),
    [string]$Output = (Join-Path $PSScriptRoot '../example/multispectrum_sim/src/BuiltinIcrpMaterials.inc'),
    [switch]$Check
)
$ErrorActionPreference = 'Stop'
$culture = [Globalization.CultureInfo]::InvariantCulture
$elements = [ordered]@{ H=1; C=6; N=7; O=8; Na=11; Mg=12; P=15; S=16; Cl=17; Ar=18; K=19; Ca=20; Fe=26; I=53 }
$rows = @(Import-Csv -LiteralPath $Source)
if ($rows.Count -ne 105) { throw 'Expected 104 sex-specific ICRP145 media plus outside air.' }
$ids = @{}
$lines = [Collections.Generic.List[string]]::new()
$lines.Add('// Generated from data/materials/icrp145/media.csv; do not edit manually.')
$lines.Add('// Regenerate: powershell -File tools/generate_icrp_materials.ps1')
$lines.Add('// ICRP145 Annex B, Tables B.1/B.2, pp. 89-92; DOI 10.1177/0146645319893605.')
$lines.Add('// Published mass-fraction rounding is retained; XCOM normalizes the sum.')
function Escape-Cpp([string]$value) { return $value.Replace('\', '\\').Replace('"', '\"') }
foreach ($row in $rows) {
    if ($ids.ContainsKey($row.id)) { throw "Duplicate material ID: $($row.id)" }
    $ids[$row.id] = $true
    $medium = [int]$row.medium_id
    if ($row.sex -eq 'any') {
        if ($row.id -ne 'icrp145.air_outside' -or $medium -ne 0 -or $row.source -ne 'existing_air_preset') { throw 'Invalid external-air provenance.' }
        $display = 'ICRP145 outside air'
    } else {
        if ($row.sex -notin @('male','female') -or $medium -lt 1 -or $medium -gt 52 -or $row.id -ne "icrp145.$($row.sex).$medium") { throw "Invalid ICRP medium: $($row.id)" }
        $table = if ($row.sex -eq 'male') { 'ICRP145_B1' } else { 'ICRP145_B2' }
        if ($row.source -ne $table) { throw "Incorrect source table: $($row.id)" }
        $display = "ICRP145 $($row.sex) $($row.name)"
    }
    $density = [double]::Parse($row.density_g_cm3, $culture)
    if ([double]::IsNaN($density) -or [double]::IsInfinity($density) -or $density -le 0) { throw "Invalid density: $($row.id)" }
    $parts = [Collections.Generic.List[string]]::new()
    $sum = 0.0
    foreach ($element in $elements.Keys) {
        $fraction = [double]::Parse($row.$element, $culture)
        if ([double]::IsNaN($fraction) -or [double]::IsInfinity($fraction) -or $fraction -lt 0 -or $fraction -gt 1) { throw "Invalid $element fraction: $($row.id)" }
        $sum += $fraction
        if ($fraction -gt 0) { $parts.Add("$($elements[$element]):$($fraction.ToString('G12', $culture))") }
    }
    if ([Math]::Abs($sum - 1) -gt 0.0015) { throw "Mass fractions do not sum to one: $($row.id): $sum" }
    $lines.Add(('    {{"{0}", "ICRP145", "{1}", "", {2}, "{3}"}},' -f (Escape-Cpp $row.id), (Escape-Cpp $display), $density.ToString('G12', $culture), ($parts -join ',')))
}
foreach ($sex in 'male','female') {
    foreach ($medium in 1..52) { if (-not $ids.ContainsKey("icrp145.$sex.$medium")) { throw "Missing $sex medium $medium" } }
}
$text = ($lines -join "`n") + "`n"
if ($Check) {
    if (-not (Test-Path -LiteralPath $Output) -or ([IO.File]::ReadAllText($Output).Replace("`r`n", "`n") -cne $text)) { throw 'ICRP145 generated material table is stale; regenerate it.' }
    Write-Output 'ICRP145 source data and generated table agree (105 entries).'
} else {
    [IO.File]::WriteAllText([IO.Path]::GetFullPath($Output), $text, [Text.UTF8Encoding]::new($false))
    Write-Output "Generated 105 material entries: $Output"
}
