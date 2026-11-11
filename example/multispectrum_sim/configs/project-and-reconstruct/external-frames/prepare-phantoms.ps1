# Generate small local label phantoms. No downloads or optional packages.
$ErrorActionPreference = 'Stop'
$dest = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../../out/external-frames/phantoms'))
[IO.Directory]::CreateDirectory($dest) | Out-Null
$sphere = New-Object byte[] (64*64*64)
$slab = New-Object byte[] (64*64*64)
for ($z=0; $z -lt 64; $z++) {
    $pz=($z-31.5)*4
    for ($y=0; $y -lt 64; $y++) {
        $py=($y-31.5)*4
        for ($x=0; $x -lt 64; $x++) {
            $px=($x-31.5)*4
            $index=($z*64+$y)*64+$x
            $hole=(($px-28)*($px-28)+$py*$py+$pz*$pz -lt 12*12)
            if (($px*$px+$py*$py+$pz*$pz -le 64*64) -and !$hole) { $sphere[$index]=1 }
            if (($px*$px+$pz*$pz -le 64*64) -and ([Math]::Abs($py) -le 16) -and !$hole) { $slab[$index]=1 }
        }
    }
}
[IO.File]::WriteAllBytes((Join-Path $dest 'water-sphere.raw'),$sphere)
[IO.File]::WriteAllBytes((Join-Path $dest 'water-slab.raw'),$slab)
Write-Host "Prepared 64^3 uint8 phantoms (4 mm voxels): $dest"
