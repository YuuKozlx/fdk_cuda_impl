Get-ChildItem -Recurse -Include *.cpp,*.hpp,*.cuh,*.cu,*.h |
ForEach-Object {
    $bytes = [System.IO.File]::ReadAllBytes($_.FullName)
    
    # 检测是否已有 UTF-8 BOM（EF BB BF）
    $hasUtf8Bom = $bytes.Length -ge 3 -and
                  $bytes[0] -eq 0xEF -and
                  $bytes[1] -eq 0xBB -and
                  $bytes[2] -eq 0xBF

    if ($hasUtf8Bom) {
        Write-Host "Already UTF-8 BOM: $($_.Name)"
        return
    }

    # 检测是否是 UTF-16 BOM
    $hasUtf16Bom = $bytes.Length -ge 2 -and
                   (($bytes[0] -eq 0xFF -and $bytes[1] -eq 0xFE) -or
                    ($bytes[0] -eq 0xFE -and $bytes[1] -eq 0xFF))
    if ($hasUtf16Bom) {
        Write-Host "UTF-16, skip: $($_.Name)"
        return
    }

    # 尝试以 GBK 读取，转为 UTF-8 BOM
    try {
        $content = [System.IO.File]::ReadAllText(
            $_.FullName,
            [System.Text.Encoding]::GetEncoding(936))  # 936 = GBK
        [System.IO.File]::WriteAllText(
            $_.FullName,
            $content,
            [System.Text.UTF8Encoding]::new($true))    # true = with BOM
        Write-Host "Converted GBK -> UTF-8 BOM: $($_.Name)"
    }
    catch {
        Write-Host "Failed: $($_.Name) - $_"
    }
}