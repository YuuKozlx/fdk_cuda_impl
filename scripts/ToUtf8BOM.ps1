Get-ChildItem -Recurse -Include *.cpp,*.hpp,*.cuh,*.cu,*.h |
Where-Object { $_.FullName -notlike "*\thirdparty\*" } |
ForEach-Object {
    $bytes = [System.IO.File]::ReadAllBytes($_.FullName)

    # 已有 UTF-8 BOM，跳过
    $hasUtf8Bom = $bytes.Length -ge 3 -and
                  $bytes[0] -eq 0xEF -and
                  $bytes[1] -eq 0xBB -and
                  $bytes[2] -eq 0xBF
    if ($hasUtf8Bom) {
        Write-Host "Already UTF-8 BOM, skip: $($_.Name)"
        return
    }

    # UTF-16 BOM，跳过
    $hasUtf16Bom = $bytes.Length -ge 2 -and
                   (($bytes[0] -eq 0xFF -and $bytes[1] -eq 0xFE) -or
                    ($bytes[0] -eq 0xFE -and $bytes[1] -eq 0xFF))
    if ($hasUtf16Bom) {
        Write-Host "UTF-16, skip: $($_.Name)"
        return
    }

    # 优先尝试 UTF-8 无 BOM 解码（如果成功说明本来就是 UTF-8）
    $utf8 = [System.Text.UTF8Encoding]::new($false)  # false = 无 BOM
    $isUtf8 = $true
    try {
        # GetDecoder 严格模式，遇到非法字节会抛异常
        $decoder = $utf8.GetDecoder()
        $decoder.Fallback = [System.Text.DecoderExceptionFallback]::new()
        $charCount = $decoder.GetCharCount($bytes, 0, $bytes.Length)
        $chars = New-Object char[] $charCount
        $decoder.GetChars($bytes, 0, $bytes.Length, $chars, 0) | Out-Null
    }
    catch {
        $isUtf8 = $false
    }

    if ($isUtf8) {
        # 本来就是 UTF-8，只加 BOM
        $content = [System.Text.Encoding]::UTF8.GetString($bytes)
        [System.IO.File]::WriteAllText(
            $_.FullName, $content,
            [System.Text.UTF8Encoding]::new($true))
        Write-Host "Added BOM (was UTF-8): $($_.Name)"
    }
    else {
        # 不是合法 UTF-8，当 GBK 处理
        $content = [System.IO.File]::ReadAllText(
            $_.FullName,
            [System.Text.Encoding]::GetEncoding(936))
        [System.IO.File]::WriteAllText(
            $_.FullName, $content,
            [System.Text.UTF8Encoding]::new($true))
        Write-Host "Converted GBK -> UTF-8 BOM: $($_.Name)"
    }
}