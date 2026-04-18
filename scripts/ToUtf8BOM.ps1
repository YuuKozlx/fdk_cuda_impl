Get-ChildItem -Recurse -Include *.cpp,*.hpp,*.cuh,*.cu,*.h,*.c |
ForEach-Object {
    $content = Get-Content $_.FullName -Raw -Encoding Default
    [System.IO.File]::WriteAllText($_.FullName, $content, [System.Text.UTF8Encoding]::new($true))
}