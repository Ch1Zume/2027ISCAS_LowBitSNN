param(
    [string]$Server = 'IMECAS-yufan',
    [string]$Destination = 'E:\IMECAS_SNN',
    [int]$Port = 22
)
$ErrorActionPreference = 'Stop'
$RemoteDirectory = '/home/zhangyufan/Project/1_2026Q3_T22_SNNCIM/2_Ruiqi/Algorithm/migration/downloads'
if (-not (Get-Command scp -ErrorAction SilentlyContinue)) { throw 'Windows OpenSSH scp is required.' }
New-Item -ItemType Directory -Force -Path $Destination | Out-Null
Push-Location $Destination
try {
    & scp -P $Port "${Server}:$RemoteDirectory/SHA256SUMS" '.\SHA256SUMS'
    if ($LASTEXITCODE -ne 0) { throw 'Cannot download SHA256SUMS.' }
    $Packages = @()
    foreach ($Line in Get-Content '.\SHA256SUMS') {
        if ($Line -notmatch '^([a-fA-F0-9]{64})\s+([A-Za-z0-9._-]+)$') { throw "Invalid checksum line: $Line" }
        $Expected = $Matches[1].ToLowerInvariant()
        $Name = $Matches[2]
        $LocalFile = Join-Path $Destination $Name
        $ValidExisting = (Test-Path -LiteralPath $LocalFile) -and ((Get-FileHash -LiteralPath $LocalFile -Algorithm SHA256).Hash.ToLowerInvariant() -eq $Expected)
        if (-not $ValidExisting) {
            Write-Host "Downloading $Name"
            & scp -P $Port "${Server}:$RemoteDirectory/$Name" ".\$Name"
            if ($LASTEXITCODE -ne 0) { throw "Download interrupted: $Name. Re-run this script." }
        }
        if ((Get-FileHash -LiteralPath $LocalFile -Algorithm SHA256).Hash.ToLowerInvariant() -ne $Expected) { throw "Checksum mismatch: $Name" }
        $Packages += [PSCustomObject]@{ name = $Name; sha256 = $Expected }
    }
    foreach ($Package in $Packages) {
        if ($Package.name.EndsWith('.zip')) {
            Write-Host "Extracting $($Package.name)"
            Expand-Archive -LiteralPath (Join-Path $Destination $Package.name) -DestinationPath $Destination -Force
        }
    }
    $ProjectDirectory = Join-Path $Destination 'Algorithm'
    $Manifest = Get-Content -LiteralPath (Join-Path $ProjectDirectory 'migration\payload_manifest.json') -Raw | ConvertFrom-Json
    $Verified = 0
    foreach ($Property in $Manifest.sha256.PSObject.Properties) {
        $RelativePath = $Property.Name.Replace('/', '\')
        if ($RelativePath.StartsWith('\') -or $RelativePath.Contains('..') -or $RelativePath.Contains(':')) { throw 'Invalid payload path.' }
        $Path = Join-Path $ProjectDirectory $RelativePath
        if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "Missing retained file: $RelativePath" }
        if ((Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $Property.Value) { throw "Payload changed: $RelativePath" }
        $Verified++
    }
    $Receipt = [PSCustomObject]@{
        completed_at = (Get-Date).ToString('o')
        destination = $ProjectDirectory
        verified_payload_files = $Verified
        packages = $Packages
        full_TEST_performed = $false
    }
    $Receipt | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $ProjectDirectory 'migration\WINDOWS_RECEIPT.json') -Encoding UTF8
    Write-Host "Download, extraction and checksums passed: $ProjectDirectory"
    Write-Host "Restore manual: $ProjectDirectory\migration\RESTORE_MANUAL.md"
} finally {
    Pop-Location
}
