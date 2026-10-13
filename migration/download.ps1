param(
    [string]$Server = 'IMECAS-yufan',
    [string]$Destination = 'E:\IMECAS_SNN',
    [int]$Port = 22,
    [switch]$NoReceiptUpload
)
$ErrorActionPreference = 'Stop'
$Remote = '/home/zhangyufan/Project/1_2026Q3_T22_SNNCIM/2_Ruiqi/Algorithm/migration/releases/20261013'
$Release = Join-Path $Destination '20261013'
$Archives = Join-Path $Release 'archives'
$Project = Join-Path $Release 'Algorithm'
foreach ($Command in @('scp', 'tar')) {
    if (-not (Get-Command $Command -ErrorAction SilentlyContinue)) {
        throw "$Command is required (Windows OpenSSH and Windows tar)."
    }
}
New-Item -ItemType Directory -Force -Path $Archives | Out-Null

function Get-RemoteFile([string]$Name, [string]$Target) {
    if ($Name -notmatch '^[A-Za-z0-9._-]+$') { throw "Invalid delivery filename: $Name" }
    & scp -P $Port ('{0}:{1}/{2}' -f $Server, $Remote, $Name) $Target
    if ($LASTEXITCODE -ne 0) { throw "Download interrupted: $Name. Re-run this script to resume." }
}

function Test-Digest([string]$Path, [string]$Expected) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $false }
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() -eq $Expected
}

# Refresh metadata so an old toolkit/index cannot select stale packages.
$IndexFile = Join-Path $Archives 'DOWNLOAD_INDEX.json'
Get-RemoteFile 'DOWNLOAD_INDEX.json' "$IndexFile.partial"
Move-Item -LiteralPath "$IndexFile.partial" -Destination $IndexFile -Force
$Index = Get-Content -LiteralPath $IndexFile -Raw -Encoding UTF8 | ConvertFrom-Json
if ($Index.schema -ne 1 -or $Index.release -ne '20261013') { throw 'Unexpected delivery index.' }
$Required = [long]$Index.compressed_bytes * 2 + [long]$Index.payload_bytes + 1GB
$DriveName = [System.IO.Path]::GetPathRoot([System.IO.Path]::GetFullPath($Destination))
$Drive = [System.IO.DriveInfo]::new($DriveName)
Write-Host ('Delivery commit: ' + $Index.git_commit)
Write-Host ('Conservative space estimate: {0:N1} GiB; free: {1:N1} GiB' -f ($Required / 1GB), ($Drive.AvailableFreeSpace / 1GB))
# Existing verified files make a resume require less space.
if ($Drive.AvailableFreeSpace -lt 2GB) { throw 'Less than 2 GiB free on the destination drive.' }

foreach ($Archive in $Index.archives) {
    if ($Archive.name -notmatch '^[A-Za-z0-9._-]+$') { throw 'Invalid archive filename.' }
    $Final = Join-Path $Archives $Archive.name
    if (Test-Digest $Final $Archive.sha256) {
        Write-Host ('Verified existing: ' + $Archive.name)
        continue
    }
    foreach ($Part in $Archive.parts) {
        if ($Part.name -notmatch '^[A-Za-z0-9._-]+$') { throw 'Invalid chunk filename.' }
        $PartPath = Join-Path $Archives $Part.name
        if (-not (Test-Digest $PartPath $Part.sha256)) {
            Write-Host ('Downloading: ' + $Part.name)
            Get-RemoteFile $Part.name "$PartPath.partial"
            if (-not (Test-Digest "$PartPath.partial" $Part.sha256)) { throw ('Chunk checksum mismatch: ' + $Part.name) }
            Move-Item -LiteralPath "$PartPath.partial" -Destination $PartPath -Force
        }
    }
    $Output = [System.IO.File]::Open("$Final.partial", [System.IO.FileMode]::Create, [System.IO.FileAccess]::Write)
    try {
        foreach ($Part in $Archive.parts) {
            $InputStream = [System.IO.File]::OpenRead((Join-Path $Archives $Part.name))
            try { $InputStream.CopyTo($Output, 4194304) } finally { $InputStream.Dispose() }
        }
    } finally { $Output.Dispose() }
    if (-not (Test-Digest "$Final.partial" $Archive.sha256)) { throw ('Archive checksum mismatch: ' + $Archive.name) }
    Move-Item -LiteralPath "$Final.partial" -Destination $Final -Force
    # Remove redundant transport chunks only after verifying the full archive.
    foreach ($Part in $Archive.parts) { Remove-Item -LiteralPath (Join-Path $Archives $Part.name) }
}

$Manifest = Join-Path $Archives 'PAYLOAD_FILES.json'
Get-RemoteFile 'PAYLOAD_FILES.json' "$Manifest.partial"
if (-not (Test-Digest "$Manifest.partial" $Index.payload_manifest.sha256)) { throw 'Payload manifest checksum mismatch.' }
Move-Item -LiteralPath "$Manifest.partial" -Destination $Manifest -Force

foreach ($Archive in $Index.archives) {
    $Path = Join-Path $Archives $Archive.name
    if ($Archive.kind -eq 'source') {
        Write-Host ('Extracting project: ' + $Archive.name)
        Expand-Archive -LiteralPath $Path -DestinationPath $Release -Force
    } elseif ($Archive.kind -eq 'payload') {
        Write-Host ('Extracting retained data/models/caches: ' + $Archive.name)
        & tar -xzf $Path -C $Release
        if ($LASTEXITCODE -ne 0) { throw ('Extraction failed: ' + $Archive.name) }
    }
}
# Keep Linux environment archives intact on Windows to preserve links and modes.
$Payload = Get-Content -LiteralPath $Manifest -Raw -Encoding UTF8 | ConvertFrom-Json
$Verified = 0
foreach ($Property in $Payload.files.PSObject.Properties) {
    $Relative = $Property.Name
    if ($Relative.StartsWith('/') -or $Relative.Contains('\') -or $Relative.Contains(':') -or ($Relative.Split('/') -contains '..')) {
        throw "Invalid payload path: $Relative"
    }
    $Path = Join-Path $Project $Relative.Replace('/', [System.IO.Path]::DirectorySeparatorChar)
    if (-not (Test-Digest $Path $Property.Value.sha256)) { throw "Missing/changed extracted file: $Relative" }
    $Verified++
    if (($Verified % 2000) -eq 0) { Write-Host "Verified extracted files: $Verified" }
}
$Receipt = [PSCustomObject]@{
    completed_at = (Get-Date).ToString('o')
    destination = $Project
    git_commit = $Index.git_commit
    verified_payload_files = $Verified
    archives = @($Index.archives | Select-Object name, bytes, sha256)
    linux_environment_archives_verified = $true
    linux_environment_executed_on_this_client = $false
    full_TEST_performed_on_this_client = $false
}
$ReceiptPath = Join-Path $Release 'WINDOWS_RECEIPT.json'
$Receipt | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $ReceiptPath -Encoding UTF8
if (-not $NoReceiptUpload) {
    & scp -P $Port $ReceiptPath ('{0}:{1}/WINDOWS_RECEIPT.json' -f $Server, $Remote)
    if ($LASTEXITCODE -ne 0) { Write-Warning 'Local verification passed; receipt upload failed. Keep WINDOWS_RECEIPT.json.' }
}
Write-Host "Download, extraction and SHA256 verification passed: $Project"
Write-Host "Verified files: $Verified; receipt: $ReceiptPath"
Write-Host "Restore manual: $Project\migration\RESTORE_MANUAL.md"
