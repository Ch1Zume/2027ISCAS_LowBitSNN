param([string]$Downloader, [string]$Fixture, [string]$Destination)
$ErrorActionPreference = 'Stop'
$global:Attempts = @{}
$global:InterruptOnce = $true
function global:scp {
    param([Alias('P')][int]$Port,
          [Parameter(Position=0)][string]$Source,
          [Parameter(Position=1)][string]$Target)
    $Name = ($Source -split '/')[-1]
    if (-not $global:Attempts.ContainsKey($Name)) { $global:Attempts[$Name] = 0 }
    $global:Attempts[$Name]++
    if ($global:InterruptOnce -and $Name -like '*.part001') {
        $global:InterruptOnce = $false
        $global:LASTEXITCODE = 1
        return
    }
    Copy-Item -LiteralPath (Join-Path $Fixture $Name) -Destination $Target -Force
    $global:LASTEXITCODE = 0
}
$Interrupted = $false
try { & $Downloader -Destination $Destination -NoReceiptUpload } catch {
    if ($_.Exception.Message -notlike 'Download interrupted:*') { throw }
    $Interrupted = $true
}
if (-not $Interrupted) { throw 'Expected simulated interruption did not occur.' }
$First = Join-Path $Destination '20261013/archives/SNN-project-20261013.zip.part000'
if (-not (Test-Path -LiteralPath $First)) { throw 'Verified chunk was not retained after interruption.' }
& $Downloader -Destination $Destination -NoReceiptUpload
if ($global:Attempts['SNN-project-20261013.zip.part000'] -ne 1) { throw 'Resume unnecessarily downloaded an already verified chunk.' }
$Receipt = Get-Content (Join-Path $Destination '20261013/WINDOWS_RECEIPT.json') -Raw | ConvertFrom-Json
if ($Receipt.verified_payload_files -ne 3) { throw 'Payload verification count differs.' }
$Before = ($global:Attempts.GetEnumerator() | Where-Object { $_.Name -like '*.part*' } | Measure-Object -Property Value -Sum).Sum
Set-Content -LiteralPath (Join-Path $Destination '20261013/Algorithm/test.txt') -Value 'corrupted'
& $Downloader -Destination $Destination -NoReceiptUpload
$After = ($global:Attempts.GetEnumerator() | Where-Object { $_.Name -like '*.part*' } | Measure-Object -Property Value -Sum).Sum
if ($Before -ne $After) { throw 'Already verified archives were downloaded again.' }
$Receipt = Get-Content (Join-Path $Destination '20261013/WINDOWS_RECEIPT.json') -Raw | ConvertFrom-Json
if ($Receipt.verified_payload_files -ne 3) { throw 'Re-extraction verification failed.' }
[PSCustomObject]@{
    parser_and_execution = 'passed'
    interrupted_transfer_preserves_verified_chunks = $true
    resumed_transfer_skips_verified_chunks = $true
    verified_archives_reused = $true
    corrupted_extracted_file_repaired = $true
    unicode_payload_filename_verified = $true
    verified_payload_files = 3
    platform = [System.Runtime.InteropServices.RuntimeInformation]::OSDescription
    scope = 'Local mocked SCP integration; not actual Windows receipt'
} | ConvertTo-Json -Depth 5
