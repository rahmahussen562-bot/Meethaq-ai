<#
.SYNOPSIS
Import a signed Microsoft VC++ x64 runtime CAB into this project's local cache.
.DESCRIPTION
This explicit setup operation performs no download, installer action, service
change, or write to Windows directories. Existing runtime DLLs are never replaced.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$CabPath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$projectRoot = [IO.Path]::GetFullPath($PSScriptRoot)
$runtimeRoot = Join-Path $projectRoot ".cache\native-runtime"
$destination = Join-Path $runtimeRoot "dll"
$minimumVersion = [version]"14.44.0.0"
$maxExpandedBytes = 32MB
$diskReserveBytes = 64MB
$expectedNames = @(
    "concrt140.dll", "msvcp140.dll", "msvcp140_1.dll", "msvcp140_2.dll",
    "msvcp140_atomic_wait.dll", "msvcp140_codecvt_ids.dll", "vcamp140.dll",
    "vccorlib140.dll", "vcomp140.dll", "vcruntime140.dll",
    "vcruntime140_1.dll", "vcruntime140_threads.dll"
)
$stageRoot = $null
$runtimeProcess = $null

function Assert-ProjectPath([string]$Path) {
    $absolute = [IO.Path]::GetFullPath($Path)
    $prefix = $projectRoot.TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
    if (-not $absolute.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Runtime setup paths must remain inside the project directory."
    }
    return $absolute
}

function Assert-CacheParents {
    foreach ($directory in @((Join-Path $projectRoot ".cache"), $runtimeRoot, $destination)) {
        [void](Assert-ProjectPath $directory)
        if (Test-Path -LiteralPath $directory) {
            $item = Get-Item -LiteralPath $directory -Force
            if (-not $item.PSIsContainer -or ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
                throw "A runtime cache directory is a file or filesystem link: $directory"
            }
        }
    }
}

function Assert-FreeSpace([long]$RequiredBytes) {
    $driveRoot = [IO.Path]::GetPathRoot($projectRoot)
    $drive = [IO.DriveInfo]::new($driveRoot)
    if ($drive.AvailableFreeSpace -lt ($RequiredBytes + $diskReserveBytes)) {
        throw "Insufficient disk space for runtime setup and its 64 MiB reserve."
    }
}

function Get-VerifiedDlls([string]$Directory, [bool]$CabNames) {
    $items = @(Get-ChildItem -LiteralPath $Directory -Force)
    $allowed = @($expectedNames)
    if ($CabNames) { $allowed = @($expectedNames | ForEach-Object { $_ + "_amd64" }) }
    $payload = @($items | Where-Object { $_.Name -ne "provisioned.json" })
    if ($CabNames -and $payload.Count -ne $items.Count) {
        throw "The extracted CAB unexpectedly contains a manifest."
    }
    if ($payload.Count -ne $allowed.Count) {
        throw "The runtime cache must contain exactly the twelve expected DLLs."
    }
    $hashes = [ordered]@{}
    $versions = [ordered]@{}
    [long]$total = 0
    foreach ($item in $items) {
        if ($item.PSIsContainer -or ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
            throw "Directories and filesystem links are not allowed inside the runtime cache."
        }
        if ($item.Name -eq "provisioned.json") { continue }
        if ($item.Name -cnotin $allowed -or $item.Length -le 0) {
            throw "An unexpected or empty file was found in the runtime cache."
        }
        $total += $item.Length
        if ($total -gt $maxExpandedBytes) { throw "The extracted runtime exceeds its 32 MiB limit." }
        $signature = Get-AuthenticodeSignature -LiteralPath $item.FullName
        if ($signature.Status -ne "Valid" -or $null -eq $signature.SignerCertificate -or
            $signature.SignerCertificate.Subject -notmatch "(?:^|,\s*)O=Microsoft Corporation(?:,|$)") {
            throw "The runtime file does not have a valid Microsoft signature: $($item.Name)"
        }
        $fileVersion = [version]$item.VersionInfo.FileVersion
        if ($fileVersion -lt $minimumVersion) {
            throw "The runtime requires Microsoft VC++ version 14.44 or later: $($item.Name)"
        }
        $reader = [IO.BinaryReader]::new([IO.File]::OpenRead($item.FullName))
        try {
            if ($reader.ReadUInt16() -ne 0x5A4D) { throw "Runtime file is not a PE DLL." }
            $reader.BaseStream.Position = 0x3C
            $peOffset = $reader.ReadInt32()
            if ($peOffset -lt 64 -or $peOffset -gt ($reader.BaseStream.Length - 6)) {
                throw "Runtime file has an invalid PE header."
            }
            $reader.BaseStream.Position = $peOffset
            if ($reader.ReadUInt32() -ne 0x00004550 -or $reader.ReadUInt16() -ne 0x8664) {
                throw "Runtime file is not an x64 Windows DLL."
            }
        }
        finally { $reader.Dispose() }
        $name = if ($CabNames) { $item.Name.Substring(0, $item.Name.Length - "_amd64".Length) } else { $item.Name }
        $hashes[$name] = (Get-FileHash -LiteralPath $item.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        $versions[$name] = $fileVersion.ToString()
    }
    $runtimeVersion = $versions["msvcp140.dll"]
    foreach ($version in $versions.Values) {
        if ($version -ne $runtimeVersion) { throw "All runtime DLLs must belong to the same Microsoft release." }
    }
    return [pscustomobject]@{ Hashes = $hashes; Versions = $versions; RuntimeVersion = $runtimeVersion }
}

try {
    if ([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT) {
        throw "This optional runtime importer is for Windows only."
    }
    Assert-CacheParents
    $cabItem = Get-Item -LiteralPath $CabPath -Force
    if ($cabItem.PSIsContainer -or $cabItem.Length -le 0 -or $cabItem.Length -gt 16MB) {
        throw "Supply the Microsoft vcRuntimeMinimum_amd64 CAB, no larger than 16 MiB."
    }
    $cabAbsolute = $cabItem.FullName
    $expand = Join-Path ([Environment]::GetFolderPath("Windows")) "System32\expand.exe"
    $listing = @(& $expand "-D" $cabAbsolute 2>&1)
    if ($LASTEXITCODE -ne 0) { throw "Windows expand could not inspect the supplied CAB." }
    $prefixPattern = "^" + [regex]::Escape($cabAbsolute) + ":\s+(?<Entry>.+)$"
    $entries = @($listing | ForEach-Object {
        if ([string]$_ -match $prefixPattern) { $Matches["Entry"].Trim() }
    })
    $cabExpected = @($expectedNames | ForEach-Object { $_ + "_amd64" })
    if ($entries.Count -ne $cabExpected.Count -or @($entries | Select-Object -Unique).Count -ne $cabExpected.Count) {
        throw "The CAB must contain exactly the twelve expected x64 runtime entries."
    }
    foreach ($entry in $entries) {
        if ($entry -cnotin $cabExpected) { throw "The CAB contains an unexpected path or filename." }
    }
    Assert-FreeSpace $maxExpandedBytes
    [void](New-Item -ItemType Directory -Path $runtimeRoot -Force)
    $stageRoot = Assert-ProjectPath (Join-Path $runtimeRoot ("import-" + [guid]::NewGuid().ToString("N")))
    $rawDirectory = Join-Path $stageRoot "raw"
    [void](New-Item -ItemType Directory -Path $rawDirectory -Force)
    # Start only the fixed Windows extractor; quote paths for its native argv.
    $arguments = @("-F:*", ('"{0}"' -f $cabAbsolute), ('"{0}"' -f $rawDirectory))
    $runtimeProcess = Start-Process -FilePath $expand -ArgumentList $arguments -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $stageRoot "expand.stdout") `
        -RedirectStandardError (Join-Path $stageRoot "expand.stderr")
    # Retain the native handle even when a small CAB exits before the first poll.
    $null = $runtimeProcess.Handle
    $deadline = [DateTime]::UtcNow.AddSeconds(60)
    while (-not $runtimeProcess.HasExited) {
        $expanded = @(Get-ChildItem -LiteralPath $rawDirectory -File -Force)
        $expandedBytes = 0L
        foreach ($file in $expanded) { $expandedBytes += $file.Length }
        if (($expandedBytes -gt $maxExpandedBytes) -or [DateTime]::UtcNow -gt $deadline) {
            throw "Runtime extraction exceeded its size or time limit."
        }
        Assert-FreeSpace 0
        Start-Sleep -Milliseconds 100
        $runtimeProcess.Refresh()
    }
    $runtimeProcess.WaitForExit()
    if ($runtimeProcess.ExitCode -ne 0) { throw "Windows expand failed to extract the supplied CAB." }
    $verified = Get-VerifiedDlls $rawDirectory $true
    if (Test-Path -LiteralPath $destination) {
        $current = Get-VerifiedDlls $destination $false
        foreach ($name in $expectedNames) {
            if ($current.Hashes[$name] -ne $verified.Hashes[$name]) {
                throw "The existing runtime cache differs from this CAB. No loaded DLLs were replaced."
            }
        }
    }
    else {
        $stagedDlls = Join-Path $stageRoot "dll"
        [void](New-Item -ItemType Directory -Path $stagedDlls)
        foreach ($name in $expectedNames) {
            $source = Join-Path $rawDirectory ($name + "_amd64")
            Move-Item -LiteralPath $source -Destination (Join-Path $stagedDlls $name)
        }
        [void](Get-VerifiedDlls $stagedDlls $false)
        Assert-CacheParents
        $publishSource = Assert-ProjectPath $stagedDlls
        $publishTarget = Assert-ProjectPath $destination
        [IO.Directory]::Move($publishSource, $publishTarget)
    }
    $manifest = [ordered]@{
        schema_version = 1
        runtime_version = $verified.RuntimeVersion
        minimum_version = $minimumVersion.ToString()
        files = $verified.Hashes
        file_versions = $verified.Versions
    } | ConvertTo-Json -Depth 5
    Assert-FreeSpace 4096
    $manifestPath = Join-Path $destination "provisioned.json"
    $stagedManifest = Assert-ProjectPath (Join-Path $stageRoot "provisioned.json")
    [IO.File]::WriteAllText($stagedManifest, $manifest, [Text.UTF8Encoding]::new($false))
    if (Test-Path -LiteralPath $manifestPath) {
        $manifestBackup = Assert-ProjectPath (Join-Path $stageRoot "previous-manifest.json")
        [IO.File]::Replace($stagedManifest, $manifestPath, $manifestBackup)
    }
    else { [IO.File]::Move($stagedManifest, $manifestPath) }
    Write-Output ([IO.Path]::GetFullPath($destination))
}
finally {
    if ($null -ne $runtimeProcess -and -not $runtimeProcess.HasExited) {
        # This handle belongs to the extractor started above, never an existing service.
        $runtimeProcess.Kill()
        $runtimeProcess.WaitForExit()
    }
    if ($null -ne $runtimeProcess) { $runtimeProcess.Dispose() }
    if ($null -ne $stageRoot -and (Test-Path -LiteralPath $stageRoot)) {
        $checkedStage = Assert-ProjectPath $stageRoot
        $allowedPrefix = [IO.Path]::GetFullPath($runtimeRoot).TrimEnd("\") + "\import-"
        if (-not $checkedStage.StartsWith($allowedPrefix, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Refusing to clean a path outside the importer staging directory."
        }
        Remove-Item -LiteralPath $checkedStage -Recurse -Force
    }
}
