[CmdletBinding()]
param(
    [int]$BackendPort = 8000,
    [string]$OllamaModel = "llama3.2:3b",
    [int]$StartupTimeoutSeconds = 240,
    [switch]$DeployWorker,
    [string]$WorkerName = "meethaq-ai",
    [string]$WranglerVersion = "3.114.15"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"
$cloudflaredPath = Join-Path $projectRoot "cloudflared.exe"
$logsDirectory = Join-Path $projectRoot "logs"
$runtimeDirectory = Join-Path $projectRoot ".runtime"
$backendOrigin = "http://127.0.0.1:$BackendPort"
$ollamaOrigin = if ($env:MEETHAQ_OLLAMA_URL) {
    $env:MEETHAQ_OLLAMA_URL.TrimEnd("/")
} else {
    "http://127.0.0.1:11434"
}

foreach ($required in @($pythonPath, $cloudflaredPath)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required executable not found: $required"
    }
}
New-Item -ItemType Directory -Force -Path $logsDirectory, $runtimeDirectory | Out-Null

function Get-OllamaModels {
    try {
        $payload = Invoke-RestMethod -Uri "$ollamaOrigin/api/tags" -TimeoutSec 10
        return @($payload.models | ForEach-Object { $_.name })
    } catch {
        return $null
    }
}

$ollamaProcess = $null
$models = Get-OllamaModels
if ($null -eq $models) {
    $ollamaCommand = Get-Command ollama.exe -ErrorAction SilentlyContinue
    if (-not $ollamaCommand) {
        throw "Ollama is unreachable at $ollamaOrigin and ollama.exe was not found. Install/start Ollama, then run: ollama pull $OllamaModel"
    }
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $ollamaProcess = Start-Process -FilePath $ollamaCommand.Source -ArgumentList @("serve") `
        -WorkingDirectory $projectRoot -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $logsDirectory "ollama-$stamp.out.log") `
        -RedirectStandardError (Join-Path $logsDirectory "ollama-$stamp.err.log") -PassThru
    $deadline = (Get-Date).AddSeconds($StartupTimeoutSeconds)
    do {
        Start-Sleep -Seconds 2
        $models = Get-OllamaModels
    } while ($null -eq $models -and (Get-Date) -lt $deadline -and -not $ollamaProcess.HasExited)
    if ($null -eq $models) {
        throw "Ollama did not become reachable. Review logs/ollama-$stamp.err.log."
    }
}
if ($models -notcontains $OllamaModel) {
    throw "Ollama is running but model '$OllamaModel' is missing. Run: ollama pull $OllamaModel"
}

function Get-BackendHealth {
    try {
        return Invoke-RestMethod -Uri "$backendOrigin/api/health" -TimeoutSec 15
    } catch {
        return $null
    }
}

$backendProcess = $null
$health = Get-BackendHealth
if ($health -and ($null -eq $health.llm_ready -or $health.llm_ready -ne $true)) {
    throw "Port $BackendPort is occupied by an unready or outdated backend. Stop that process and rerun this script."
}
if (-not $health) {
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $backendProcess = Start-Process -FilePath $pythonPath -ArgumentList @("-B", "api.py") `
        -WorkingDirectory $projectRoot -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $logsDirectory "backend-$stamp.out.log") `
        -RedirectStandardError (Join-Path $logsDirectory "backend-$stamp.err.log") -PassThru
    $deadline = (Get-Date).AddSeconds($StartupTimeoutSeconds)
    do {
        Start-Sleep -Seconds 2
        $health = Get-BackendHealth
    } while (-not $health -and (Get-Date) -lt $deadline -and -not $backendProcess.HasExited)
    if (-not $health) {
        throw "FastAPI did not become healthy. Review logs/backend-$stamp.err.log."
    }
}
if ($health.ready -ne $true -or $health.llm_ready -ne $true) {
    throw "Backend health failed: $($health | ConvertTo-Json -Depth 5 -Compress)"
}

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$tunnelOut = Join-Path $logsDirectory "cloudflared-$stamp.out.log"
$tunnelError = Join-Path $logsDirectory "cloudflared-$stamp.err.log"
$tunnelProcess = Start-Process -FilePath $cloudflaredPath `
    -ArgumentList @("tunnel", "--protocol", "http2", "--url", $backendOrigin, "--no-autoupdate") `
    -WorkingDirectory $projectRoot -WindowStyle Hidden `
    -RedirectStandardOutput $tunnelOut -RedirectStandardError $tunnelError -PassThru

$tunnelOrigin = $null
$deadline = (Get-Date).AddSeconds($StartupTimeoutSeconds)
do {
    Start-Sleep -Seconds 2
    if (Test-Path -LiteralPath $tunnelError) {
        $match = Select-String -Path $tunnelError `
            -Pattern "https://[a-z0-9-]+\.trycloudflare\.com" -AllMatches |
            Select-Object -Last 1
        if ($match) {
            $tunnelOrigin = $match.Matches[0].Value
        }
    }
} while (-not $tunnelOrigin -and (Get-Date) -lt $deadline -and -not $tunnelProcess.HasExited)
if (-not $tunnelOrigin) {
    throw "Cloudflare Tunnel did not publish a URL. Review $tunnelError."
}

if ($DeployWorker) {
    $pnpmCommand = Get-Command pnpm.cmd -ErrorAction Stop
    $previousViteApiUrl = $env:VITE_API_URL
    try {
        $env:VITE_API_URL = ""
        & $pnpmCommand.Source run build
        if ($LASTEXITCODE -ne 0) { throw "Frontend production build failed." }
        $wranglerPackage = "wrangler@$WranglerVersion"
        $tunnelOrigin | & $pnpmCommand.Source dlx $wranglerPackage secret put MEETHAQ_API_ORIGIN --name $WorkerName
        if ($LASTEXITCODE -ne 0) { throw "Updating MEETHAQ_API_ORIGIN failed." }
        & $pnpmCommand.Source dlx $wranglerPackage deploy --name $WorkerName
        if ($LASTEXITCODE -ne 0) { throw "Cloudflare Worker deployment failed." }
    } finally {
        $env:VITE_API_URL = $previousViteApiUrl
    }
}

$runtime = [ordered]@{
    started_at = (Get-Date).ToString("o")
    backend_origin = $backendOrigin
    backend_pid = if ($backendProcess) { $backendProcess.Id } else { $null }
    backend_ready = $health.ready
    indexed_chunks = $health.indexed_chunks
    calibration_status = $health.calibration_status
    llm_model = $health.llm_model
    llm_ready = $health.llm_ready
    ollama_pid = if ($ollamaProcess) { $ollamaProcess.Id } else { $null }
    tunnel_origin = $tunnelOrigin
    tunnel_pid = $tunnelProcess.Id
    tunnel_protocol = "http2"
    worker_deployed = [bool]$DeployWorker
}
$runtimePath = Join-Path $runtimeDirectory "production.json"
$runtime | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $runtimePath -Encoding UTF8

Write-Output "Backend: $backendOrigin"
Write-Output "Backend tunnel: $tunnelOrigin"
Write-Output "Indexed chunks: $($health.indexed_chunks)"
Write-Output "Ollama: $($health.llm_model) ($($health.llm_status))"
Write-Output "Runtime metadata: $runtimePath"
if ($DeployWorker) {
    Write-Output "Worker: https://$WorkerName.rahmahussen562.workers.dev"
}
