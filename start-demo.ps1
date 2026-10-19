$ErrorActionPreference = 'Stop'

$projectRoot = $PSScriptRoot
$python = 'C:\Codex\venvs\bharat-rozgaar-sentinel\Scripts\python.exe'
$pidFile = Join-Path $projectRoot 'work\prototype-api.pid'
$logRoot = 'C:\Codex\logs'

if (-not (Test-Path -LiteralPath $python)) {
    throw "Project Python environment not found: $python"
}

$websiteIndex = Join-Path $projectRoot 'frontend\dist\index.html'
if (-not (Test-Path -LiteralPath $websiteIndex)) {
    throw "Built website not found. Build it with pnpm from the frontend folder before launching."
}

New-Item -ItemType Directory -Force -Path @(
    'C:\Codex\tmp',
    'C:\Codex\cache\matplotlib',
    'C:\Codex\cache\huggingface',
    'C:\Codex\cache\torch',
    'C:\Codex\logs',
    (Join-Path $projectRoot 'work')
) | Out-Null

$env:BRS_MODEL = 'baseline'
$env:MPLCONFIGDIR = 'C:\Codex\cache\matplotlib'
$env:HF_HOME = 'C:\Codex\cache\huggingface'
$env:HF_HUB_CACHE = 'C:\Codex\cache\huggingface\hub'
$env:TORCH_HOME = 'C:\Codex\cache\torch'
$env:BRS_TMP_DIR = 'C:\Codex\tmp'

$url = 'http://127.0.0.1:8000/'

function Open-DemoPage {
    try {
        Start-Process -FilePath $url -ErrorAction Stop | Out-Null
    } catch {
        Write-Warning "The server is ready, but Windows could not open a browser automatically. Open $url manually."
    }
}

try {
    $health = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/health' -TimeoutSec 2
    if ($health.status -eq 'ready' -and $health.model -eq 'baseline') {
        $schema = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/openapi.json' -TimeoutSec 3
        if ($schema.paths.PSObject.Properties.Name -contains '/extract') {
            Open-DemoPage
            Write-Host 'Bharat Rozgaar Sentinel website is already running at http://127.0.0.1:8000/'
            return
        }

        Write-Host 'The running demo predates document upload. Restarting the verified project server...'
        & (Join-Path $projectRoot 'stop-demo.ps1')
        Start-Sleep -Seconds 1
    }
} catch {
    if ($health -and $health.status -eq 'ready' -and $health.model -eq 'baseline') { throw }
    if ($_.Exception.Message -match 'predates document upload') { throw }
    # No healthy demo server; start one below.
}

$listener = Get-NetTCPConnection -State Listen -LocalPort 8000 -ErrorAction SilentlyContinue
if ($listener) {
    throw 'Port 8000 is already in use by another service. Close it or change the demo port before continuing.'
}

$outLog = Join-Path $logRoot 'prototype-api.out.log'
$errLog = Join-Path $logRoot 'prototype-api.err.log'
$server = Start-Process -FilePath $python `
    -ArgumentList @('-m', 'uvicorn', 'src.inference.api:app', '--host', '127.0.0.1', '--port', '8000') `
    -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput $outLog -RedirectStandardError $errLog
Set-Content -LiteralPath $pidFile -Value $server.Id -Encoding ascii

for ($attempt = 0; $attempt -lt 30; $attempt++) {
    if (-not (Get-Process -Id $server.Id -ErrorAction SilentlyContinue)) {
        $details = if (Test-Path -LiteralPath $errLog) { Get-Content -LiteralPath $errLog -Tail 20 | Out-String } else { '' }
        Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue
        throw "The demo API stopped during startup. $details"
    }
    try {
        $health = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/health' -TimeoutSec 2
        if ($health.status -eq 'ready' -and $health.model -eq 'baseline') {
            Open-DemoPage
            Write-Host 'Bharat Rozgaar Sentinel website is ready at http://127.0.0.1:8000/'
            Write-Host 'The full website and baseline assessment API are ready. Upload a PDF/PNG/JPG or paste posting details.'
            return
        }
    } catch {
        # Keep waiting while Uvicorn starts.
    }
    Start-Sleep -Seconds 1
}

Stop-Process -Id $server.Id -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue
$details = if (Test-Path -LiteralPath $errLog) { Get-Content -LiteralPath $errLog -Tail 20 | Out-String } else { '' }
throw "The demo API did not become ready within 30 seconds. $details"
