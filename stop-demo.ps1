$ErrorActionPreference = 'Stop'
$pidFile = Join-Path $PSScriptRoot 'work\prototype-api.pid'
$python = 'C:\Codex\venvs\bharat-rozgaar-sentinel\Scripts\python.exe'
$approvedPythonPaths = @($python, 'C:\Codex\Python312\python.exe')
$serverPid = $null
if (Test-Path -LiteralPath $pidFile) {
    $recordedPid = [int](Get-Content -LiteralPath $pidFile -Raw).Trim()
    if (Get-Process -Id $recordedPid -ErrorAction SilentlyContinue) { $serverPid = $recordedPid }
}

if (-not $serverPid) {
    $listeners = Get-NetTCPConnection -State Listen -LocalPort 8000 -ErrorAction SilentlyContinue
    foreach ($listener in $listeners) {
        $candidate = Get-CimInstance Win32_Process -Filter "ProcessId=$($listener.OwningProcess)" -ErrorAction SilentlyContinue
        if ($candidate.ExecutablePath -in $approvedPythonPaths -and $candidate.CommandLine -match 'uvicorn.*src\.inference\.api:app') {
            $serverPid = [int]$candidate.ProcessId
            break
        }
    }
}

if (-not $serverPid) {
    if (Test-Path -LiteralPath $pidFile) { Remove-Item -LiteralPath $pidFile -Force }
    Write-Host 'No verified prototype server process is running.'
    return
}

$process = Get-Process -Id $serverPid -ErrorAction SilentlyContinue
try {
    $commandLine = (Get-CimInstance Win32_Process -Filter "ProcessId=$serverPid" -ErrorAction Stop).CommandLine
} catch {
    throw 'Could not verify the recorded process command line; leaving the process untouched.'
}

if ($process.Path -notin $approvedPythonPaths -or $commandLine -notmatch 'uvicorn.*src\.inference\.api:app') {
    throw 'The candidate process does not match the prototype server; leaving it untouched.'
}

Stop-Process -Id $serverPid
if (Test-Path -LiteralPath $pidFile) { Remove-Item -LiteralPath $pidFile -Force }
Write-Host 'Bharat Rozgaar Sentinel prototype server stopped.'
