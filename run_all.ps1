# Runs the whole Task 1 pipeline in order (Windows PowerShell).
#   powershell -ExecutionPolicy Bypass -File run_all.ps1
#
# - Finished steps are skipped (their metrics file exists), so you can always just run this again.
# - A failed step is retried automatically (up to 3 times, 1 minute apart).
# - The GraphRAG and LightRAG indexes are built at the same time (they share the gateway's
#   pool of free model slots).
# - Remote restart: if the file control\restart.flag appears, the running steps are stopped and
#   started again (used to load a bug fix without anyone touching the keyboard).
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$py = (Resolve-Path ".\.venv\Scripts\python.exe").Path
$env:PYTHONIOENCODING = "utf-8"
New-Item -ItemType Directory -Force -Path "control" | Out-Null
New-Item -ItemType Directory -Force -Path "logs" | Out-Null
$flag = Join-Path $PSScriptRoot "control\restart.flag"
if (Test-Path $flag) { Remove-Item $flag -Force }

# ---- the gateway runs as its own program for the whole run (all steps share it) ----
$script:gw = $null
function Start-Gateway {
    if ($script:gw -and -not $script:gw.HasExited) { return }
    $script:gw = Start-Process -FilePath $py -ArgumentList "gateway.py" -NoNewWindow -PassThru `
        -RedirectStandardError "logs\gateway_stderr.txt" -RedirectStandardOutput "logs\gateway_stdout.txt"
    for ($i = 0; $i -lt 60; $i++) {
        try {
            $wc = New-Object System.Net.WebClient
            $wc.Proxy = $null                       # never send local traffic through the VPN proxy
            $null = $wc.DownloadString("http://127.0.0.1:8800/health")
            return
        } catch { Start-Sleep -Milliseconds 500 }
    }
    throw "The gateway did not start - see logs\gateway_stderr.txt"
}
function Stop-Gateway {
    if ($script:gw -and -not $script:gw.HasExited) { & taskkill.exe /PID $script:gw.Id /T /F | Out-Null }
    $script:gw = $null
}

function Invoke-Steps($items) {
    # Starts several python scripts at once and waits for all of them.
    # Returns "restart" when the flag appears, otherwise a list of the scripts that failed.
    $procs = @()
    foreach ($it in $items) {
        Write-Host "`n==== $($it.Title) ====" -ForegroundColor Cyan
        $p = Start-Process -FilePath $py -ArgumentList $it.Script -NoNewWindow -PassThru
        $null = $p.Handle   # makes ExitCode available after the process ends
        $procs += [pscustomobject]@{ Item = $it; Proc = $p }
    }
    while (@($procs | Where-Object { -not $_.Proc.HasExited }).Count -gt 0) {
        if (Test-Path $flag) {
            Remove-Item $flag -Force
            Write-Host "`n>>> Restart requested - stopping and starting again with the new code <<<" -ForegroundColor Yellow
            foreach ($x in $procs) { if (-not $x.Proc.HasExited) { & taskkill.exe /PID $x.Proc.Id /T /F | Out-Null } }
            Stop-Gateway
            Start-Sleep -Seconds 3
            Start-Gateway
            return "restart"
        }
        if ($script:gw.HasExited) { Write-Host "Gateway stopped - starting it again" -ForegroundColor Yellow; Start-Gateway }
        Start-Sleep -Seconds 5
    }
    $failed = @()
    foreach ($x in $procs) {
        $x.Proc.WaitForExit()
        if ($x.Proc.ExitCode -ne 0) { $failed += $x.Item }
    }
    return ,$failed
}

function Steps($items) {
    $todo = @($items | Where-Object { -not ($_.Done -and (Test-Path $_.Done)) })
    foreach ($it in $items) {
        if ($it.Done -and (Test-Path $it.Done)) { Write-Host "`n==== $($it.Title) - already done, skipped ====" -ForegroundColor DarkGray }
    }
    $failures = 0
    while ($todo.Count -gt 0) {
        $result = Invoke-Steps $todo
        if ($result -is [string] -and $result -eq "restart") {
            $todo = @($todo | Where-Object { -not ($_.Done -and (Test-Path $_.Done)) })
            continue
        }
        $todo = @($result)
        if ($todo.Count -eq 0) { return }
        $failures += 1
        $names = ($todo | ForEach-Object { $_.Title }) -join ", "
        if ($failures -ge 3) { Stop-Gateway; throw "$names failed 3 times - see the error messages above" }
        Write-Host "$names stopped with an error - trying again in 60 seconds ($failures/3)" -ForegroundColor Yellow
        Start-Sleep -Seconds 60
    }
}
function S($title, $script, $done) { [pscustomobject]@{ Title = $title; Script = $script; Done = $done } }

Start-Gateway
try {
    Steps @(S "0. Test the free LLM and the embedding model" "check_setup.py" $null)
    Steps @(S "1. PDFs -> clean text" "01_prepare_data.py" "metrics\corpus_stats.csv")
    Steps @(S "2. Build plain RAG index" "02_build_rag.py" "metrics\rag_index.json")
    Steps @((S "3. Build GraphRAG index" "03_build_graphrag.py" "metrics\graphrag_index.json"),
            (S "4. Build LightRAG index (at the same time)" "04_build_lightrag.py" "metrics\lightrag_index.json"))
    Steps @(S "5. Evaluate the engines on 30 questions" "05_evaluate.py" "eval\dialogue_test.csv")
    Steps @(S "6. Make report tables and figures" "06_make_report_assets.py" $null)
} finally {
    Stop-Gateway
}
Write-Host "`nAll done. Start the chat app with:  .\.venv\Scripts\python.exe app.py" -ForegroundColor Green
