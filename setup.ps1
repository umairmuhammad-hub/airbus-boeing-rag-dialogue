# One-time setup for Task 1 (Windows PowerShell).
# Run from the task1_rag folder:   powershell -ExecutionPolicy Bypass -File setup.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "`n[1/4] Creating the Python virtual environment (.venv) ..." -ForegroundColor Cyan
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    try { py -3.12 -m venv .venv } catch { python -m venv .venv }
}
$py = ".\.venv\Scripts\python.exe"
& $py --version

Write-Host "`n[2/4] Installing packages (GraphRAG, LightRAG, ...). This takes 5-15 minutes ..." -ForegroundColor Cyan
& $py -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
& $py -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
if ($LASTEXITCODE -ne 0) { throw "pip install failed - see the error messages above" }

Write-Host "`n[3/4] Checking Ollama and the bge-m3 embedding model ..." -ForegroundColor Cyan
$ollama = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
if (-not (Test-Path $ollama)) { $ollama = "ollama" }
& $ollama list

Write-Host "`n[4/4] Checking the settings file (.env) ..." -ForegroundColor Cyan
if (-not (Test-Path ".env")) { throw ".env is missing - copy env.example.txt to .env and put your Zhipu API key in it" }
& $py -c "import graphrag, lightrag, gradio, faiss; print('All packages import correctly')"
Write-Host "`nSetup finished. Next: powershell -ExecutionPolicy Bypass -File run_all.ps1" -ForegroundColor Green
