param(
    [string]$Python = 'C:\Users\Lenovo\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe',
    [ValidateSet('cuda','cpu')][string]$Backend = 'cuda'
)
$ErrorActionPreference = 'Stop'
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
        & $Python -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed' }
    }
    $taskPython = Join-Path $PWD '.venv\Scripts\python.exe'
    $taskIndex = if ($Backend -eq 'cuda') { 'https://download.pytorch.org/whl/cu124' } else { 'https://download.pytorch.org/whl/cpu' }
    & $taskPython -m pip install torch==2.6.0 torchvision==0.21.0 --index-url $taskIndex
    if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed' }
    & $taskPython -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
    & $taskPython -c "import torch; print('Torch:',torch.__version__,'CUDA:',torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
    Write-Host 'Environment prepared. No training started.'
} finally { Pop-Location }
