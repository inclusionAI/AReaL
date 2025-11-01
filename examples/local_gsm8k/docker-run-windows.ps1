# PowerShell script to run AReaL GRPO Docker container on Windows 11 with CUDA GPU
# Prerequisites:
#   1. Docker Desktop for Windows with WSL2 backend
#   2. NVIDIA Container Toolkit installed
#   3. CUDA-capable GPU with drivers installed

param(
    [string]$ContainerName = "areal-grpo-container",
    [string]$ProjectPath = "C:\Users\$env:USERNAME\GT\CS7643_Deep_Learning\ProjectLLM\AReaL",
    [string]$SharedMemory = "8g",
    [switch]$RemoveExisting = $false
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "AReaL GRPO Docker Setup (Windows CUDA)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Docker is running
Write-Host "Checking Docker..." -ForegroundColor Yellow
try {
    docker ps | Out-Null
    Write-Host "✓ Docker is running" -ForegroundColor Green
} catch {
    Write-Host "✗ Docker is not running. Please start Docker Desktop." -ForegroundColor Red
    exit 1
}

# Check for NVIDIA GPU
Write-Host "Checking for NVIDIA GPU..." -ForegroundColor Yellow
try {
    $gpuInfo = docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ NVIDIA GPU detected" -ForegroundColor Green
        Write-Host $gpuInfo
    } else {
        Write-Host "⚠ Warning: GPU access may not be available" -ForegroundColor Yellow
    }
} catch {
    Write-Host "⚠ Warning: Could not verify GPU access" -ForegroundColor Yellow
    Write-Host "   Container will still run but may use CPU only" -ForegroundColor Yellow
}

# Stop and remove existing container if requested
if ($RemoveExisting) {
    Write-Host "Removing existing container..." -ForegroundColor Yellow
    docker stop $ContainerName 2>$null
    docker rm $ContainerName 2>$null
    Write-Host "✓ Container removed" -ForegroundColor Green
}

# Check if container already exists
$existing = docker ps -a --filter "name=$ContainerName" --format "{{.Names}}"
if ($existing -and -not $RemoveExisting) {
    Write-Host "Container '$ContainerName' already exists." -ForegroundColor Yellow
    Write-Host "Options:" -ForegroundColor Yellow
    Write-Host "  1. Start existing: docker start -i $ContainerName" -ForegroundColor Cyan
    Write-Host "  2. Remove and recreate: .\docker-run-windows.ps1 -RemoveExisting" -ForegroundColor Cyan
    exit 0
}

# Convert Windows path to WSL format (for better performance)
$wslPath = $ProjectPath -replace 'C:\\', '/mnt/c/' -replace '\\', '/'
Write-Host "Project path: $ProjectPath" -ForegroundColor Cyan
Write-Host "WSL path: $wslPath" -ForegroundColor Cyan
Write-Host ""

# Build Docker image if it doesn't exist
Write-Host "Building Docker image..." -ForegroundColor Yellow
docker build -t areal-grpo:local -f Dockerfile .
if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Failed to build Docker image" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Docker image ready" -ForegroundColor Green
Write-Host ""

# Run Docker container
Write-Host "Starting Docker container..." -ForegroundColor Yellow
Write-Host "Container name: $ContainerName" -ForegroundColor Cyan
Write-Host "Shared memory: $SharedMemory" -ForegroundColor Cyan
Write-Host ""

$dockerRun = @"
docker run -it --name $ContainerName `
    --gpus all `
    --shm-size=$SharedMemory `
    --network host `
    -v ${ProjectPath}:/workspace/AReaL:rw `
    -v ${PWD}/wandb:/workspace/AReaL/examples/local_gsm8k/wandb:rw `
    -v ${PWD}/outputs:/workspace/AReaL/examples/local_gsm8k/outputs:rw `
    -w /workspace/AReaL `
    -e PYTHONPATH=/workspace/AReaL `
    -e WANDB_MODE=disabled `
    areal-grpo:local `
    /bin/bash
"@

Write-Host "Running: docker run ..." -ForegroundColor Gray
Write-Host ""

Invoke-Expression $dockerRun

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "✗ Failed to start container" -ForegroundColor Red
    Write-Host "Troubleshooting:" -ForegroundColor Yellow
    Write-Host "  1. Ensure Docker Desktop is running" -ForegroundColor Yellow
    Write-Host "  2. Enable WSL2 backend in Docker Desktop settings" -ForegroundColor Yellow
    Write-Host "  3. Check NVIDIA Container Toolkit is installed" -ForegroundColor Yellow
    exit 1
}

