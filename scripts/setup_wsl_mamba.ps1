#Requires -Version 5.1
[CmdletBinding()]
param(
    [string]$Distro = "Ubuntu",
    [string]$LinuxUser = "aicdlab",
    [string]$EnvName = "aicdlab-mamba",
    [string]$ProjectPath = "",
    [string]$WslProjectDir = "",
    [ValidateSet("cu121", "cu124")]
    [string]$TorchCuda = "cu121",
    [string]$DatasetSlug = "sandeshbhat/animal-image-classificationdogs-cats",
    [switch]$NoDataSync,
    [switch]$SkipDatasetDownload,
    [switch]$SkipDistroInstall
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Get-InstalledWslDistros {
    $raw = & wsl.exe --list --quiet 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $raw) {
        return @()
    }
    return @(
        $raw |
            ForEach-Object { ($_ -replace "`0", "").Trim() } |
            Where-Object { $_ }
    )
}

function ConvertTo-WslMountPath {
    param([string]$WindowsPath)

    $resolvedPath = (Resolve-Path -LiteralPath $WindowsPath).Path
    if ($resolvedPath -match "^([A-Za-z]):\\(.*)$") {
        $drive = $matches[1].ToLowerInvariant()
        $tail = $matches[2] -replace "\\", "/"
        return "/mnt/$drive/$tail"
    }

    throw "Cannot convert path to a WSL mount path: $resolvedPath"
}

if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
    throw "wsl.exe was not found. Use Windows 10 2004+ or Windows 11, then run this script from PowerShell."
}

if ([string]::IsNullOrWhiteSpace($ProjectPath)) {
    $ProjectPath = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
} else {
    $ProjectPath = (Resolve-Path -LiteralPath $ProjectPath).Path
}

if ([string]::IsNullOrWhiteSpace($WslProjectDir)) {
    $WslProjectDir = "/home/$LinuxUser/AICDLab1"
}

Write-Step "Checking WSL distro $Distro"
$distros = Get-InstalledWslDistros
if ($distros -notcontains $Distro) {
    if ($SkipDistroInstall) {
        throw "$Distro is not installed and -SkipDistroInstall was provided."
    }

    Write-Step "Installing $Distro with WSL"
    & wsl.exe --set-default-version 2 | Out-Null
    & wsl.exe --install -d $Distro --no-launch
    if ($LASTEXITCODE -ne 0) {
        throw "WSL distro installation failed. If Windows asks for a reboot, reboot and run this script again."
    }
}

Write-Step "Starting $Distro as root"
& wsl.exe -d $Distro -u root -- bash -lc "echo wsl_ready"
if ($LASTEXITCODE -ne 0) {
    throw "Could not start $Distro. Reboot if WSL was just installed, then run this script again."
}

$wslSource = ConvertTo-WslMountPath $ProjectPath

$includeData = if ($NoDataSync) { "0" } else { "1" }
$skipDatasetDownload = if ($SkipDatasetDownload) { "1" } else { "0" }

$bootstrap = @'
set -euo pipefail

SOURCE="$1"
TARGET="$2"
LINUX_USER="$3"
ENV_NAME="$4"
TORCH_CUDA="$5"
INCLUDE_DATA="$6"
DATASET_SLUG="$7"
SKIP_DATASET_DOWNLOAD="$8"

if [ "$(id -u)" -ne 0 ]; then
    echo "This bootstrap must run as root." >&2
    exit 1
fi

export DEBIAN_FRONTEND=noninteractive

echo "Installing Ubuntu packages..."
apt-get update
apt-get install -y \
    build-essential \
    ca-certificates \
    curl \
    g++-12 \
    gcc-12 \
    git \
    ninja-build \
    pkg-config \
    python3 \
    python3-pip \
    python3-venv \
    rsync \
    sudo

if ! id "$LINUX_USER" >/dev/null 2>&1; then
    echo "Creating Linux user: $LINUX_USER"
    useradd -m -s /bin/bash "$LINUX_USER"
fi

usermod -aG sudo "$LINUX_USER" || true
echo "$LINUX_USER ALL=(ALL) NOPASSWD:ALL" > "/etc/sudoers.d/$LINUX_USER"
chmod 0440 "/etc/sudoers.d/$LINUX_USER"
printf '[user]\ndefault=%s\n' "$LINUX_USER" > /etc/wsl.conf

echo "Syncing project into WSL: $TARGET"
mkdir -p "$TARGET"
RSYNC_EXCLUDES=(
    --exclude .git
    --exclude .venv
    --exclude __pycache__
    --exclude artifacts
    --exclude outputs
    --exclude outputs_*
    --exclude reports
)
if [ "$INCLUDE_DATA" != "1" ]; then
    RSYNC_EXCLUDES+=(--exclude Data)
fi
rsync -a "${RSYNC_EXCLUDES[@]}" "$SOURCE/" "$TARGET/"
chown -R "$LINUX_USER:$LINUX_USER" "$TARGET"

USER_SETUP="/tmp/aicdlab_mamba_user_setup.sh"
cat > "$USER_SETUP" <<'USER_SETUP_EOF'
#!/usr/bin/env bash
set -euo pipefail

TARGET="$1"
ENV_NAME="$2"
TORCH_CUDA="$3"
DATASET_SLUG="$4"
SKIP_DATASET_DOWNLOAD="$5"

MINIFORGE="$HOME/miniforge3"
INSTALLER="/tmp/miniforge3-linux-x86_64.sh"

if [ ! -x "$MINIFORGE/bin/conda" ]; then
    echo "Installing Miniforge into $MINIFORGE"
    curl -L --retry 3 -o "$INSTALLER" \
        "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
    bash "$INSTALLER" -b -p "$MINIFORGE"
fi

source "$MINIFORGE/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Creating conda env: $ENV_NAME"
    conda create -y -n "$ENV_NAME" python=3.11 pip
fi

conda activate "$ENV_NAME"
PYTORCH_VERSION="2.5.1"
TORCHVISION_VERSION="0.20.1"
python -m pip install --upgrade pip setuptools wheel packaging ninja
python -m pip install \
    "torch==${PYTORCH_VERSION}+${TORCH_CUDA}" \
    "torchvision==${TORCHVISION_VERSION}+${TORCH_CUDA}" \
    --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"
python -m pip install -r "$TARGET/requirements.txt"
ENV_PREFIX="$MINIFORGE/envs/$ENV_NAME"
export CUDA_HOME="$ENV_PREFIX"
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
python -m pip install -r "$TARGET/requirements-mamba.txt"
python -m pip check

cd "$TARGET"
if [ "$SKIP_DATASET_DOWNLOAD" != "1" ]; then
    export AICDLAB_TARGET="$TARGET"
    export AICDLAB_DATASET_SLUG="$DATASET_SLUG"
    python - <<'PY'
import os
import shutil
from pathlib import Path

import kagglehub

target = Path(os.environ["AICDLAB_TARGET"])
data_dir = target / "Data"
dataset_slug = os.environ["AICDLAB_DATASET_SLUG"]
image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

has_images = data_dir.exists() and any(path.suffix.lower() in image_extensions for path in data_dir.rglob("*"))
if has_images:
    print(f"dataset_ready={data_dir}")
else:
    print(f"downloading_dataset={dataset_slug}")
    source = Path(kagglehub.dataset_download(dataset_slug))
    if data_dir.exists():
        shutil.rmtree(data_dir)
    shutil.copytree(source, data_dir)
    print(f"dataset_ready={data_dir}")
PY
fi

if [ -d Data ]; then
    python scripts/prepare_splits.py --data-dir Data --classes cats dogs --folds 5
fi

python - <<'PY'
import torch
from mamba_ssm import Mamba
from src.models import build_model, count_parameters

print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit(
        "CUDA is not visible in WSL. Install or update the NVIDIA Windows driver "
        "with WSL CUDA support, then run this setup script again."
    )

model = build_model("mamba", num_classes=2, image_size=224).cuda().eval()
x = torch.randn(1, 3, 224, 224, device="cuda")
with torch.inference_mode():
    y = model(x)
print("mamba_parameters", count_parameters(model))
print("mamba_forward_shape", tuple(y.shape))
PY

cat > "$HOME/run_aicdlab_mamba.sh" <<RUN_SCRIPT_EOF
#!/usr/bin/env bash
set -euo pipefail
source "$MINIFORGE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
cd "$TARGET"
python scripts/prepare_splits.py --data-dir Data --classes cats dogs --folds 5
BATCH_SIZE="\${BATCH_SIZE:-16}"
EPOCHS="\${EPOCHS:-30}"
GRAD_ACCUM_STEPS="\${GRAD_ACCUM_STEPS:-2}"
OUTPUT_DIR="\${OUTPUT_DIR:-animal_binary_mamba}"
WORKERS="\${WORKERS:-4}"
FREEZE_BACKBONE_EPOCHS="\${FREEZE_BACKBONE_EPOCHS:-1}"
python train.py \\
  --train-csv Data/folds/fold_0_train.csv \\
  --val-csv Data/folds/fold_0_val.csv \\
  --output-dir "\$OUTPUT_DIR" \\
  --batch-size "\$BATCH_SIZE" \\
  --grad-accum-steps "\$GRAD_ACCUM_STEPS" \\
  --epochs "\$EPOCHS" \\
  --workers "\$WORKERS" \\
  --freeze-backbone-epochs "\$FREEZE_BACKBONE_EPOCHS" \\
  --amp \\
  --use-randaugment
RUN_SCRIPT_EOF
chmod +x "$HOME/run_aicdlab_mamba.sh"

echo "Training launcher written to $HOME/run_aicdlab_mamba.sh"
USER_SETUP_EOF

chmod +x "$USER_SETUP"
sudo -H -u "$LINUX_USER" bash "$USER_SETUP" "$TARGET" "$ENV_NAME" "$TORCH_CUDA" "$DATASET_SLUG" "$SKIP_DATASET_DOWNLOAD"

echo "WSL setup complete."
'@

Write-Step "Bootstrapping official Mamba environment inside WSL"
$bootstrap | & wsl.exe -d $Distro -u root -- bash -s -- $wslSource $WslProjectDir $LinuxUser $EnvName $TorchCuda $includeData $DatasetSlug $skipDatasetDownload
if ($LASTEXITCODE -ne 0) {
    throw "WSL Mamba bootstrap failed."
}

Write-Step "Done"
Write-Host "Project in WSL: $WslProjectDir"
Write-Host "Run training with:"
Write-Host "  wsl -d $Distro -u $LinuxUser -- /home/$LinuxUser/run_aicdlab_mamba.sh" -ForegroundColor Green
Write-Host ""
Write-Host "If this was the first time installing the distro, run this once to apply the default user for interactive WSL shells:"
Write-Host "  wsl --terminate $Distro"
