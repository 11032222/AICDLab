# Animal Binary Classification

Official Mamba SSM binary classifier for the Kaggle Cats/Dogs image dataset.

The default `--model mamba` path uses ImageNet EfficientNet-B0 visual features followed by official `mamba_ssm.Mamba` sequence blocks. This is much more reliable for the requested binary task than training a pure patch Mamba from scratch.

## Quick Start

Use Windows PowerShell from this project directory. If you need to set up WSL first, run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass -Force
.\scripts\setup_wsl_mamba.ps1 -Distro Ubuntu -LinuxUser nobodiiiii
```

Start training:

```powershell
wsl -d Ubuntu -u nobodiiiii -- /home/nobodiiiii/run_aicdlab_mamba.sh
```

Run a 1-epoch smoke test:

```powershell
wsl -d Ubuntu -u nobodiiiii -- bash -lc 'EPOCHS=1 OUTPUT_DIR=mamba_smoke ~/run_aicdlab_mamba.sh'
```

## Goal And Metrics

The dataset folds are balanced, so the default training record is intentionally compact:

- primary metric: `val_acc`
- target: `target_acc=0.85`
- default checkpoint to report: `best_val_acc.pt`
- saved diagnostics: loss, ACC, confusion matrix, `target_met`

Use `--full-metrics` only when you need per-class precision/recall/F1, balanced accuracy, or threshold search for a report.

## Verified Result

The included WSL/CUDA verification run exceeded the requested target:

- run: `artifacts/training_outputs/mamba_acc85_probe/mamba/`
- checkpoint: `best_val_acc.pt`
- best epoch: `4`
- validation ACC: `0.9876`
- target met: `true`
- confusion matrix: `[[790, 14], [6, 798]]`

## Useful Overrides

Change training length or batch size without editing files:

```powershell
wsl -d Ubuntu -u nobodiiiii -- bash -lc 'EPOCHS=10 BATCH_SIZE=16 GRAD_ACCUM_STEPS=2 OUTPUT_DIR=mamba_10e ~/run_aicdlab_mamba.sh'
```

Train the old pure patch Mamba baseline:

```powershell
wsl -d Ubuntu -u nobodiiiii -- bash -lc 'cd ~/AICDLab1 && source ~/miniforge3/etc/profile.d/conda.sh && conda activate aicdlab-mamba && python train.py --train-csv Data/folds/fold_0_train.csv --val-csv Data/folds/fold_0_val.csv --model mamba --mamba-architecture patch --output-dir mamba_patch --epochs 30 --batch-size 16 --grad-accum-steps 2 --workers 4 --amp --use-randaugment --full-metrics'
```

Evaluate a checkpoint:

```powershell
wsl -d Ubuntu -u nobodiiiii -- bash -lc 'cd ~/AICDLab1 && source ~/miniforge3/etc/profile.d/conda.sh && conda activate aicdlab-mamba && python scripts/evaluate_checkpoint.py --checkpoint artifacts/training_outputs/animal_binary_mamba/mamba/best_val_acc.pt --val-csv Data/folds/fold_0_val.csv --batch-size 16 --workers 4'
```

## Outputs

Training outputs are written inside WSL:

```text
/home/nobodiiiii/AICDLab1/artifacts/training_outputs/<run_name>/mamba/
```

Key files:

- `history.json`
- `metrics.jsonl`
- `progress.json`
- `last.pt`
- `best_val_acc.pt`

When `--full-metrics` is enabled, the run can also write `best_balanced_acc.pt` and `best_threshold_balanced_acc.pt`.

## Model

Default Mamba classifier:

- image size: `224`
- feature backbone: ImageNet EfficientNet-B0
- sequence length: `49` feature tokens + final class token
- Mamba embedding dim: `256`
- Mamba depth: `4`
- `d_state`: `16`
- `d_conv`: `4`
- expand: `2`
