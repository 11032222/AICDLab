# Animal Binary Classification

统一训练入口，支持 **ViT** (Vision Transformer) 和 **Mamba SSM** 两种模型，可从5类动物中任选2类进行二分类训练。

## 快速开始

### 使用 ViT（Windows/Linux 均可，只需 PyTorch）

```bash
# 安装依赖
pip install -r requirements.txt

# 训练 Cat vs Dog
python train.py --model vit --class1 Cat --class2 Dog --epochs 30 --batch-size 32

# 训练 Elephant vs Panda
python train.py --model vit --class1 Elephant --class2 Panda --epochs 30 --batch-size 32
```

### 使用 Mamba（WSL + CUDA，需要 mamba-ssm 库）

```bash
# 在 WSL 中安装依赖
pip install -r requirements-mamba.txt

# 训练 Cat vs Dog
python train.py --model mamba --class1 Cat --class2 Dog --epochs 30 --batch-size 32 --amp
```

## 命令行参数

### 模型与数据

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model` | 模型类型: `vit`, `mamba`, `efficientnet_b0`, `efficientnet_b7`, `resnet18` | `vit` |
| `--class1` | 第1类动物名 (如 Cat) | 无 |
| `--class2` | 第2类动物名 (如 Dog) | 无 |
| `--train-dir` | 训练数据根目录 | `Training Data` |
| `--val-dir` | 验证数据根目录 | `Validation Data` |
| `--train-csv` | 训练集 CSV（优先级高于 --class1/--class2） | 自动生成 |
| `--val-csv` | 验证集 CSV | 自动生成 |
| `--image-size` | 输入图像尺寸 | 根据模型自动选择 |

### 训练参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--epochs` | 训练轮数 | 30 |
| `--batch-size` | 批次大小 | 16 |
| `--lr` | 学习率 | 3e-4 |
| `--weight-decay` | 权重衰减 | 0.05 |
| `--amp` | 启用混合精度训练 | False |
| `--grad-accum-steps` | 梯度累积步数 | 1 |
| `--warmup-epochs` | 学习率预热轮数 | 3 |
| `--label-smoothing` | 标签平滑 | 0.1 |
| `--use-randaugment` | 使用 RandAugment 数据增强 | False |
| `--workers` | 数据加载线程数 | 0 |
| `--device` | 设备 | cuda (if available) |
| `--seed` | 随机种子 | 42 |

### Mamba 特殊参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--mamba-architecture` | Mamba架构: `hybrid`(EfficientNet特征+Mamba), `patch`(纯Mamba) | `hybrid` |

### 其他

| 参数 | 说明 |
|------|------|
| `--output-dir` | 输出目录 |
| `--resume` | 从上次训练恢复 |
| `--full-metrics` | 输出详细指标（F1, Balanced Accuracy等） |
| `--target-acc` | 目标准确率（用于判断 target_met） |

## 数据集结构

```
Training Data/
├── Cat/   (1500 images)
├── Dog/   (1500 images)
├── Elephant/ (1500 images)
├── Panda/ (1500 images)
└── Cow/   (1500 images)

Validation Data/
├── Cat/   (500 images)
├── Dog/   (500 images)
├── Elephant/ (500 images)
├── Panda/ (500 images)
└── Cow/   (500 images)
```

## 输出

```
artifacts/training_outputs/<model>/
├── history.json       # 训练历史
├── metrics.jsonl      # 指标日志
├── progress.json      # 实时进度
├── best_val_acc.pt    # 最佳模型检查点
├── last.pt            # 最后模型检查点
└── best_balanced_acc.pt  # 最佳平衡准确率检查点 (--full-metrics)
```

## 示例命令

```bash
# ViT: Cat vs Dog, 30 epochs
python train.py --model vit --class1 Cat --class2 Dog --epochs 30

# ViT: 大象 vs 熊猫, 50 epochs, 使用 RandAugment
python train.py --model vit --class1 Elephant --class2 Panda --epochs 50 --use-randaugment

# Mamba: Cat vs Dog, 混合精度
python train.py --model mamba --class1 Cat --class2 Dog --amp --epochs 50

# 使用已有 CSV
python train.py --model vit --train-csv _csv/train.csv --val-csv _csv/val.csv
```

## 要求

- Python 3.10+
- PyTorch 2.5+, torchvision 0.20+
- (可选) mamba-ssm: WSL + CUDA 环境