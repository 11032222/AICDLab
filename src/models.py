from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    EfficientNet_B7_Weights,
    ResNet18_Weights,
    efficientnet_b0,
    efficientnet_b7,
    resnet18,
)
from torchvision.transforms import InterpolationMode


@dataclass(frozen=True)
class ModelDataConfig:
    image_size: int
    eval_resize_size: int
    interpolation: InterpolationMode


def load_official_mamba_class() -> type[nn.Module]:
    try:
        from mamba_ssm import Mamba
    except ImportError as exc:
        raise ImportError(
            "Official Mamba is required for --model mamba. Install it in a supported "
            "Linux + NVIDIA CUDA environment with: "
            "pip install mamba-ssm[causal-conv1d] --no-build-isolation"
        ) from exc
    return Mamba


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class PatchEmbed(nn.Module):
    def __init__(self, image_size: int, patch_size: int, in_chans: int, embed_dim: int) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class OfficialMambaBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        mamba_cls = load_official_mamba_class()
        self.norm = nn.LayerNorm(dim)
        self.mixer = mamba_cls(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop_path(self.mixer(self.norm(x)))


class OfficialMambaVisionClassifier(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 2,
        embed_dim: int = 192,
        depth: int = 8,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        drop_rate: float = 0.1,
        drop_path_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(image_size, patch_size, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        drop_path_values = torch.linspace(0, drop_path_rate, steps=depth).tolist()
        self.blocks = nn.ModuleList(
            [
                OfficialMambaBlock(
                    dim=embed_dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    drop_path=drop_path_values[index],
                )
                for index in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self._init_own_weights()

    def _init_own_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.patch_embed.proj.weight, std=0.02)
        if self.patch_embed.proj.bias is not None:
            nn.init.zeros_(self.patch_embed.proj.bias)
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def set_backbone_trainable(self, trainable: bool) -> None:
        for name, parameter in self.named_parameters():
            parameter.requires_grad = trainable or name.startswith("head.")

    def parameter_groups(self, lr: float, weight_decay: float, backbone_lr_scale: float) -> list[dict[str, object]]:
        head_params = []
        backbone_params = []
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith("head."):
                head_params.append(parameter)
            else:
                backbone_params.append(parameter)

        groups: list[dict[str, object]] = []
        if backbone_params:
            groups.append({"params": backbone_params, "lr": lr * backbone_lr_scale, "weight_decay": weight_decay})
        if head_params:
            groups.append({"params": head_params, "lr": lr, "weight_decay": weight_decay})
        return groups

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([x, cls_tokens], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x[:, -1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


class EfficientNetMambaVisionClassifier(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        num_classes: int = 2,
        embed_dim: int = 256,
        depth: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        drop_rate: float = 0.1,
        drop_path_rate: float = 0.05,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        if image_size != 224:
            raise ValueError("EfficientNet-Mamba uses image_size=224.")

        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = efficientnet_b0(weights=weights)
        self.feature_extractor = backbone.features
        self.feature_projection = nn.Conv2d(1280, embed_dim, kernel_size=1)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 50, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        drop_path_values = torch.linspace(0, drop_path_rate, steps=depth).tolist()
        self.blocks = nn.ModuleList(
            [
                OfficialMambaBlock(
                    dim=embed_dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    drop_path=drop_path_values[index],
                )
                for index in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim * 2, num_classes)
        self._init_own_weights()

    def _init_own_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.feature_projection.weight, std=0.02)
        if self.feature_projection.bias is not None:
            nn.init.zeros_(self.feature_projection.bias)
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def set_backbone_trainable(self, trainable: bool) -> None:
        for parameter in self.feature_extractor.parameters():
            parameter.requires_grad = trainable
        for name, parameter in self.named_parameters():
            if not name.startswith("feature_extractor."):
                parameter.requires_grad = True

    def parameter_groups(self, lr: float, weight_decay: float, backbone_lr_scale: float) -> list[dict[str, object]]:
        backbone_params = []
        classifier_params = []
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith("feature_extractor."):
                backbone_params.append(parameter)
            else:
                classifier_params.append(parameter)

        groups: list[dict[str, object]] = []
        if backbone_params:
            groups.append({"params": backbone_params, "lr": lr * backbone_lr_scale, "weight_decay": weight_decay})
        if classifier_params:
            groups.append({"params": classifier_params, "lr": lr, "weight_decay": weight_decay})
        return groups

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        x = self.feature_projection(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([x, cls_tokens], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        cls_feature = x[:, -1]
        mean_feature = x[:, :-1].mean(dim=1)
        return torch.cat([cls_feature, mean_feature], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


class TorchvisionClassifier(nn.Module):
    def __init__(self, model: nn.Module, head_kind: str, num_classes: int) -> None:
        super().__init__()
        self.model = model
        self.head_kind = head_kind
        self._replace_head(num_classes)

    def _replace_head(self, num_classes: int) -> None:
        if self.head_kind == "efficientnet":
            in_features = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Linear(in_features, num_classes)
            nn.init.trunc_normal_(self.model.classifier[-1].weight, std=0.02)
            nn.init.zeros_(self.model.classifier[-1].bias)
            return
        if self.head_kind == "resnet":
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)
            nn.init.trunc_normal_(self.model.fc.weight, std=0.02)
            nn.init.zeros_(self.model.fc.bias)
            return
        raise ValueError(f"Unsupported head kind: {self.head_kind}")

    def set_backbone_trainable(self, trainable: bool) -> None:
        head_prefix = "classifier" if self.head_kind == "efficientnet" else "fc"
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad = trainable or name.startswith(head_prefix)

    def parameter_groups(self, lr: float, weight_decay: float, backbone_lr_scale: float) -> list[dict[str, object]]:
        head_prefix = "classifier" if self.head_kind == "efficientnet" else "fc"
        head_params = []
        backbone_params = []
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith(head_prefix):
                head_params.append(parameter)
            else:
                backbone_params.append(parameter)

        groups: list[dict[str, object]] = []
        if backbone_params:
            groups.append({"params": backbone_params, "lr": lr * backbone_lr_scale, "weight_decay": weight_decay})
        if head_params:
            groups.append({"params": head_params, "lr": lr, "weight_decay": weight_decay})
        return groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def get_model_data_config(model_name: str, image_size: int) -> ModelDataConfig:
    normalized = model_name.lower()
    if normalized == "mamba":
        return ModelDataConfig(image_size=224, eval_resize_size=256, interpolation=InterpolationMode.BICUBIC)
    if normalized == "efficientnet_b7":
        return ModelDataConfig(image_size=600, eval_resize_size=600, interpolation=InterpolationMode.BICUBIC)
    if normalized == "efficientnet_b0":
        return ModelDataConfig(image_size=224, eval_resize_size=256, interpolation=InterpolationMode.BICUBIC)
    if normalized == "resnet18":
        return ModelDataConfig(
            image_size=image_size,
            eval_resize_size=int(round(image_size * 256 / 224)),
            interpolation=InterpolationMode.BILINEAR,
        )
    return ModelDataConfig(
        image_size=image_size,
        eval_resize_size=int(round(image_size * 256 / 224)),
        interpolation=InterpolationMode.BILINEAR,
    )


def build_model(
    model_name: str,
    num_classes: int,
    image_size: int,
    pretrained: bool = True,
    mamba_architecture: str = "hybrid",
) -> nn.Module:
    normalized = model_name.lower()
    if normalized == "mamba":
        if image_size != 224:
            raise ValueError("mamba uses image_size=224. Omit --image-size or pass 224.")
        if mamba_architecture == "hybrid":
            return EfficientNetMambaVisionClassifier(
                num_classes=num_classes,
                image_size=image_size,
                pretrained=pretrained,
            )
        if mamba_architecture == "patch":
            return OfficialMambaVisionClassifier(num_classes=num_classes, image_size=image_size)
        raise ValueError(f"Unsupported mamba architecture: {mamba_architecture}")
    if normalized == "efficientnet_b7":
        if image_size != 600:
            raise ValueError("efficientnet_b7 uses image_size=600. Omit --image-size or pass 600.")
        weights = EfficientNet_B7_Weights.DEFAULT if pretrained else None
        return TorchvisionClassifier(efficientnet_b7(weights=weights), head_kind="efficientnet", num_classes=num_classes)
    if normalized == "efficientnet_b0":
        if image_size != 224:
            raise ValueError("efficientnet_b0 uses image_size=224. Omit --image-size or pass 224.")
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        return TorchvisionClassifier(efficientnet_b0(weights=weights), head_kind="efficientnet", num_classes=num_classes)
    if normalized == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        return TorchvisionClassifier(resnet18(weights=weights), head_kind="resnet", num_classes=num_classes)
    raise ValueError(f"Unsupported model name: {model_name}")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
