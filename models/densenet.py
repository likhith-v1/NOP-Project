"""
DenseNet-121 wrapper for binary chest X-ray classification.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


class ChestXRayDenseNet(nn.Module):
    """
    DenseNet-121 fine-tuned for binary chest X-ray classification.

    Args:
        pretrained      : Load ImageNet weights (recommended)
        dropout_rate    : Dropout before final linear layer
        num_classes     : 2 for NORMAL / PNEUMONIA
        freeze_features : If True, freeze all layers except classifier head
    """

    def __init__(
        self,
        pretrained: bool = True,
        dropout_rate: float = 0.5,
        num_classes: int = 2,
        freeze_features: bool = False,
    ):
        super().__init__()

        # Load backbone
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.densenet121(weights=weights)

        # Feature extractor (all layers except classifier)
        self.features = backbone.features    # output: [B, 1024, 7, 7]
        self.relu     = nn.ReLU(inplace=True)
        self.avgpool  = nn.AdaptiveAvgPool2d((1, 1))

        # Custom classifier head
        in_features = backbone.classifier.in_features   # 1024
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes),
        )

        # Weight init for classifier
        nn.init.xavier_uniform_(self.classifier[1].weight)
        nn.init.zeros_(self.classifier[1].bias)

        # Optional feature freeze
        if freeze_features:
            self.freeze_backbone()

    # Forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input  : x of shape [B, 3, 224, 224]
        Output : logits of shape [B, 2]
        """
        x = self.features(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)          # [B, 1024]
        x = self.classifier(x)           # [B, 2]
        return x

    # Freeze / unfreeze helpers

    def freeze_backbone(self) -> None:
        """Freeze all feature layers — only classifier trains."""
        for param in self.features.parameters():
            param.requires_grad = False
        print("[Model] Backbone frozen — training classifier head only.")

    def unfreeze_backbone(self) -> None:
        """Unfreeze all layers for full fine-tuning."""
        for param in self.features.parameters():
            param.requires_grad = True
        print("[Model] Backbone unfrozen — full fine-tuning enabled.")

    def get_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def print_param_summary(self) -> None:
        total     = self.get_total_params()
        trainable = self.get_trainable_params()
        frozen    = total - trainable
        print(f"\n  DenseNet-121 Parameter Summary")
        print(f"  {'─'*35}")
        print(f"  Total params      : {total:>10,}")
        print(f"  Trainable params  : {trainable:>10,}")
        print(f"  Frozen params     : {frozen:>10,}")
        print(f"  {'─'*35}\n")


# Factory function

def build_model(cfg: dict, freeze_features: bool = False) -> ChestXRayDenseNet:
    """
    Build model from config dict.

    During the first `freeze_features_epochs` epochs, backbone is frozen.
    train.py calls model.unfreeze_backbone() after warmup.
    """
    model_cfg = cfg["model"]
    model = ChestXRayDenseNet(
        pretrained=model_cfg.get("pretrained", True),
        dropout_rate=model_cfg.get("dropout_rate", 0.5),
        num_classes=cfg["data"]["num_classes"],
        freeze_features=freeze_features,
    )
    return model

# Smoke test

if __name__ == "__main__":
    model = ChestXRayDenseNet(pretrained=False)   # pretrained=False for quick test
    model.print_param_summary()

    dummy = torch.randn(4, 3, 224, 224)
    out   = model(dummy)
    print(f"  Output shape : {out.shape}")   # should be [4, 2]
    assert out.shape == (4, 2), "Forward pass shape mismatch!"
    print("  ✓ Forward pass OK")
