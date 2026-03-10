"""DenseNet-121 model for Retinal OCT classification."""

import torch
import torch.nn as nn
from torchvision import models


class RetinalOCTDenseNet(nn.Module):
    """DenseNet-121 fine-tuned for 4-class Retinal OCT classification."""

    def __init__(self, pretrained=True, dropout_rate=0.5, num_classes=4,
                 freeze_features=False):
        super().__init__()

        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.densenet121(weights=weights)

        self.features = backbone.features
        self.relu     = nn.ReLU(inplace=True)
        self.avgpool  = nn.AdaptiveAvgPool2d((1, 1))

        in_features = backbone.classifier.in_features
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes),
        )

        nn.init.xavier_uniform_(self.classifier[1].weight)
        nn.init.zeros_(self.classifier[1].bias)

        if freeze_features:
            self.freeze_backbone()

    def forward(self, x):
        """Input: [B, 3, 224, 224]. Output: logits [B, num_classes]."""
        x = self.features(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def freeze_backbone(self):
        for param in self.features.parameters():
            param.requires_grad = False
        print("[Model] Backbone frozen — training classifier head only.")

    def unfreeze_backbone(self):
        for param in self.features.parameters():
            param.requires_grad = True
        print("[Model] Backbone unfrozen — full fine-tuning enabled.")

    def get_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self):
        return sum(p.numel() for p in self.parameters())

    def print_param_summary(self):
        total     = self.get_total_params()
        trainable = self.get_trainable_params()
        frozen    = total - trainable
        print(f"\n  DenseNet-121 Parameter Summary")
        print(f"  {'─'*35}")
        print(f"  Total params      : {total:>10,}")
        print(f"  Trainable params  : {trainable:>10,}")
        print(f"  Frozen params     : {frozen:>10,}")
        print(f"  {'─'*35}\n")


def build_model(cfg, freeze_features=False):
    """Build model from config."""
    model_cfg = cfg["model"]
    model = RetinalOCTDenseNet(
        pretrained=model_cfg.get("pretrained", True),
        dropout_rate=model_cfg.get("dropout_rate", 0.5),
        num_classes=cfg["data"]["num_classes"],
        freeze_features=freeze_features,
    )
    return model

