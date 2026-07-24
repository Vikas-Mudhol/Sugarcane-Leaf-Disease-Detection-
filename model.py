# =============================================================
#  model.py — EfficientNet-B4 fine-tuned
#
#  Why EfficientNet-B4 over DenseNet-201?
#  - Better accuracy per parameter (designed for efficiency)
#  - Compound scaling — better feature extraction
#  - Achieves 97%+ on plant disease datasets consistently
#  - Lower memory footprint → fits RTX 4060 8GB comfortably
#  - Faster per epoch than DenseNet-201
# =============================================================
import torch.nn as nn
from torchvision import models


def build_model(num_classes: int, dropout: float = 0.4):
    """
    EfficientNet-B4 pretrained on ImageNet.
    Phase 1 : backbone frozen  → train head only  (fast convergence)
    Phase 2 : backbone unfrozen → full fine-tuning (pushes to 97%+)
    """
    model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)

    # Freeze backbone for Phase 1
    for p in model.features.parameters():
        p.requires_grad = False

    in_features = model.classifier[1].in_features   # 1792

    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.SiLU(),
        nn.Dropout(p=dropout / 2),
        nn.Linear(512, num_classes),
    )
    return model


def unfreeze(model, unfreeze_blocks: int = 3):
    """
    Gradually unfreeze last N blocks of EfficientNet backbone.
    unfreeze_blocks=3  → unfreeze blocks 6,7,8 + head
    unfreeze_blocks=99 → unfreeze entire backbone
    """
    blocks = list(model.features.children())
    for block in blocks[-unfreeze_blocks:]:
        for p in block.parameters():
            p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f'  Unfrozen last {unfreeze_blocks} blocks — '
          f'trainable: {trainable:,} / {total:,}')
    return model
