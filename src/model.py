from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def build_resnet18(num_classes: int, imagenet_pretrained: bool, dropout: float = 0.0) -> nn.Module:
    if imagenet_pretrained:
        weights = models.ResNet18_Weights.DEFAULT
    else:
        weights = None

    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features

    head: list[nn.Module] = []
    if dropout and dropout > 0:
        head.append(nn.Dropout(p=float(dropout)))
    head.append(nn.Linear(in_features, num_classes))
    model.fc = nn.Sequential(*head)
    return model


def build_resnet18_encoder(imagenet_pretrained: bool) -> nn.Module:
    """ResNet-18 trunk: outputs a 512-d embedding (before the original classifier)."""
    if imagenet_pretrained:
        weights = models.ResNet18_Weights.DEFAULT
    else:
        weights = None
    model = models.resnet18(weights=weights)
    model.fc = nn.Identity()
    return model


class SimCLR(nn.Module):
    """Standard SimCLR: encoder + 2-layer projection head."""

    def __init__(
        self,
        encoder: nn.Module,
        projection_dim: int = 128,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.projector = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        z = self.projector(h)
        z = F.normalize(z, dim=1)
        return h, z


def nt_xent_loss(z_i: torch.Tensor, z_j: torch.Tensor, temperature: float) -> torch.Tensor:
    """SimCLR NT-Xent: full (2B-1)-way softmax per anchor; mask self on the diagonal."""
    b = z_i.size(0)
    z = torch.cat([z_i, z_j], dim=0)
    logits = torch.mm(z, z.t()) / temperature
    logits.fill_diagonal_(float("-inf"))

    pos = torch.cat([torch.arange(b, 2 * b, device=z.device), torch.arange(0, b, device=z.device)], dim=0)
    return F.cross_entropy(logits, pos)


def forward_logits(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    return model(x)


def load_encoder_weights_into_classifier(
    classifier: nn.Module,
    encoder_state: dict[str, torch.Tensor],
    strict_encoder: bool = True,
) -> None:
    """Copy backbone weights from a saved encoder state_dict into a full ResNet classifier."""
    enc_keys = {k: v for k, v in encoder_state.items() if not k.startswith("fc")}
    missing, unexpected = classifier.load_state_dict(enc_keys, strict=False)
    # Expect fc keys missing in checkpoint; encoder should match.
    if strict_encoder:
        bad_missing = [k for k in missing if not k.startswith("fc")]
        if bad_missing:
            raise RuntimeError(f"Encoder load missing unexpected keys: {bad_missing[:10]}...")
        if unexpected:
            raise RuntimeError(f"Unexpected keys when loading encoder: {unexpected[:10]}...")
