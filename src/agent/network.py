"""
Architecture réseau pour l'agent RL Teeworlds.

L'observation est un Dict avec "image" (84x84 grayscale) et "position" (x, y).
On utilise un extracteur custom pour Stable-Baselines3 qui :
1. Passe l'image dans un CNN
2. Passe la position dans un petit MLP
3. Concatène les deux features
"""

import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class TeeWorldsExtractor(BaseFeaturesExtractor):
    """
    Extracteur de features custom pour observation Dict.

    Architecture:
        image (84x84x1) → CNN → 256 features
        position (2,)   → MLP → 32 features
        concat          → 288 features
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 288):
        super().__init__(observation_space, features_dim)

        image_shape = observation_space["image"].shape  # (84, 84, 1)
        n_channels = image_shape[2]

        # CNN pour l'image (inspiré de la NatureCNN d'Atari)
        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculer la taille de sortie du CNN
        with torch.no_grad():
            sample = torch.zeros(1, n_channels, image_shape[0], image_shape[1])
            cnn_out_size = self.cnn(sample).shape[1]

        self.cnn_fc = nn.Sequential(
            nn.Linear(cnn_out_size, 256),
            nn.ReLU(),
        )

        # MLP pour la position
        pos_size = observation_space["position"].shape[0]  # 2
        self.pos_mlp = nn.Sequential(
            nn.Linear(pos_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        # Vérification
        assert features_dim == 256 + 32, \
            f"features_dim ({features_dim}) != 256 + 32 = 288"

    def forward(self, observations: dict) -> torch.Tensor:
        # Image: (batch, H, W, C) → (batch, C, H, W) pour PyTorch
        image = observations["image"].float() / 255.0
        if image.dim() == 4:
            image = image.permute(0, 3, 1, 2)

        cnn_features = self.cnn_fc(self.cnn(image))

        # Position
        pos = observations["position"].float()
        pos_features = self.pos_mlp(pos)

        # Concaténation
        return torch.cat([cnn_features, pos_features], dim=1)


def get_policy_kwargs() -> dict:
    """Retourne les kwargs pour la policy SB3."""
    return {
        "features_extractor_class": TeeWorldsExtractor,
        "features_extractor_kwargs": {"features_dim": 288},
        "net_arch": {
            "pi": [128, 64],   # réseau policy (acteur)
            "vf": [128, 64],   # réseau value (critique)
        },
    }
