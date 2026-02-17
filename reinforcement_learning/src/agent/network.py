"""
Architectures réseau pour l'agent RL Teeworlds.

Ar_1 : DQN simple — ResNet50 backbone + MLP, une seule sortie Q-values.
Ar_2 : Multi-head — ResNet50 backbone + têtes séparées pour chaque action
        (move, jump, hook, fire en discret + aim en continu).
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


# ======================================================================
# Architecture 1 — DQN classique (une seule tête)
# ======================================================================

class Ar_1(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, backbone_pretrained=False):
        """
        Args:
            input_size: Taille des variables supplémentaires (position, hp, etc.)
            hidden_size: Taille de la couche après le backbone (projection)
            output_size: Nombre d'actions discrètes
            backbone_pretrained: Utiliser les poids ImageNet
        """
        super(Ar_1, self).__init__()

        # ResNet50 sans la dernière couche FC → sortie (batch, 2048)
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT if backbone_pretrained else None)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        # Projection des features backbone
        self.backbone_fc = nn.Sequential(
            nn.Linear(2048, hidden_size),
            nn.ReLU(),
        )

        self.network = nn.Sequential(
            nn.Linear(hidden_size + input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
        )

    def forward(self, img, inp):
        h = self.backbone(img)
        h = h.flatten(1)                        # (batch, 2048)
        h = self.backbone_fc(h)                  # (batch, hidden_size)
        combined = torch.cat([h, inp], dim=1)    # (batch, hidden_size + input_size)
        return self.network(combined)


class DQNLoss(nn.Module):
    def __init__(self, gamma=0.99, loss_fn="mse"):
        super(DQNLoss, self).__init__()
        self.gamma = gamma

        if loss_fn == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss_fn == "huber":
            self.loss_fn = nn.HuberLoss()
        else:
            raise ValueError(f"loss_fn must be 'mse' or 'huber', got '{loss_fn}'")

    def forward(self, output, actions, rewards, next_output, dones):
        """
        Args:
            output      (batch, n_actions)  -- Q-values état courant
            actions     (batch,)            -- actions prises
            rewards     (batch,)            -- récompenses reçues
            next_output (batch, n_actions)  -- Q-values état suivant (target network)
            dones       (batch,)            -- 1.0 si terminal, 0.0 sinon
        """
        q_taken = output.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            best_next_q = next_output.max(dim=1).values
            target = rewards + self.gamma * best_next_q * (1 - dones)

        return self.loss_fn(q_taken, target)


# ======================================================================
# Architecture 2 — Multi-head (discret + continu)
# ======================================================================

class Ar_2(nn.Module):
    def __init__(self, input_size, backbone_pretrained=False):
        """
        Args:
            input_size: Taille des variables supplémentaires (position, hp, etc.)
            backbone_pretrained: Utiliser les poids ImageNet
        """
        super(Ar_2, self).__init__()  # FIX: était super(Ar_1, self)

        backbone = resnet50(weights=ResNet50_Weights.DEFAULT if backbone_pretrained else None)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # → (batch, 2048)
        
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.network = nn.Sequential(
            nn.Linear(2048 + input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        # Discrete heads
        self.head_move = nn.Linear(32, 3)   # left, none, right
        self.head_jump = nn.Linear(32, 2)   # no jump, jump
        self.head_hook = nn.Linear(32, 2)   # no hook, hook
        self.head_fire = nn.Linear(32, 2)   # no fire, fire
        self.head_weapon = nn.Linear(32, 3)   # 0: rien, 1: scroll haut, 2: scroll bas

        # Continuous head — aim as (sin, cos)
        self.head_aim = nn.Linear(32, 2)

    def forward(self, image, extra_vars):
        img_features = self.backbone(image).flatten(1)               # (batch, 2048)
        combined = torch.cat([img_features, extra_vars], dim=1)      # (batch, 2048 + input_size)
        x = self.network(combined)                                    # (batch, 128)

        return {
            "move": self.head_move(x),   # (batch, 3)
            "jump": self.head_jump(x),   # (batch, 2)
            "hook": self.head_hook(x),   # (batch, 2)
            "fire": self.head_fire(x),   # (batch, 2)
            "aim":  self.head_aim(x),    # (batch, 2) ← (sin, cos)
        }


class Ar_2Loss(nn.Module):
    DISCRETE_HEADS = ["move", "jump", "hook", "fire", "weapon"]

    def __init__(self, gamma=0.99, aim_weight=1.0):
        super().__init__()
        self.gamma = gamma
        self.aim_weight = aim_weight
        self.dqn_loss = nn.HuberLoss()
        self.aim_loss = nn.MSELoss()

    def forward(self, outputs, actions, rewards, next_outputs, dones, aim_targets):
        """
        Args:
            outputs      -- dict Q-values courantes (main model)
            actions      -- dict actions prises {"move": tensor, "jump": tensor, ...}
            rewards      -- (batch,)
            next_outputs -- dict Q-values suivantes (target network)
            dones        -- (batch,)
            aim_targets  -- (batch, 2) ground truth (sin, cos)
        """
        total_loss = 0

        for head in self.DISCRETE_HEADS:
            q_taken = outputs[head].gather(1, actions[head].unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                best_next_q = next_outputs[head].max(dim=1).values
                target = rewards + self.gamma * best_next_q * (1 - dones)

            total_loss += self.dqn_loss(q_taken, target)

        total_loss += self.aim_weight * self.aim_loss(outputs["aim"], aim_targets)

        return total_loss