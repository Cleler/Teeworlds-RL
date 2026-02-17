import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class Ar_1(nn.Module):
    def __init__(self, img_size, input_size, hidden_size, output_size, backbone_pretrained=False):
        super(Ar_1, self).__init__()
        
        self.backbone = resnet50(input_size=img_size, output_size=hidden_size, pretrained=backbone_pretrained) 

        self.network = nn.Sequential(
            nn.Linear(hidden_size + input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )
    
    def forward(self, img, inp):
        h = self.backbone(img)
        h = h.flatten(1)    # (batch, 2048)
        combined = torch.cat([h, inp], dim=1)
        return self.network(combined)

class DQNLoss(nn.Module):
    def __init__(self, gamma=0.99, loss_fn="mse"):
        super(DQNLoss, self).__init__()

        self.gamma = gamma

        # Swappable loss function — MSE is standard, Huber is more robust to outliers
        if loss_fn == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss_fn == "huber":
            self.loss_fn = nn.HuberLoss()
        else:
            raise ValueError(f"loss_fn must be 'mse' or 'huber', got '{loss_fn}'")

    def forward(self, output, actions, rewards, next_output, dones):
        """
        Args:
            output      (batch, n_actions)  -- Q-values for current state
            actions     (batch,)            -- actions actually taken
            rewards     (batch,)            -- rewards received
            next_output (batch, n_actions)  -- Q-values for next state (from target network)
            dones       (batch,)            -- 1.0 if terminal state, 0.0 otherwise
        """
        # Q-value of the action actually taken
        q_taken = output.gather(1, actions.unsqueeze(1)).squeeze(1)   # (batch,)

        # Build target — no gradients needed here
        with torch.no_grad():
            best_next_q = next_output.max(dim=1).values               # (batch,)
            target = rewards + self.gamma * best_next_q * (1 - dones) # (batch,)

        return self.loss_fn(q_taken, target)


class Ar_2(nn.Module):
    def __init__(self, input_size, backbone_pretrained=False):
        super(Ar_1, self).__init__()

        backbone = resnet50(weights=ResNet50_Weights.DEFAULT if backbone_pretrained else None)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # (batch, 2048)

        self.network = nn.Sequential(
            nn.Linear(2048 + input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # Discrete heads
        self.head_move  = nn.Linear(128, 3)   # left, none, right
        self.head_jump  = nn.Linear(128, 2)   # no jump, jump
        self.head_hook  = nn.Linear(128, 2)   # no hook, hook
        self.head_fire  = nn.Linear(128, 2)   # no fire, fire

        # Continuous head — aim as (sin, cos) 
        self.head_aim   = nn.Linear(128, 2)

    def forward(self, image, extra_vars):
        img_features = self.backbone(image).flatten(1)               # (batch, 2048)
        combined     = torch.cat([img_features, extra_vars], dim=1)  # (batch, 2048 + input_size)
        x            = self.network(combined)                         # (batch, 128)

        return {
            "move" : self.head_move(x),   # (batch, 3)
            "jump" : self.head_jump(x),   # (batch, 2)
            "hook" : self.head_hook(x),   # (batch, 2)
            "fire" : self.head_fire(x),   # (batch, 2)
            "aim"  : self.head_aim(x),    # (batch, 2)  ← (sin, cos)
        }


class Ar_2Loss(nn.Module):
    DISCRETE_HEADS = ["move", "jump", "hook", "fire"]

    def __init__(self, gamma=0.99, aim_weight=1.0):
        super().__init__()
        self.gamma      = gamma
        self.aim_weight = aim_weight   # how much to weight aim loss vs DQN loss
        self.dqn_loss   = nn.HuberLoss()
        self.aim_loss   = nn.MSELoss()

    def forward(self, outputs, actions, rewards, next_outputs, dones, aim_targets):
        """
        outputs      -- dict of current Q-values  (from main model)
        actions      -- dict of actions taken      e.g. {"move": tensor, "jump": tensor, ...}
        rewards      -- (batch,)
        next_outputs -- dict of next Q-values      (from target network)
        dones        -- (batch,)
        aim_targets  -- (batch, 2)  ground truth (sin, cos) aim angle
        """
        total_loss = 0

        # DQN loss for each discrete head
        for head in self.DISCRETE_HEADS:
            q_taken = outputs[head].gather(1, actions[head].unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                best_next_q = next_outputs[head].max(dim=1).values
                target      = rewards + self.gamma * best_next_q * (1 - dones)

            total_loss += self.dqn_loss(q_taken, target)

        # Regression loss for aim
        total_loss += self.aim_weight * self.aim_loss(outputs["aim"], aim_targets)

        return total_loss
