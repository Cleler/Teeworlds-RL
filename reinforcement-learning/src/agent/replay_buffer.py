"""
Replay Buffer pour le DQN.

Stocke les transitions (state, action, reward, next_state, done)
et permet l'échantillonnage de mini-batches aléatoires.
"""

import numpy as np
import torch
from collections import deque
import random


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state: dict, action: dict, reward: float,
             next_state: dict, done: bool, aim_target: np.ndarray):
        """
        Stocke une transition.

        Args:
            state:      {"image": np.ndarray (H,W,C), "position": np.ndarray (2,)}
            action:     {"move": int, "jump": int, "hook": int, "fire": int, "aim": np.ndarray (2,)}
            reward:     float
            next_state: même format que state
            done:       bool
            aim_target: np.ndarray (2,) — (sin, cos) de l'angle de visée réel
        """
        self.buffer.append((state, action, reward, next_state, done, aim_target))

    def sample(self, batch_size: int, device: torch.device) -> dict:
        """
        Échantillonne un mini-batch et le convertit en tensors.

        Returns:
            dict avec toutes les clés nécessaires pour Ar_2Loss.
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, aim_targets = zip(*batch)

        # --- Images: (batch, C, H, W) pour PyTorch ---
        images = np.stack([s["image"] for s in states])
        images = torch.FloatTensor(images).permute(0, 3, 1, 2).to(device) / 255.0

        next_images = np.stack([s["image"] for s in next_states])
        next_images = torch.FloatTensor(next_images).permute(0, 3, 1, 2).to(device) / 255.0

        # --- Position: (batch, 2) ---
        positions = torch.FloatTensor(np.stack([s["position"] for s in states])).to(device)
        next_positions = torch.FloatTensor(np.stack([s["position"] for s in next_states])).to(device)

        # --- Actions discrètes ---
        action_tensors = {
            "move": torch.LongTensor([a["move"] for a in actions]).to(device),
            "jump": torch.LongTensor([a["jump"] for a in actions]).to(device),
            "hook": torch.LongTensor([a["hook"] for a in actions]).to(device),
            "fire": torch.LongTensor([a["fire"] for a in actions]).to(device),
            "weapon": torch.LongTensor([a["weapon"] for a in actions]).to(device),
        }

        # --- Rewards, dones ---
        rewards_t = torch.FloatTensor(rewards).to(device)
        dones_t = torch.FloatTensor(dones).to(device)

        # --- Aim targets ---
        aim_targets_t = torch.FloatTensor(np.stack(aim_targets)).to(device)

        return {
            "images": images,
            "positions": positions,
            "actions": action_tensors,
            "rewards": rewards_t,
            "next_images": next_images,
            "next_positions": next_positions,
            "dones": dones_t,
            "aim_targets": aim_targets_t,
        }

    def __len__(self):
        return len(self.buffer)