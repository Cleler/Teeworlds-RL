# """
# Replay Buffer pour le DQN.

# Stocke les transitions (state, action, reward, next_state, done)
# et permet l'échantillonnage de mini-batches aléatoires.
# """

# import numpy as np
# import torch
# from collections import deque
# import random


# class ReplayBuffer:
#     def __init__(self, capacity: int = 100_000):
#         self.buffer = deque(maxlen=capacity)

#     def push(self, state: dict, action: dict, reward: float,
#              next_state: dict, done: bool, aim_target: np.ndarray):
#         """
#         Stocke une transition.

#         Args:
#             state:      {"image": np.ndarray (H,W,C), "position": np.ndarray (2,)}
#             action:     {"move": int, "jump": int, "hook": int, "fire": int, "aim": np.ndarray (2,)}
#             reward:     float
#             next_state: même format que state
#             done:       bool
#             aim_target: np.ndarray (2,) — (sin, cos) de l'angle de visée réel
#         """
#         self.buffer.append((state, action, reward, next_state, done, aim_target))

#     def sample(self, batch_size: int, device: torch.device) -> dict:
#         """
#         Échantillonne un mini-batch et le convertit en tensors.

#         Returns:
#             dict avec toutes les clés nécessaires pour Ar_2Loss.
#         """
#         batch = random.sample(self.buffer, batch_size)
#         states, actions, rewards, next_states, dones, aim_targets = zip(*batch)

#         # --- Images: (batch, C, H, W) pour PyTorch ---
#         images = np.stack([s["image"] for s in states])
#         images = torch.FloatTensor(images).permute(0, 3, 1, 2).to(device) / 255.0

#         next_images = np.stack([s["image"] for s in next_states])
#         next_images = torch.FloatTensor(next_images).permute(0, 3, 1, 2).to(device) / 255.0

#         # --- Position: (batch, 2) ---
#         positions = torch.FloatTensor(np.stack([s["position"] for s in states])).to(device)
#         next_positions = torch.FloatTensor(np.stack([s["position"] for s in next_states])).to(device)

#         # --- Actions discrètes ---
#         action_tensors = {
#             "move": torch.LongTensor([a["move"] for a in actions]).to(device),
#             "jump": torch.LongTensor([a["jump"] for a in actions]).to(device),
#             "hook": torch.LongTensor([a["hook"] for a in actions]).to(device),
#             "fire": torch.LongTensor([a["fire"] for a in actions]).to(device),
#             "weapon": torch.LongTensor([a["weapon"] for a in actions]).to(device),
#         }

#         # --- Rewards, dones ---
#         rewards_t = torch.FloatTensor(rewards).to(device)
#         dones_t = torch.FloatTensor(dones).to(device)

#         # --- Aim targets ---
#         aim_targets_t = torch.FloatTensor(np.stack(aim_targets)).to(device)

#         return {
#             "images": images,
#             "positions": positions,
#             "actions": action_tensors,
#             "rewards": rewards_t,
#             "next_images": next_images,
#             "next_positions": next_positions,
#             "dones": dones_t,
#             "aim_targets": aim_targets_t,
#         }

#     def __len__(self):
#         return len(self.buffer)
    
"""
Replay Buffer Haute Performance pour le DQN.

Utilise des matrices Numpy pré-allouées (Lazy Initialization) pour éviter 
les goulets d'étranglement CPU lors de la création des batchs (Zéro list comprehension, Zéro np.stack).
"""

import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.capacity = capacity
        self.pos = 0
        self.size = 0
        self.initialized = False

    def _init_buffers(self, state: dict, aim_target: np.ndarray):
        """Initialise tous les tableaux Numpy lors de la toute première insertion."""
        # Images (uint8 pour sauver énormément de RAM)
        img_shape = state["image"].shape
        self.images = np.zeros((self.capacity, *img_shape), dtype=np.uint8)
        self.next_images = np.zeros((self.capacity, *img_shape), dtype=np.uint8)

        # Positions
        pos_shape = state["position"].shape
        self.positions = np.zeros((self.capacity, *pos_shape), dtype=np.float32)
        self.next_positions = np.zeros((self.capacity, *pos_shape), dtype=np.float32)

        # Actions (int64 pour PyTorch)
        self.actions_move = np.zeros(self.capacity, dtype=np.int64)
        self.actions_jump = np.zeros(self.capacity, dtype=np.int64)
        self.actions_hook = np.zeros(self.capacity, dtype=np.int64)
        self.actions_fire = np.zeros(self.capacity, dtype=np.int64)
        self.actions_weapon = np.zeros(self.capacity, dtype=np.int64)

        # Récompenses et Dones
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)

        # Aim (Visée vectorielle)
        aim_shape = aim_target.shape
        self.aim_targets = np.zeros((self.capacity, *aim_shape), dtype=np.float32)

        self.initialized = True

    def push(self, state: dict, action: dict, reward: float,
             next_state: dict, done: bool, aim_target: np.ndarray):
        
        # Initialisation "paresseuse" (on s'adapte automatiquement à tes formats d'image)
        if not self.initialized:
            self._init_buffers(state, aim_target)

        # Écriture directe en mémoire au pointeur actuel
        self.images[self.pos] = state["image"]
        self.positions[self.pos] = state["position"]

        self.actions_move[self.pos] = action["move"]
        self.actions_jump[self.pos] = action["jump"]
        self.actions_hook[self.pos] = action["hook"]
        self.actions_fire[self.pos] = action["fire"]
        self.actions_weapon[self.pos] = action.get("weapon", 0) # .get() au cas où l'arme n'est pas passée au début

        self.rewards[self.pos] = reward

        self.next_images[self.pos] = next_state["image"]
        self.next_positions[self.pos] = next_state["position"]

        self.dones[self.pos] = float(done)
        self.aim_targets[self.pos] = aim_target

        # On fait tourner l'index (si on dépasse la capacité, on écrase les plus anciens)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> dict:
        """Échantillonne ultra-rapidement un mini-batch directement sur les tableaux Numpy."""
        
        # On tire aléatoirement `batch_size` index
        idxs = np.random.randint(0, self.size, size=batch_size)

        # --- Tenseurs d'images ---
        # On extrait le sous-tableau, on l'envoie sur le Device, puis on permute et on normalise (0-1)
        # Faire la division par 255.0 sur le GPU est bien plus rapide que sur le CPU
        batch_images = torch.FloatTensor(self.images[idxs]).permute(0, 3, 1, 2).to(device) / 255.0
        batch_next_images = torch.FloatTensor(self.next_images[idxs]).permute(0, 3, 1, 2).to(device) / 255.0

        # --- Autres tenseurs ---
        batch_positions = torch.FloatTensor(self.positions[idxs]).to(device)
        batch_next_positions = torch.FloatTensor(self.next_positions[idxs]).to(device)

        actions = {
            "move": torch.LongTensor(self.actions_move[idxs]).to(device),
            "jump": torch.LongTensor(self.actions_jump[idxs]).to(device),
            "hook": torch.LongTensor(self.actions_hook[idxs]).to(device),
            "fire": torch.LongTensor(self.actions_fire[idxs]).to(device),
            "weapon": torch.LongTensor(self.actions_weapon[idxs]).to(device),
        }

        rewards = torch.FloatTensor(self.rewards[idxs]).to(device)
        dones = torch.FloatTensor(self.dones[idxs]).to(device)
        aim_targets = torch.FloatTensor(self.aim_targets[idxs]).to(device)

        return {
            "images": batch_images,
            "positions": batch_positions,
            "actions": actions,
            "rewards": rewards,
            "next_images": batch_next_images,
            "next_positions": batch_next_positions,
            "dones": dones,
            "aim_targets": aim_targets,
        }

    def __len__(self):
        return self.size