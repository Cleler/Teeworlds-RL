"""
Entraînement DQN custom pour Ar_2 (multi-head).

Boucle classique DQN :
1. Collecter des transitions avec epsilon-greedy
2. Stocker dans le replay buffer
3. Échantillonner un batch et calculer la loss Ar_2Loss
4. Mettre à jour le target network périodiquement
"""

import os
import copy
import math
import logging
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from src.env.teeworlds_env import TeeWorldsEnv
from src.agent.network import Ar_2, Ar_2Loss
from src.agent.replay_buffer import ReplayBuffer

logger = logging.getLogger(__name__)

DISCRETE_HEADS = ["move", "jump", "hook", "fire"]
HEAD_SIZES = {"move": 3, "jump": 2, "hook": 2, "fire": 2}


# ======================================================================
# Sélection d'action (epsilon-greedy + aim)
# ======================================================================

def select_action(model: Ar_2, image: torch.Tensor, position: torch.Tensor,
                  epsilon: float, device: torch.device) -> dict:
    """
    Sélectionne une action epsilon-greedy pour les têtes discrètes
    et utilise directement la sortie du réseau pour aim.

    Returns:
        {"move": int, "jump": int, "hook": int, "fire": int, "aim": np.ndarray(2,)}
    """
    action = {}

    if np.random.random() < epsilon:
        # Exploration aléatoire
        for head, size in HEAD_SIZES.items():
            action[head] = np.random.randint(size)
        action["aim"] = np.random.uniform(-1, 1, size=(2,)).astype(np.float32)
    else:
        # Exploitation
        with torch.no_grad():
            outputs = model(image, position)

        for head in DISCRETE_HEADS:
            action[head] = outputs[head].argmax(dim=1).item()

        aim = outputs["aim"].cpu().numpy()[0]
        # Normaliser en (sin, cos) unitaire
        norm = np.linalg.norm(aim)
        if norm > 0:
            aim = aim / norm
        action["aim"] = aim

    return action


def get_epsilon(step: int, eps_start: float, eps_end: float, eps_decay: int) -> float:
    """Epsilon décroissant exponentiellement."""
    return eps_end + (eps_start - eps_end) * math.exp(-step / eps_decay)


# ======================================================================
# Conversion observation → tensors
# ======================================================================

def obs_to_tensors(obs: dict, device: torch.device):
    """Convertit une observation Gymnasium en tensors pour le modèle."""
    image = torch.FloatTensor(obs["image"]).unsqueeze(0).permute(0, 3, 1, 2).to(device) / 255.0
    position = torch.FloatTensor(obs["position"]).unsqueeze(0).to(device)
    return image, position


def action_to_env(action: dict) -> dict:
    """Convertit l'action du modèle au format attendu par l'env Gymnasium."""
    aim = action["aim"]
    angle = math.atan2(aim[0], aim[1])  # sin, cos → angle
    aim_x = math.cos(angle)
    aim_y = math.sin(angle)

    return {
        "keys": np.array([action["move"], action["jump"], action["fire"], action["hook"]]),
        "aim": np.array([aim_x, aim_y], dtype=np.float32),
    }


# ======================================================================
# Boucle d'entraînement
# ======================================================================

def train(config: dict):
    """Lance l'entraînement DQN."""

    train_cfg = config["training"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Dossiers
    os.makedirs(train_cfg["save_path"], exist_ok=True)
    os.makedirs(train_cfg["log_path"], exist_ok=True)
    writer = SummaryWriter(train_cfg["log_path"])

    # ---- Environnement ----
    logger.info("Création de l'environnement...")
    env = TeeWorldsEnv(config)
    env.setup()

    # ---- Modèles ----
    input_size = 2  # position (x, y)
    model = Ar_2(input_size=input_size, backbone_pretrained=train_cfg.get("pretrained", False)).to(device)
    target_model = copy.deepcopy(model)
    target_model.eval()

    # ---- Optimiseur et loss ----
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])
    criterion = Ar_2Loss(
        gamma=train_cfg["gamma"],
        aim_weight=train_cfg.get("aim_weight", 1.0),
    )

    # ---- Replay buffer ----
    buffer = ReplayBuffer(capacity=train_cfg.get("buffer_size", 100_000))

    # ---- Hyperparamètres ----
    batch_size = train_cfg["batch_size"]
    eps_start = train_cfg.get("eps_start", 1.0)
    eps_end = train_cfg.get("eps_end", 0.05)
    eps_decay = train_cfg.get("eps_decay", 50_000)
    target_update = train_cfg.get("target_update", 1000)
    min_buffer_size = train_cfg.get("min_buffer_size", 1000)
    total_timesteps = train_cfg["total_timesteps"]
    save_freq = train_cfg["save_freq"]

    # ---- Boucle ----
    global_step = 0
    episode = 0

    logger.info(f"Début de l'entraînement ({total_timesteps} timesteps)...")

    try:
        while global_step < total_timesteps:
            obs, info = env.reset()
            episode_reward = 0
            episode_steps = 0
            done = False
            episode += 1

            while not done and global_step < total_timesteps:
                global_step += 1
                episode_steps += 1

                # Epsilon-greedy
                epsilon = get_epsilon(global_step, eps_start, eps_end, eps_decay)
                image, position = obs_to_tensors(obs, device)
                action = select_action(model, image, position, epsilon, device)

                # Step dans l'environnement
                env_action = action_to_env(action)
                next_obs, reward, terminated, truncated, info = env.step(env_action)
                done = terminated or truncated

                # Aim target = l'aim qu'on a réellement envoyé (pour supervision)
                aim_target = action["aim"]

                # Stocker la transition
                buffer.push(obs, action, reward, next_obs, float(done), aim_target)
                obs = next_obs
                episode_reward += reward

                # ---- Entraînement ----
                if len(buffer) >= min_buffer_size:
                    batch = buffer.sample(batch_size, device)

                    # Forward main model
                    outputs = model(batch["images"], batch["positions"])

                    # Forward target model
                    with torch.no_grad():
                        next_outputs = target_model(batch["next_images"], batch["next_positions"])

                    # Loss
                    loss = criterion(
                        outputs=outputs,
                        actions=batch["actions"],
                        rewards=batch["rewards"],
                        next_outputs=next_outputs,
                        dones=batch["dones"],
                        aim_targets=batch["aim_targets"],
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                    optimizer.step()

                    # Logging
                    if global_step % 100 == 0:
                        writer.add_scalar("train/loss", loss.item(), global_step)
                        writer.add_scalar("train/epsilon", epsilon, global_step)

                # ---- Mise à jour du target network ----
                if global_step % target_update == 0:
                    target_model.load_state_dict(model.state_dict())
                    logger.debug(f"Target network mis à jour (step {global_step})")

                # ---- Sauvegarde ----
                if global_step % save_freq == 0:
                    path = os.path.join(train_cfg["save_path"], f"ar2_step_{global_step}.pt")
                    torch.save(model.state_dict(), path)
                    logger.info(f"Checkpoint sauvegardé: {path}")

            # Fin de l'épisode
            writer.add_scalar("episode/reward", episode_reward, episode)
            writer.add_scalar("episode/steps", episode_steps, episode)
            writer.add_scalar("episode/kills", info.get("kills", 0), episode)
            writer.add_scalar("episode/deaths", info.get("deaths", 0), episode)

            logger.info(
                f"Episode {episode} | steps={episode_steps} | reward={episode_reward:.2f} | "
                f"kills={info.get('kills', 0)} | deaths={info.get('deaths', 0)} | "
                f"eps={epsilon:.3f} | global_step={global_step}"
            )

    except KeyboardInterrupt:
        logger.info("Entraînement interrompu")
    finally:
        final_path = os.path.join(train_cfg["save_path"], "ar2_final.pt")
        torch.save(model.state_dict(), final_path)
        logger.info(f"Modèle final sauvegardé: {final_path}")
        writer.close()
        env.close()


# ======================================================================
# Évaluation
# ======================================================================

def load_and_play(config: dict, model_path: str, n_episodes: int = 10):
    """Charge un modèle et le fait jouer (deterministic, epsilon=0)."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = TeeWorldsEnv(config)
    env.setup()

    model = Ar_2(input_size=2, backbone_pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    for ep in range(n_episodes):
        obs, info = env.reset()
        total_reward = 0
        done = False

        while not done:
            image, position = obs_to_tensors(obs, device)
            action = select_action(model, image, position, epsilon=0.0, device=device)
            env_action = action_to_env(action)
            obs, reward, terminated, truncated, info = env.step(env_action)
            total_reward += reward
            done = terminated or truncated

        logger.info(
            f"Episode {ep + 1}: reward={total_reward:.2f} "
            f"kills={info.get('kills', 0)} deaths={info.get('deaths', 0)}"
        )

    env.close()