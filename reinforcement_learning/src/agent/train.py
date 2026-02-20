"""
Entraînement DQN custom pour Ar_2 (multi-head).

Supporte :
- Mode single  : 1 client, 1 env (debug)
- Mode multi   : N clients en grille, N envs en parallèle (training)
"""

import os
import copy
import math
import logging
import numpy as np
import torch
import cv2
import time
import json
import matplotlib.pyplot as plt

# import requests
from concurrent.futures import ThreadPoolExecutor
from torch.utils.tensorboard import SummaryWriter

from reinforcement_learning.src.env.teeworlds_env import TeeWorldsEnv
from reinforcement_learning.src.env.multi_env import MultiEnvManager
from reinforcement_learning.src.agent.network import Ar_2, Ar_2Loss
from reinforcement_learning.src.agent.replay_buffer import ReplayBuffer

logger = logging.getLogger(__name__)

DISCRETE_HEADS = ["move", "jump", "hook", "fire", "weapon"]
HEAD_SIZES = {"move": 3, "jump": 2, "hook": 2, "fire": 2, "weapon": 3}


# ======================================================================
# Sélection d'action
# ======================================================================

def select_action(model: Ar_2, image: torch.Tensor, position: torch.Tensor,
                  epsilon: float, device: torch.device) -> dict:
    """Epsilon-greedy pour les têtes discrètes + sortie réseau pour aim."""
    action = {}

    if np.random.uniform(0, 0.5) < epsilon:
        for head, size in HEAD_SIZES.items():
            action[head] = np.random.randint(size)
        action["aim"] = np.random.uniform(-1, 1, size=(2,)).astype(np.float32)
        #print("random_action : ", action)
    else:
        with torch.no_grad():
            outputs = model(image, position)
        for head in DISCRETE_HEADS:
            action[head] = outputs[head].argmax(dim=1).item()
        aim = outputs["aim"].cpu().numpy()[0]
        norm = np.linalg.norm(aim)
        if norm > 0:
            aim = aim / norm
        action["aim"] = aim

    return action


def select_actions_batch(model: Ar_2, images: torch.Tensor, positions: torch.Tensor,
                         epsilon: float, device: torch.device, n: int) -> list[dict]:
    """Sélection d'actions pour N envs en batch."""
    actions = []

    # Forward en un seul batch pour l'exploitation
    with torch.no_grad():
        outputs = model(images, positions)
    print("epsilon : ",epsilon)
    for i in range(n):
        action = {}
        if np.random.random() < epsilon:
            for head, size in HEAD_SIZES.items():
                action[head] = np.random.randint(size)
            action["aim"] = np.random.uniform(-1, 1, size=(2,)).astype(np.float32)
            #print("random_action : ", action)
        else:
            for head in DISCRETE_HEADS:
                action[head] = outputs[head][i].argmax().item()
            aim = outputs["aim"][i].cpu().numpy()
            norm = np.linalg.norm(aim)
            if norm > 0:
                aim = aim / norm
            action["aim"] = aim
        actions.append(action)

    return actions


def get_epsilon(step: int, eps_start: float, eps_end: float, eps_decay: int) -> float:
    return eps_end + (eps_start - eps_end) * math.exp(-step / eps_decay)


# ======================================================================
# Conversion observation → tensors
# ======================================================================

def obs_to_tensors(obs: dict, device: torch.device):
    """Observation unique → tensors (batch=1)."""
    image = torch.FloatTensor(obs["image"]).unsqueeze(0).permute(0, 3, 1, 2).to(device) / 255.0
    position = torch.FloatTensor(obs["position"]).unsqueeze(0).to(device)
    return image, position


def obs_list_to_tensors(obs_list: list[dict], device: torch.device):
    """Liste d'observations → tensors (batch=N)."""
    images = np.stack([o["image"] for o in obs_list])
    images = torch.FloatTensor(images).permute(0, 3, 1, 2).to(device) / 255.0
    positions = np.stack([o["position"] for o in obs_list])
    positions = torch.FloatTensor(positions).to(device)
    return images, positions


def action_to_env(action: dict) -> dict:
    """Convertit l'action du modèle au format attendu par l'env."""
    aim = action["aim"]
    angle = math.atan2(aim[0], aim[1])
    aim_x = math.cos(angle)
    aim_y = math.sin(angle)
    return {
        "keys": np.array([action["move"], action["jump"], action["fire"], action["hook"], action["weapon"]]),
        "aim": np.array([aim_x, aim_y], dtype=np.float32),
    }


# ======================================================================
# Entraînement commun (update)
# ======================================================================

def train_step(model, target_model, optimizer, criterion, buffer, batch_size, device):
    """Un step d'entraînement sur un batch du replay buffer."""
    batch = buffer.sample(batch_size, device)

    outputs = model(batch["images"], batch["positions"])
    with torch.no_grad():
        next_outputs = target_model(batch["next_images"], batch["next_positions"])

    # print("actions", batch["actions"])
    # print("rewards", batch["rewards"])
    # print("aim", batch["aim_targets"])

    loss = criterion(
        outputs=outputs,
        actions=batch["actions"],
        rewards=batch["rewards"],
        next_outputs=next_outputs,
        dones=batch["dones"],
        aim_targets=batch["aim_targets"],
    )
    print("loss = ", loss)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
    optimizer.step()

    return loss.item()


# ======================================================================
# Mode multi-env (grille de clients)
# ======================================================================

def train_multi(config: dict):
    """Entraînement avec N environnements en parallèle."""

    train_cfg = config["training"]
    multi_cfg = config.get("multi_env", {})
    # visualizer_cfg = config.get("visualizer", {})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    n_envs = multi_cfg.get("n_envs", 4)
    grid_cols = multi_cfg.get("grid_cols", None)
    screen_w = multi_cfg.get("screen_width", 1920)
    screen_h = multi_cfg.get("screen_height", 1080)
    tw_binary = multi_cfg.get("tw_binary", "teeworlds")
    server_ip = config["server"]["host"]
    server_port = multi_cfg.get("game_port", 8303)

    os.makedirs(train_cfg["save_path"], exist_ok=True)
    os.makedirs(train_cfg["log_path"], exist_ok=True)
    writer = SummaryWriter(train_cfg["log_path"])

    # ---- Multi-env ----
    logger.info(f"Lancement de {n_envs} environnements en grille...")
    manager = MultiEnvManager(
        config=config,
        n_envs=n_envs,
        grid_cols=grid_cols,
        screen_width=screen_w,
        screen_height=screen_h,
        tw_binary=tw_binary,
        server_ip=server_ip,
        server_port=server_port,
    )
    manager.launch_clients(connect_delay=multi_cfg.get("connect_delay", 3.0))
    manager.create_envs()

    actual_n = len(manager.envs)
    if actual_n == 0:
        logger.error("Aucun environnement créé, abandon")
        return
    logger.info(f"{actual_n} environnements actifs")

    # ---- Modèles ----
    input_size = 2
    model = Ar_2(input_size=input_size, backbone_pretrained=train_cfg.get("pretrained", False)).to(device)
    target_model = copy.deepcopy(model)
    target_model.eval()
    inference_model = copy.deepcopy(model)
    inference_model.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=train_cfg["learning_rate"],          # pic au milieu
        total_steps=train_cfg["total_timesteps"]*actual_n,
        pct_start=0.1,        # 30% du training pour monter, 70% pour descendre
        div_factor=1e1,      # lr_start = max_lr / 25
        final_div_factor=1e4, # lr_end  = max_lr / (25 * 10000)
        anneal_strategy="cos",
    )

    criterion = Ar_2Loss(gamma=train_cfg["gamma"], aim_weight=train_cfg.get("aim_weight", 1.0))
    buffer = ReplayBuffer(capacity=train_cfg.get("buffer_size", 100_000))

    loss_history = []

    # ---- Hyperparamètres ----
    batch_size = train_cfg["batch_size"]
    eps_start = train_cfg.get("eps_start", 1.0)
    eps_end = train_cfg.get("eps_end", 0.05)
    eps_decay = train_cfg.get("eps_decay", 50_000)
    target_update = train_cfg.get("target_update", 100)
    min_buffer_size = train_cfg.get("min_buffer_size", 100)
    total_timesteps = train_cfg["total_timesteps"]
    save_freq = train_cfg["save_freq"]

    # ---- Boucle ----
    global_step = 0
    episode_counts = [0] * actual_n
    episode_rewards = [0.0] * actual_n

    # Reset initial
    obs_list = manager.reset_all()

    train_executor = ThreadPoolExecutor(max_workers=1)
    train_future = None

    logger.info(f"Début de l'entraînement ({total_timesteps} timesteps, {actual_n} envs)...")
    
    try:
        while global_step < total_timesteps * actual_n:
            start_time = time.time()
            global_step += actual_n
            epsilon = get_epsilon(global_step, eps_start, eps_end, eps_decay)

            # Inférence sur inference_model (safe, pas de backward dessus)
            images, positions = obs_list_to_tensors(obs_list, device)
            actions = select_actions_batch(inference_model, images, positions, epsilon, device, actual_n)

            env_actions = [action_to_env(a) for a in actions]
            results = manager.step_all(env_actions)

            # Stocker les transitions
            new_obs_list = []
            for i, (obs, act, (next_obs, reward, terminated, truncated, info)) in \
                    enumerate(zip(obs_list, actions, results)):
                done = terminated or truncated
                buffer.push(obs, act, reward, next_obs, float(done), act["aim"])
                episode_rewards[i] += reward

                if done:
                    episode_counts[i] += 1
                    writer.add_scalar(f"env_{i}/episode_reward", episode_rewards[i], episode_counts[i])
                    writer.add_scalar(f"env_{i}/kills", info.get("kills", 0), episode_counts[i])
                    writer.add_scalar(f"env_{i}/deaths", info.get("deaths", 0), episode_counts[i])
                    logger.info(
                        f"Env {i} | Episode {episode_counts[i]} | "
                        f"reward={episode_rewards[i]:.2f} | "
                        f"kills={info.get('kills', 0)} deaths={info.get('deaths', 0)}"
                    )
                    episode_rewards[i] = 0.0
                    reset_obs, _ = manager.envs[i].reset()
                    new_obs_list.append(reset_obs)
                else:
                    new_obs_list.append(next_obs)

            obs_list = new_obs_list

            # ---- Training asynchrone — UN SEUL bloc ----
            if len(buffer) >= min_buffer_size:
                if train_future is None:
                    train_future = train_executor.submit(
                        train_step, model, target_model, optimizer,
                        criterion, buffer, batch_size, device
                    )
                elif train_future.done():
                    try:
                        loss = train_future.result()
                        ## add loss to history
                        loss_history.append((global_step/actual_n, loss))

                        scheduler.step()  # ← ici
                        current_lr = optimizer.param_groups[0]["lr"]

                        inference_model.load_state_dict(model.state_dict())
                        inference_model.eval()
                        if global_step % (200 * actual_n) == 0:
                            writer.add_scalar("train/loss", loss, global_step)
                            writer.add_scalar("train/epsilon", epsilon, global_step)
                            writer.add_scalar("train/buffer_size", len(buffer), global_step)
                            writer.add_scalar("train/learning_rate", current_lr, global_step)
                    except Exception as e:
                        logger.error(f"Erreur train_step: {e}")

                    train_future = train_executor.submit(
                        train_step, model, target_model, optimizer,
                        criterion, buffer, batch_size, device
                    )

            # ---- Target network ----
            if global_step % target_update == 0:
                target_model.load_state_dict(model.state_dict())

            # ---- Sauvegarde ----
            if global_step % save_freq == 0:
                path = os.path.join(train_cfg["save_path"], f"ar2_step_{global_step}.pt")
                torch.save(model.state_dict(), path)
                logger.info(f"Checkpoint: {path} (eps={epsilon:.3f})")
            print(f"STEP: {global_step/actual_n:.0f} | lr={optimizer.param_groups[0]['lr']:.2e} | eps={epsilon:.3f}")

            estimated_time = (time.time() - start_time)*(total_timesteps-global_step/actual_n)
            print('estimated time = ', f"{estimated_time//3600}h {estimated_time % 3600}s")

    except KeyboardInterrupt:
        logger.info("Entraînement interrompu")
    finally:
        if train_future is not None:
            train_future.cancel()
        train_executor.shutdown(wait=False)
        

        loss_path = os.path.join(train_cfg["save_path"], "loss_history.json")
        with open(loss_path, "w") as f:
            json.dump(loss_history, f)
        logger.info(f"Loss history saved: {loss_path}")

        # Plot
        if loss_history:

            steps, losses = zip(*loss_history)
            plt.figure(figsize=(12, 4))
            plt.plot(steps, losses, alpha=0.4, color="steelblue", label="loss brute")
            
            # Moyenne glissante
            window = min(50, len(losses))
            moving_avg = np.convolve(losses, np.ones(window)/window, mode="valid")
            plt.plot(steps[window-1:], moving_avg, color="red", linewidth=2, label=f"moyenne ({window})")
            
            plt.xlabel("Steps")
            plt.ylabel("Loss")
            plt.title("Training Loss")
            plt.legend()
            plt.tight_layout()
            plot_path = os.path.join(train_cfg["save_path"], "loss_plot.png")
            plt.savefig(plot_path, dpi=150)
            plt.close()
            logger.info(f"Loss plot saved: {plot_path}")

        final_path = os.path.join(train_cfg["save_path"], "ar2_final.pt")
        torch.save(model.state_dict(), final_path)
        logger.info(f"Modèle final: {final_path}")
        writer.close()
        manager.close()


# ======================================================================
# Mode single-env (debug / évaluation)
# ======================================================================

def train_single(config: dict):
    """Entraînement avec un seul environnement (mode debug)."""

    train_cfg = config["training"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    os.makedirs(train_cfg["save_path"], exist_ok=True)
    os.makedirs(train_cfg["log_path"], exist_ok=True)
    writer = SummaryWriter(train_cfg["log_path"])

    # ---- Environnement ----
    env = TeeWorldsEnv(config)
    env.setup()

    # ---- Modèles ----
    input_size = 2
    model = Ar_2(input_size=input_size, backbone_pretrained=train_cfg.get("pretrained", False)).to(device)
    target_model = copy.deepcopy(model)
    target_model.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])
    criterion = Ar_2Loss(gamma=train_cfg["gamma"], aim_weight=train_cfg.get("aim_weight", 1.0))
    buffer = ReplayBuffer(capacity=train_cfg.get("buffer_size", 100_000))

    batch_size = train_cfg["batch_size"]
    eps_start = train_cfg.get("eps_start", 1.0)
    eps_end = train_cfg.get("eps_end", 0.05)
    eps_decay = train_cfg.get("eps_decay", 50_000)
    target_update = train_cfg.get("target_update", 1000)
    min_buffer_size = train_cfg.get("min_buffer_size", 100)
    total_timesteps = train_cfg["total_timesteps"]
    save_freq = train_cfg["save_freq"]

    global_step = 0
    episode = 0

    logger.info(f"Début de l'entraînement single ({total_timesteps} timesteps)...")

    try:
        while global_step < total_timesteps:
            obs, info = env.reset()
            # print("RAW position:", obs["position"])  # Is it actually [0,0] from the env?
            # print("RAW image shape:", obs["image"].shape)
            # print("RAW image min/max:", obs["image"].min(), obs["image"].max())
            episode_reward = 0
            done = False
            episode += 1

            while not done and global_step < total_timesteps:
                global_step += 1
                epsilon = get_epsilon(global_step, eps_start, eps_end, eps_decay)

                image, position = obs_to_tensors(obs, device)
                action = select_action(model, image, position, epsilon, device)

                env_action = action_to_env(action)
                next_obs, reward, terminated, truncated, info = env.step(env_action)
                done = terminated or truncated

                buffer.push(obs, action, reward, next_obs, float(done), action["aim"])
                obs = next_obs
                episode_reward += reward

                if len(buffer) >= min_buffer_size:
                    loss = train_step(model, target_model, optimizer, criterion,
                                      buffer, batch_size, device)
                    if global_step % 100 == 0:
                        writer.add_scalar("train/loss", loss, global_step)
                        writer.add_scalar("train/epsilon", epsilon, global_step)

                if global_step % target_update == 0:
                    target_model.load_state_dict(model.state_dict())

                if global_step % save_freq == 0:
                    path = os.path.join(train_cfg["save_path"], f"ar2_step_{global_step}.pt")
                    torch.save(model.state_dict(), path)

            writer.add_scalar("episode/reward", episode_reward, episode)
            logger.info(
                f"Episode {episode} | reward={episode_reward:.2f} | "
                f"kills={info.get('kills', 0)} deaths={info.get('deaths', 0)} | "
                f"eps={epsilon:.3f} | step={global_step}"
            )

    except KeyboardInterrupt:
        logger.info("Entraînement interrompu")
    finally:
        final_path = os.path.join(train_cfg["save_path"], "ar2_final.pt")
        torch.save(model.state_dict(), final_path)
        logger.info(f"Modèle final: {final_path}")
        writer.close()
        env.close()


# ======================================================================
# Point d'entrée
# ======================================================================

def train(config: dict):
    """Dispatch vers single ou multi selon la config."""
    if config.get("multi_env", {}).get("enabled", False):
        train_multi(config)
    else:
        train_single(config)


def load_and_play(config: dict, model_path: str, n_episodes: int = 10):
    """Charge un modèle et le fait jouer (epsilon=0)."""
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