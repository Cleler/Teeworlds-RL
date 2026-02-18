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
        print("random_action : ", action)
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
    print(epsilon)
    for i in range(n):
        action = {}
        if np.random.random() < epsilon:
            for head, size in HEAD_SIZES.items():
                action[head] = np.random.randint(size)
            action["aim"] = np.random.uniform(-1, 1, size=(2,)).astype(np.float32)
            print("random_action : ", action)
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

    print("actions", batch["actions"])
    print("rewards", batch["rewards"])
    print("aim", batch["aim_targets"])

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

    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])
    criterion = Ar_2Loss(gamma=train_cfg["gamma"], aim_weight=train_cfg.get("aim_weight", 1.0))
    buffer = ReplayBuffer(capacity=train_cfg.get("buffer_size", 100_000))

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

    # visualizer_ip = visualizer_cfg.get("ip", "192.168.22.116")
    # visualizer_url = f"http://{visualizer_ip}:5000/update/"
    
    # network_executor = ThreadPoolExecutor(max_workers=4)
    
    # def send_frame_to_visualizer(bot_id, frame):
    #     try:
    #         if len(frame.shape) == 3 and frame.shape[2] == 1:
    #             frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    #         ret, buffer = cv2.imencode('.jpg', frame)
            
    #         if ret:
    #             requests.post(f"{visualizer_url}{bot_id}", data=buffer.tobytes(), timeout=2.0)
    #         else:
    #             print(f"⚠️ Erreur: OpenCV n'a pas pu encoder l'image du bot {bot_id}")
                
    #     except Exception as e:
    #         print(f"⚠️ Erreur réseau (Visualizer Bot {bot_id}) : {e}")
                    
    logger.info(f"Début de l'entraînement ({total_timesteps} timesteps, {actual_n} envs)...")

    try:
        while global_step < total_timesteps:
            global_step += actual_n
            epsilon = get_epsilon(global_step, eps_start, eps_end, eps_decay)

            # Sélection d'actions en batch
            images, positions = obs_list_to_tensors(obs_list, device)
            actions = select_actions_batch(model, images, positions, epsilon, device, actual_n)

            # Step sur tous les envs
            env_actions = [action_to_env(a) for a in actions]
            print(env_actions)
            results = manager.step_all(env_actions)


            # Stocker les transitions et gérer les épisodes
            new_obs_list = []
            for i, (obs, act, (next_obs, reward, terminated, truncated, info)) in \
                    enumerate(zip(obs_list, actions, results)):

                done = terminated or truncated
                aim_target = act["aim"]
                buffer.push(obs, act, reward, next_obs, float(done), aim_target)
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

                    # Reset cet env
                    reset_obs, _ = manager.envs[i].reset()
                    new_obs_list.append(reset_obs)
                else:
                    new_obs_list.append(next_obs)

            obs_list = new_obs_list
            
            # for i, obs in enumerate(obs_list):
            #     hd_frame = manager.envs[i].capture.grab_raw()
            #     network_executor.submit(send_frame_to_visualizer, i, hd_frame)
                
            print("="*50)

            print("type_buffer", type(buffer))
            print("buffer_pos : ", buffer.positions)
            #print("buffer_actions : ", buffer.actions)
            print("buffer_aim : ", buffer.aim_targets)
            print("buffer_reward : ", buffer.rewards)
            print("buffer_next_state : ", buffer.next_positions)
            #print("buffer_aim : ", buffer.buffer[0]['aim'])
            print(min_buffer_size)
            # ---- Entraînement ----

            if len(buffer) >= min_buffer_size:
                loss = train_step(model, target_model, optimizer, criterion,
                                  buffer, batch_size, device)
                print("loss", loss)

                if global_step % (100 * actual_n) == 0:
                    writer.add_scalar("train/loss", loss, global_step)
                    writer.add_scalar("train/epsilon", epsilon, global_step)
                    writer.add_scalar("train/buffer_size", len(buffer), global_step)

            # ---- Target network ----
            if global_step % target_update == 0:
                target_model.load_state_dict(model.state_dict())

            # ---- Sauvegarde ----
            if global_step % save_freq == 0:
                path = os.path.join(train_cfg["save_path"], f"ar2_step_{global_step}.pt")
                torch.save(model.state_dict(), path)
                logger.info(f"Checkpoint: {path} (eps={epsilon:.3f})")

    except KeyboardInterrupt:
        logger.info("Entraînement interrompu")
    finally:
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
            print("RAW position:", obs["position"])  # Is it actually [0,0] from the env?
            print("RAW image shape:", obs["image"].shape)
            print("RAW image min/max:", obs["image"].min(), obs["image"].max())
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