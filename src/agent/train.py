"""
Logique d'entraînement de l'agent RL.

Utilise Stable-Baselines3 avec PPO (recommandé pour les espaces d'actions mixtes).
"""

import os
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor

from src.env.teeworlds_env import TeeWorldsEnv
from src.agent.network import get_policy_kwargs

logger = logging.getLogger(__name__)


def make_env(config: dict) -> TeeWorldsEnv:
    """Crée et initialise l'environnement."""
    env = TeeWorldsEnv(config)
    env.setup()
    env = Monitor(env)  # Wrapper SB3 pour le logging
    return env


def train(config: dict):
    """Lance l'entraînement."""

    train_cfg = config["training"]

    # Créer les dossiers
    os.makedirs(train_cfg["save_path"], exist_ok=True)
    os.makedirs(train_cfg["log_path"], exist_ok=True)

    # Environnement
    logger.info("Création de l'environnement...")
    env = make_env(config)

    # Modèle
    logger.info(f"Initialisation de {train_cfg['algorithm']}...")
    model = PPO(
        policy="MultiInputPolicy",  # Gère les observations Dict
        env=env,
        learning_rate=train_cfg["learning_rate"],
        n_steps=train_cfg["n_steps"],
        batch_size=train_cfg["batch_size"],
        gamma=train_cfg["gamma"],
        verbose=1,
        tensorboard_log=train_cfg["log_path"],
        policy_kwargs=get_policy_kwargs(),
        device="auto",  # GPU si disponible
    )

    # Callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=train_cfg["save_freq"],
        save_path=train_cfg["save_path"],
        name_prefix="teeworlds_rl",
    )

    callbacks = CallbackList([checkpoint_cb])

    # Entraînement
    logger.info(f"Début de l'entraînement ({train_cfg['total_timesteps']} timesteps)...")
    try:
        model.learn(
            total_timesteps=train_cfg["total_timesteps"],
            callback=callbacks,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        logger.info("Entraînement interrompu par l'utilisateur")
    finally:
        # Sauvegarde finale
        final_path = os.path.join(train_cfg["save_path"], "final_model")
        model.save(final_path)
        logger.info(f"Modèle sauvegardé: {final_path}")
        env.close()


def load_and_play(config: dict, model_path: str, n_episodes: int = 10):
    """Charge un modèle entraîné et le fait jouer."""

    env = make_env(config)
    model = PPO.load(model_path, env=env)

    for episode in range(n_episodes):
        obs, info = env.reset()
        total_reward = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        logger.info(
            f"Episode {episode + 1}: reward={total_reward:.2f} "
            f"kills={info['kills']} deaths={info['deaths']}"
        )

    env.close()
