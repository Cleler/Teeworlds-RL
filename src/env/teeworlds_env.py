"""
Wrapper Gymnasium pour Teeworlds.

Assemble les 3 composants (econ, screen capture, input controller)
en une interface Gymnasium standard compatible avec Stable-Baselines3.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import logging
from typing import Optional

from src.env.econ_client import EconClient
from src.env.screen_capture import ScreenCapture
from src.env.input_controller import InputController

logger = logging.getLogger(__name__)


class TeeWorldsEnv(gym.Env):
    """
    Environnement Gymnasium pour Teeworlds.

    Observation:
        Dict:
            "image": Box(0, 255, shape=(84, 84, 1))  - screenshot grayscale
            "position": Box(-inf, inf, shape=(2,))    - position (x, y) du joueur

    Action:
        Dict:
            "keys": MultiDiscrete([3, 2, 2, 2])  - [direction, jump, fire, hook]
            "aim": Box(-1, 1, shape=(2,))         - direction de visée (x, y)
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, config: dict, render_mode: Optional[str] = None):
        super().__init__()
        self.config = config
        self.render_mode = render_mode

        # ---- Composants ----
        self.econ = EconClient(
            host=config["server"]["host"],
            port=config["server"]["econ_port"],
            password=config["server"]["econ_password"],
            read_timeout=config["server"]["read_timeout"],
        )
        self.capture = ScreenCapture(
            monitor=config["capture"]["monitor"],
            obs_width=config["capture"]["obs_width"],
            obs_height=config["capture"]["obs_height"],
            grayscale=config["capture"]["grayscale"],
        )

        key_mapping = config["input"]["keys"]
        monitor = config["capture"]["monitor"]
        center_x = monitor["left"] + monitor["width"] // 2
        center_y = monitor["top"] + monitor["height"] // 2

        self.controller = InputController(
            key_mapping=key_mapping,
            aim_radius=config["input"]["aim_radius"],
            screen_center=(center_x, center_y),
        )

        # ---- Paramètres ----
        self.tick_rate = config["env"]["tick_rate"]
        self.tick_interval = 1.0 / self.tick_rate
        self.max_steps = config["env"]["max_steps"]
        self.reset_delay = config["env"]["reset_delay"]
        self.reward_config = config["reward"]

        # ---- État interne ----
        self.current_step = 0
        self.episode_kills = 0
        self.episode_deaths = 0

        # ---- Espaces Gymnasium ----
        obs_h = config["capture"]["obs_height"]
        obs_w = config["capture"]["obs_width"]
        channels = 1 if config["capture"]["grayscale"] else 3

        self.observation_space = spaces.Dict({
            "image": spaces.Box(0, 255, shape=(obs_h, obs_w, channels), dtype=np.uint8),
            "position": spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32),
        })

        self.action_space = spaces.Dict({
            "keys": spaces.MultiDiscrete([3, 2, 2, 2, 3]),  # direction, jump, fire, hook, weapon switch
            "aim": spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
        })

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        logger.info("Reset de l'environnement")

        # Relâcher tous les inputs
        self.controller.release_all()

        # Restart via econ
        self.econ.restart_round()
        time.sleep(self.reset_delay)

        # Reset état interne
        self.current_step = 0
        self.episode_kills = 0
        self.episode_deaths = 0

        # Vider le buffer econ
        self.econ.poll()
        self.econ.prev_kills = self.econ.kills
        self.econ.prev_deaths = self.econ.deaths

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(self, action):
        self.current_step += 1

        # Appliquer l'action
        keys = action["keys"]
        aim = action["aim"]
        self.controller.apply_action(
            direction=int(keys[0]),
            jump=int(keys[1]),
            fire=int(keys[2]),
            hook=int(keys[3]),
            aim_x=float(aim[0]),
            aim_y=float(aim[1]),
        )

        # Attendre le prochain tick
        time.sleep(self.tick_interval)

        # Lire les événements serveur
        self.econ.poll()

        # Construire la réponse
        obs = self._get_observation()
        reward = self._compute_reward()
        terminated = self.econ.is_player_dead()
        truncated = self.current_step >= self.max_steps
        info = self._get_info()

        # Si mort, relâcher les touches
        if terminated:
            self.controller.release_all()

        return obs, reward, terminated, truncated, info

    def close(self):
        logger.info("Fermeture de l'environnement")
        self.controller.release_all()
        self.capture.stop()
        self.econ.disconnect()

    # ------------------------------------------------------------------
    # Setup / Teardown
    # ------------------------------------------------------------------

    def setup(self):
        """
        Initialise les connexions. À appeler avant le premier reset().
        Séparé de __init__ pour permettre la gestion d'erreurs.
        """
        self.capture.start()
        if not self.econ.connect():
            raise ConnectionError("Impossible de se connecter à econ")
        logger.info("Environnement prêt")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_observation(self) -> dict:
        image = self.capture.grab()
        x, y = self.econ.get_position()
        return {
            "image": image,
            "position": np.array([x, y], dtype=np.float32),
        }

    def _compute_reward(self) -> float:
        kills, deaths = self.econ.get_score()
        self.episode_kills += kills
        self.episode_deaths += deaths

        reward = 0.0
        reward += kills * self.reward_config["kill"]
        reward += deaths * self.reward_config["death"]
        reward += self.reward_config["survival_bonus"]

        return reward

    def _get_info(self) -> dict:
        return {
            "step": self.current_step,
            "kills": self.episode_kills,
            "deaths": self.episode_deaths,
            "alive": not self.econ.is_player_dead(),
        }
