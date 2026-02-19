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

from reinforcement_learning.src.env.econ_client import EconClient
from reinforcement_learning.src.env.screen_capture import ScreenCapture
from reinforcement_learning.src.env.input_controller import InputController

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

    def __init__(self, config: dict, render_mode: Optional[str] = None, shared_econ: Optional[EconClient] = None, agent_id: int = 0):
        super().__init__()
        self.config = config
        self.render_mode = render_mode
        self.agent_id = agent_id
        
        # ---- Composants ----
        self.is_shared_econ = shared_econ is not None
        self.econ = shared_econ if shared_econ else EconClient(
            host=config["server"]["host"],
            port=config["server"]["econ_port"],
            password=config["server"]["econ_password"],
            read_timeout=config["server"]["read_timeout"],
        )
        # self.econ = EconClient(
        #     host=config["server"]["host"],
        #     port=config["server"]["econ_port"],
        #     password=config["server"]["econ_password"],
        #     read_timeout=config["server"]["read_timeout"],
        # )
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
            agent_id=self.agent_id, 
            win_w=config["capture"]["monitor"]["width"],
            win_h=config["capture"]["monitor"]["height"],
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
        self.episode_damage_dealt = 0
        self._pos_history = []
        self._pos_history_size = 10
        self._last_pos = None


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
        # self.econ.restart_round()
        # time.sleep(self.reset_delay)
        if not getattr(self, "is_shared_econ", False):
            self.econ.restart_round()
            time.sleep(self.reset_delay)
            self.econ.poll()
        else:
            time.sleep(self.reset_delay)
        
        # Reset état interne
        self.current_step = 0
        self.episode_kills = 0
        self.episode_deaths = 0
        self.episode_damage_dealt = 0
        self._pos_history = []
        self._last_pos = None
        
        # # Vider le buffer econ
        # self.econ.poll()
        # self.econ.prev_kills = self.econ.kills
        # self.econ.prev_deaths = self.econ.deaths

        # obs = self._get_observation()
        # info = self._get_info()
        # return obs, info
        return self._get_observation(), self._get_info()

    def step(self, action):
        print(f"STEP {self.current_step} | KEYS {action['keys']} | AIM {action['aim']}")
        step_start = time.time()
        self.current_step += 1

        # Appliquer l'action
        keys = action["keys"]
        aim = action["aim"]
        self.controller.apply_action(
            direction=int(keys[0]),
            jump=int(keys[1]),
            fire=int(keys[2]),
            #hook=int(keys[3]),
            hook=0,
            weapon_switch=int(keys[4]) if len(keys)>4 else 0,
            aim_x=float(aim[0]),
            aim_y=float(aim[1]),
        )

        elapsed = time.time() - step_start
        sleep_time = self.tick_interval - elapsed
        
        if sleep_time > 0:
            time.sleep(sleep_time) 

        # Lire les événements serveur
        # self.econ.poll()
        if not getattr(self, "is_shared_econ", False):
            self.econ.poll()

        # Construire la réponse
        obs = self._get_observation()
        reward = self._compute_reward()
        terminated = self.econ.is_player_dead(self.agent_id)
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
        # self.econ.disconnect()
        if not getattr(self, "is_shared_econ", False):
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
        if not self.is_shared_econ:
            if not self.econ.connect():
                raise ConnectionError("Impossible de se connecter à econ")
        logger.info(f"Environnement {self.agent_id} prêt")
    
    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_observation(self) -> dict:
        image = self.capture.grab()
        pos = self.econ.get_position(self.agent_id)
        return {
            "image": image,
            "position": np.array(pos, dtype=np.float32),
        }

    def _compute_reward(self) -> float:
        kills, deaths, damage_dealt = self.econ.get_score(self.agent_id)
        self.episode_kills  += kills
        self.episode_deaths += deaths
        self.episode_damage += damage_dealt

        # ── Rewards combat (priorité haute) ──────────────────────────
        r  = kills        * self.reward_cfg["kill"]          # +10
        r += deaths       * self.reward_cfg["death"]         # -2
        r += damage_dealt * self.reward_cfg["damage_dealt"]  # +1

        # ── Reward mouvement (priorité basse) ─────────────────────────
        current_pos = self.econ.get_position(self.agent_id)
        
        if self._last_pos is not None:
            # Distance parcourue depuis le dernier step
            dx = current_pos[0] - self._last_pos[0]
            dy = current_pos[1] - self._last_pos[1]
            dist = (dx**2 + dy**2) ** 0.5

            # Historique pour détecter la stagnation
            self._pos_history.append(current_pos)
            if len(self._pos_history) > self._pos_history_size:
                self._pos_history.pop(0)

            # Dispersion sur la fenêtre glissante
            # Si l'agent tourne en rond, la dispersion sera faible
            if len(self._pos_history) >= 3:
                xs = [p[0] for p in self._pos_history]
                ys = [p[1] for p in self._pos_history]
                spread = ((max(xs)-min(xs))**2 + (max(ys)-min(ys))**2) ** 0.5
                # Normaliser : 200 unités TW = déplacement significatif
                spread_norm = min(spread / 200.0, 1.0)
            else:
                spread_norm = 0.0

            # Reward mouvement :
            # - bonus si l'agent s'est déplacé ce step
            # - multiplié par la dispersion (anti-rotation sur place)
            # - plafonné pour rester sous le reward combat
            move_bonus = min(dist / 50.0, 0.3) * (0.5 + 0.5 * spread_norm)
            r += move_bonus * self.reward_cfg.get("movement_scale", 0.1)

        self._last_pos = current_pos
        return r


    def _get_info(self) -> dict:
        return {
            "step": self.current_step,
            "kills": self.episode_kills,
            "deaths": self.episode_deaths,
            "damage_dealt": self.episode_damage_dealt,
            "alive": not self.econ.is_player_dead(self.agent_id),
        }
