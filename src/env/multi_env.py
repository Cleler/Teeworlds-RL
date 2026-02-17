"""
Gestionnaire multi-environnement pour Teeworlds RL.

Lance N clients Teeworlds, les positionne en grille sur l'écran,
et crée un environnement Gymnasium par client. Chaque env a sa propre
capture d'écran et ses propres inputs ciblés via xdotool.

Workflow :
    1. Lancer le(s) serveur(s) TW
    2. Lancer le MultiEnvManager qui ouvre N clients
    3. Chaque client rejoint le serveur automatiquement
    4. L'agent RL step() sur tous les envs en parallèle

Prérequis :
    sudo apt install xdotool xdg-utils
"""

import subprocess
import time
import math
import logging
import os
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

from src.env.teeworlds_env import TeeWorldsEnv

logger = logging.getLogger(__name__)


class MultiEnvManager:
    def __init__(self, config: dict, n_envs: int = 4,
                 grid_cols: Optional[int] = None,
                 screen_width: int = 1920, screen_height: int = 1080,
                 tw_binary: str = "teeworlds",
                 server_ip: str = "127.0.0.1", server_port: int = 8303):
        """
        Args:
            config: Config de base (sera adaptée par env).
            n_envs: Nombre d'environnements parallèles.
            grid_cols: Nombre de colonnes dans la grille (auto si None).
            screen_width: Largeur de l'écran en pixels.
            screen_height: Hauteur de l'écran en pixels.
            tw_binary: Chemin vers le binaire client Teeworlds.
            server_ip: IP du serveur TW.
            server_port: Port du serveur TW.
        """
        self.config = config
        self.n_envs = n_envs
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.tw_binary = tw_binary
        self.server_ip = server_ip
        self.server_port = server_port

        # Calculer la grille
        if grid_cols is None:
            self.grid_cols = math.ceil(math.sqrt(n_envs))
        else:
            self.grid_cols = grid_cols
        self.grid_rows = math.ceil(n_envs / self.grid_cols)

        # Taille de chaque fenêtre client
        self.cell_w = screen_width // self.grid_cols
        self.cell_h = screen_height // self.grid_rows

        logger.info(
            f"Grille: {self.grid_rows}x{self.grid_cols} "
            f"({n_envs} envs, cellule={self.cell_w}x{self.cell_h})"
        )

        # État
        self.client_processes: list[subprocess.Popen] = []
        self.client_window_ids: list[str] = []
        self.envs: list[TeeWorldsEnv] = []
        self.executor = ThreadPoolExecutor(max_workers=n_envs)

    # ------------------------------------------------------------------
    # Lancement des clients
    # ------------------------------------------------------------------

    def launch_clients(self, connect_delay: float = 2.0):
        """
        Lance N clients TW et les positionne en grille.

        Chaque client est lancé avec des arguments pour se connecter
        automatiquement au serveur.
        """
        logger.info(f"Lancement de {self.n_envs} clients Teeworlds...")

        for i in range(self.n_envs):
            row = i // self.grid_cols
            col = i % self.grid_cols
            x = col * self.cell_w
            y = row * self.cell_h

            # Lancer le client en mode fenêtré avec la bonne résolution
            process = subprocess.Popen(
                [
                    self.tw_binary,
                    # Mode fenêtré
                    "gfx_fullscreen", "0",
                    "gfx_borderless", "0",
                    # Résolution de la cellule
                    "gfx_screen_width", str(self.cell_w),
                    "gfx_screen_height", str(self.cell_h),
                    # Connexion auto au serveur
                    f"connect {self.server_ip}:{self.server_port}",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self.client_processes.append(process)
            logger.info(f"Client {i} lancé (PID={process.pid})")

            # Petit délai pour laisser la fenêtre apparaître
            time.sleep(connect_delay)

            # Trouver la fenêtre et la positionner
            window_id = self._find_newest_tw_window()
            if window_id:
                self.client_window_ids.append(window_id)
                self._position_window(window_id, x, y, self.cell_w, self.cell_h)
                logger.info(
                    f"Client {i}: fenêtre {window_id} → "
                    f"pos=({x},{y}) size={self.cell_w}x{self.cell_h}"
                )
            else:
                logger.error(f"Client {i}: fenêtre non trouvée!")
                self.client_window_ids.append(None)

    def _find_newest_tw_window(self) -> Optional[str]:
        """Trouve la fenêtre TW la plus récente non encore assignée."""
        try:
            result = subprocess.run(
                ["xdotool", "search", "--name", "Teeworlds"],
                capture_output=True, text=True, timeout=5
            )
            all_windows = [w.strip() for w in result.stdout.strip().split("\n") if w.strip()]

            # Trouver une fenêtre pas encore assignée
            for wid in reversed(all_windows):  # les plus récentes en dernier
                if wid not in self.client_window_ids:
                    return wid

            return None
        except Exception as e:
            logger.error(f"Erreur recherche fenêtre: {e}")
            return None

    def _position_window(self, window_id: str, x: int, y: int, w: int, h: int):
        """Positionne et redimensionne une fenêtre."""
        if not window_id:
            return
        try:
            # Dé-maximiser / dé-fullscreen d'abord
            subprocess.run(
                ["wmctrl", "-i", "-r", window_id, "-b", "remove,maximized_vert,maximized_horz,fullscreen"],
                capture_output=True, timeout=5
            )
            time.sleep(0.3)

            # Redimensionner puis positionner
            subprocess.run(
                ["xdotool", "windowsize", "--sync", window_id, str(w), str(h)],
                capture_output=True, timeout=5
            )
            subprocess.run(
                ["xdotool", "windowmove", "--sync", window_id, str(x), str(y)],
                capture_output=True, timeout=5
            )
        except Exception as e:
            logger.error(f"Erreur positionnement fenêtre {window_id}: {e}")

    # ------------------------------------------------------------------
    # Création des environnements
    # ------------------------------------------------------------------

    def create_envs(self):
        """Crée un TeeWorldsEnv par client, chacun ciblant sa fenêtre."""
        logger.info("Création des environnements...")

        for i, window_id in enumerate(self.client_window_ids):
            if window_id is None:
                logger.warning(f"Env {i} ignoré: pas de fenêtre")
                continue

            row = i // self.grid_cols
            col = i % self.grid_cols
            x = col * self.cell_w
            y = row * self.cell_h

            # Créer une config spécifique à cet env
            env_config = self._make_env_config(i, x, y)

            env = TeeWorldsEnv(env_config)

            # Injecter le window_id directement au lieu de chercher
            env.controller.window_id = window_id
            env.controller.win_x = x
            env.controller.win_y = y
            env.controller.win_w = self.cell_w
            env.controller.win_h = self.cell_h
            env.controller.screen_center = (x + self.cell_w // 2, y + self.cell_h // 2)

            # Configurer la capture pour cette cellule
            env.capture.monitor = {
                "top": y,
                "left": x,
                "width": self.cell_w,
                "height": self.cell_h,
            }
            env.capture.start()

            # Connexion econ (partagée ou séparée selon le setup)
            if not env.econ.connect():
                logger.error(f"Env {i}: échec connexion econ")
                continue

            self.envs.append(env)
            logger.info(f"Env {i} créé: fenêtre={window_id} capture=({x},{y},{self.cell_w},{self.cell_h})")

        logger.info(f"{len(self.envs)}/{self.n_envs} environnements prêts")

    def _make_env_config(self, env_index: int, x: int, y: int) -> dict:
        """Crée une config dédiée pour un env (copie profonde + ajustements)."""
        import copy
        cfg = copy.deepcopy(self.config)

        cfg["capture"]["monitor"] = {
            "top": y,
            "left": x,
            "width": self.cell_w,
            "height": self.cell_h,
        }

        return cfg

    # ------------------------------------------------------------------
    # Interface parallèle
    # ------------------------------------------------------------------

    def reset_all(self) -> list[dict]:
        """Reset tous les envs en parallèle."""
        def _reset(env):
            return env.reset()

        futures = [self.executor.submit(_reset, env) for env in self.envs]
        results = [f.result() for f in futures]
        return [obs for obs, info in results]

    def step_all(self, actions):
        results = []
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            logger.debug(f"Stepping env {i}...")
            results.append(env.step(action))
            logger.debug(f"Env {i} done")
        return results
            
    # ------------------------------------------------------------------
    # Nettoyage
    # ------------------------------------------------------------------

    def close(self):
        """Ferme tout : envs, clients, threads."""
        logger.info("Fermeture du MultiEnvManager...")

        # Fermer les envs
        for env in self.envs:
            try:
                env.close()
            except Exception:
                pass

        # Tuer les clients TW
        for proc in self.client_processes:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                proc.kill()

        self.executor.shutdown(wait=False)
        logger.info("Tout fermé")