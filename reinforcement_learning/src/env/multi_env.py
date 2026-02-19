"""
Gestionnaire multi-environnement pour Teeworlds RL.

Lance N clients Teeworlds, les positionne en grille sur l'écran,
et crée un environnement Gymnasium par client. Chaque env a sa propre
capture d'écran et ses propres inputs ciblés via xdotool.

Chaque Xvfb virtuel est automatiquement associé à un x11vnc + websockify,
ce qui permet de visualiser les agents en direct via noVNC sans script externe.

Ports par agent i :
    Xvfb     : DISPLAY :100+i
    VNC      : 5900+i  (x11vnc)
    WebSocket: 6080+i  (websockify → noVNC)

Workflow :
    1. Lancer le(s) serveur(s) TW
    2. Lancer le MultiEnvManager qui ouvre N clients
    3. Chaque client rejoint le serveur automatiquement
    4. L'agent RL step() sur tous les envs en parallèle
    5. Dashboard noVNC accessible sur http://<host>:8080

Prérequis :
    sudo apt install x11vnc websockify
"""

import subprocess
import time
import math
import logging
import os
import threading
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

from reinforcement_learning.src.env.teeworlds_env import TeeWorldsEnv
from reinforcement_learning.src.env.econ_client import EconClient

logger = logging.getLogger(__name__)

# ── Ports de base ─────────────────────────────────────────────────────────────
XVFB_BASE_DISPLAY = 100   # DISPLAY :100, :101, ...
VNC_BASE_PORT     = 5900  # VNC     5900, 5901, ...
WS_BASE_PORT      = 6080  # WS      6080, 6081, ... (noVNC)


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

        # Processus VNC / WS (un par agent)
        self._vnc_procs:  list[subprocess.Popen] = []
        self._ws_procs:   list[subprocess.Popen] = []
        self.display_ids: list[int] = []

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers VNC / Websockify
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _check_command(cmd: str) -> bool:
        """Vérifie si un binaire est disponible dans le PATH."""
        try:
            subprocess.run(["which", cmd], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def _launch_vnc_for(self, agent_id: int, display_id: int) -> bool:
        """
        Lance x11vnc + websockify pour l'agent donné.

        x11vnc écoute sur localhost:VNC_BASE_PORT+agent_id
        websockify expose sur 0.0.0.0:WS_BASE_PORT+agent_id → localhost:vnc_port

        Returns:
            True si les deux processus ont démarré, False sinon.
        """
        vnc_port = VNC_BASE_PORT + agent_id
        ws_port  = WS_BASE_PORT  + agent_id

        # ── x11vnc ────────────────────────────────────────────────────────────
        if not self._check_command("x11vnc"):
            logger.warning(
                f"[Agent {agent_id}] x11vnc introuvable — streaming VNC désactivé. "
                "Installez-le avec : sudo apt install x11vnc"
            )
            self._vnc_procs.append(None)
            self._ws_procs.append(None)
            return False

        vnc_proc = subprocess.Popen(
            [
                "x11vnc",
                "-display", f":{display_id}",
                "-rfbport", str(vnc_port),
                "-listen", "localhost",   # n'exposer VNC que localement
                "-nopw",                  # pas de mot de passe VNC
                "-xkb",                   # meilleure gestion du clavier
                "-forever",               # ne pas quitter après la première déconnexion
                "-quiet",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._vnc_procs.append(vnc_proc)
        logger.info(
            f"[Agent {agent_id}] x11vnc démarré — "
            f"DISPLAY=:{display_id} → localhost:{vnc_port} (PID={vnc_proc.pid})"
        )

        # Petit délai pour que x11vnc soit prêt avant websockify
        time.sleep(0.5)

        # ── websockify ─────────────────────────────────────────────────────────
        if not self._check_command("websockify"):
            logger.warning(
                f"[Agent {agent_id}] websockify introuvable — streaming WebSocket désactivé. "
                "Installez-le avec : sudo apt install websockify"
            )
            self._ws_procs.append(None)
            return False

        ws_proc = subprocess.Popen(
            [
                "websockify",
                f"0.0.0.0:{ws_port}",         # écoute WebSocket (exposé)
                f"localhost:{vnc_port}",        # cible VNC (local)
                "-D",                          # mode daemon (non-bloquant)
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._ws_procs.append(ws_proc)
        logger.info(
            f"[Agent {agent_id}] websockify démarré — "
            f"0.0.0.0:{ws_port} → localhost:{vnc_port} (PID={ws_proc.pid})"
        )

        return True

    def _log_stream_summary(self):
        """Affiche un récapitulatif des ports WebSocket disponibles."""
        lines = ["", "━" * 60, "  📺  Flux VNC disponibles", "━" * 60]
        for i in range(self.n_envs):
            ws_port = WS_BASE_PORT + i
            lines.append(f"  Agent {i:2d} → ws://<host>:{ws_port}  (noVNC port)")
        lines += [
            "━" * 60,
            "  Dashboard : http://<host>:8080/dashboard.html",
            "━" * 60,
            "",
        ]
        logger.info("\n".join(lines))

    # ──────────────────────────────────────────────────────────────────────────
    # Lancement des clients
    # ──────────────────────────────────────────────────────────────────────────

    def launch_clients(self, connect_delay: float = 2.0):
        """
        Lance N clients TW, chacun sur son propre Xvfb.
        Pour chaque Xvfb, lance également x11vnc + websockify.
        """
        logger.info(f"Lancement de {self.n_envs} clients Teeworlds...")

        def _log_hook(proc, agent_id):
            for line in proc.stderr:
                logger.info(f"[hook/{agent_id}] {line.decode().rstrip()}")

        for i in range(self.n_envs):
            display_id = XVFB_BASE_DISPLAY + i
            self.display_ids.append(display_id)

            # ── 1. Xvfb ───────────────────────────────────────────────────────
            xvfb_proc = subprocess.Popen(
                [
                    "Xvfb",
                    f":{display_id}",
                    "-screen", "0", f"{self.cell_w}x{self.cell_h}x24",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self.client_processes.append(xvfb_proc)
            logger.info(
                f"[Agent {i}] Xvfb démarré — "
                f"DISPLAY=:{display_id} {self.cell_w}x{self.cell_h}x24 (PID={xvfb_proc.pid})"
            )
            time.sleep(0.5)  # Laisser l'écran virtuel s'initialiser

            # ── 2. x11vnc + websockify pour cet écran ─────────────────────────
            self._launch_vnc_for(i, display_id)

            # ── 3. Client Teeworlds sur l'écran virtuel ────────────────────────
            env_vars = os.environ.copy()
            env_vars["DISPLAY"] = f":{display_id}"

            hook_path = os.path.join(
                os.path.dirname(__file__),  # src/env/
                "hook", "tw_input_hook.so"
            )
            env_vars["TW_AGENT_ID"] = str(i)
            env_vars["LD_PRELOAD"]  = hook_path

            tw_proc = subprocess.Popen(
                [
                    self.tw_binary,
                    "gfx_fullscreen 1",
                    f"gfx_screen_width {self.cell_w}",
                    f"gfx_screen_height {self.cell_h}",
                    f"player_name Bot_{i}",
                    f"connect {self.server_ip}:{self.server_port}",
                ],
                env=env_vars,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            self.client_processes.append(tw_proc)
            threading.Thread(
                target=_log_hook, args=(tw_proc, i), daemon=True
            ).start()
            logger.info(
                f"[Agent {i}] Client TW lancé — "
                f"DISPLAY=:{display_id} (PID={tw_proc.pid})"
            )

            time.sleep(connect_delay)

        self._log_stream_summary()

    # ──────────────────────────────────────────────────────────────────────────
    # Création des environnements
    # ──────────────────────────────────────────────────────────────────────────

    def create_envs(self):
        """Crée un TeeWorldsEnv par client, chacun ciblant son Xvfb."""
        logger.info("Création des environnements...")

        self.shared_econ = EconClient(
            host=self.server_ip,
            port=self.config["server"]["econ_port"],
            password=self.config["server"]["econ_password"],
            read_timeout=self.config["server"]["read_timeout"],
        )
        if not self.shared_econ.connect():
            logger.error("Échec de connexion du EconClient partagé !")
            return

        for i, display_id in enumerate(self.display_ids):
            env_config = self._make_env_config(i)
            env = TeeWorldsEnv(env_config, shared_econ=self.shared_econ, agent_id=i)

            env.controller.display_id = display_id
            env.controller.connect_display(display_id)
            env.controller.screen_center = (self.cell_w // 2, self.cell_h // 2)
            env.capture.display_id = display_id

            env.setup()

            self.envs.append(env)
            logger.info(f"[Agent {i}] Environnement prêt (DISPLAY=:{display_id})")

        logger.info(f"{len(self.envs)}/{self.n_envs} environnements prêts")

    def _make_env_config(self, env_index: int) -> dict:
        """Crée une config dédiée pour un env (copie profonde + ajustements)."""
        import copy
        cfg = copy.deepcopy(self.config)
        cfg["capture"]["monitor"] = {
            "top": 0,
            "left": 0,
            "width": self.cell_w,
            "height": self.cell_h,
        }
        return cfg

    # ──────────────────────────────────────────────────────────────────────────
    # Interface parallèle
    # ──────────────────────────────────────────────────────────────────────────

    def reset_all(self) -> list[dict]:
        """Reset tous les envs en parallèle."""
        if hasattr(self, "shared_econ"):
            self.shared_econ.restart_round()
            self.shared_econ.poll()

        def _reset(env):
            obs, info = env.reset()
            logger.debug(
                f"[RESET] env={id(env)} "
                f"pos={obs['position']} "
                f"img={obs['image'].shape} "
                f"min/max={obs['image'].min()}/{obs['image'].max()}"
            )
            return obs, info

        futures = [self.executor.submit(_reset, env) for env in self.envs]
        results = [f.result() for f in futures]
        return [obs for obs, info in results]

    def step_all(self, actions) -> list:
        """Step tous les envs en parallèle puis poll econ."""
        def _step(args):
            env, action = args
            return env.step(action)

        futures = [
            self.executor.submit(_step, (env, act))
            for env, act in zip(self.envs, actions)
        ]
        results = [f.result() for f in futures]

        if hasattr(self, "shared_econ"):
            self.shared_econ.poll()

        return results

    # ──────────────────────────────────────────────────────────────────────────
    # Nettoyage
    # ──────────────────────────────────────────────────────────────────────────

    def close(self):
        """Ferme tout : envs, clients TW, x11vnc, websockify, Xvfb."""
        logger.info("Fermeture du MultiEnvManager...")

        # Fermer les envs Gymnasium
        for env in self.envs:
            try:
                env.close()
            except Exception:
                pass

        # Arrêter websockify
        for proc in self._ws_procs:
            if proc is not None:
                try:
                    proc.terminate()
                    proc.wait(timeout=3)
                except Exception:
                    proc.kill()

        # Arrêter x11vnc
        for proc in self._vnc_procs:
            if proc is not None:
                try:
                    proc.terminate()
                    proc.wait(timeout=3)
                except Exception:
                    proc.kill()

        # Tuer les clients TW et Xvfb
        for proc in self.client_processes:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                proc.kill()

        # Déconnecter econ partagé
        if hasattr(self, "shared_econ"):
            self.shared_econ.disconnect()

        self.executor.shutdown(wait=False)
        logger.info("Tout fermé")