"""
Capture d'écran du client Teeworlds.

Supporte :
- X11 : via mss (rapide, ~60+ FPS)
- Wayland : via grim (wlroots/Sway) ou gnome-screenshot (GNOME)
  ⚠️ Plus lent (~5-15 FPS), acceptable pour débuter mais
  recommande de switcher sur X11 pour l'entraînement RL.
"""

import os
import subprocess
import tempfile
import numpy as np
import cv2
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ScreenCapture:
    def __init__(self, monitor: dict, obs_width: int = 84,
                 obs_height: int = 84, grayscale: bool = True):
        self.monitor = monitor
        self.obs_width = obs_width
        self.obs_height = obs_height
        self.grayscale = grayscale

        # Backend sera choisi au start()
        self._backend = None  # "mss", "grim", "gnome-screenshot"
        self._sct = None      # mss instance (X11)
        self._tmp_path = None  # fichier temp (Wayland)

    def start(self):
        """Détecte le backend et initialise."""
        session_type = os.environ.get("XDG_SESSION_TYPE", "").lower()

        if session_type != "wayland":
            # X11 — utiliser mss (rapide)
            try:
                import mss
                self._sct = mss.mss()
                self._backend = "mss"
                logger.info(f"ScreenCapture: backend mss (X11) | {self.monitor}")
                return
            except Exception as e:
                logger.warning(f"mss échoué: {e}, fallback Wayland")

        # Wayland — chercher un outil de capture
        if self._check_command("grim"):
            self._backend = "grim"
            logger.info("ScreenCapture: backend grim (Wayland wlroots)")
        elif self._check_command("gnome-screenshot"):
            self._backend = "gnome-screenshot"
            logger.info("ScreenCapture: backend gnome-screenshot (Wayland GNOME)")
        else:
            raise RuntimeError(
                "Aucun backend de capture disponible.\n"
                "Wayland détecté. Installe grim (sway/wlroots) ou gnome-screenshot (GNOME),\n"
                "ou switch sur une session X11 au login (recommandé pour RL).\n"
                "  sudo apt install grim        # wlroots\n"
                "  sudo apt install gnome-screenshot  # GNOME"
            )

        # Fichier temporaire pour les captures Wayland
        self._tmp_path = os.path.join(tempfile.gettempdir(), "tw_rl_capture.png")

    def stop(self):
        """Ferme le backend."""
        if self._sct:
            self._sct.close()
            self._sct = None
        # Fermer les instances thread-local
        if hasattr(self, '_thread_sct'):
            for sct in self._thread_sct.values():
                try:
                    sct.close()
                except Exception:
                    pass
            self._thread_sct.clear()
        if self._tmp_path and os.path.exists(self._tmp_path):
            os.remove(self._tmp_path)

    def grab(self) -> np.ndarray:
        """
        Capture l'écran et retourne l'observation.

        Returns:
            np.ndarray: (obs_height, obs_width, 1) si grayscale
                        (obs_height, obs_width, 3) si RGB. dtype=uint8.
        """
        if self._backend == "mss":
            return self._grab_mss()
        elif self._backend == "grim":
            return self._grab_grim()
        elif self._backend == "gnome-screenshot":
            return self._grab_gnome()
        else:
            raise RuntimeError("ScreenCapture non démarré (appeler start())")

    # ------------------------------------------------------------------
    # Backend: mss (X11)
    # ------------------------------------------------------------------

    def _grab_mss(self) -> np.ndarray:
        import threading
        # mss utilise un handle X11 thread-local.
        # Si on est dans un thread secondaire, créer une instance dédiée.
        if not hasattr(self, '_thread_sct'):
            self._thread_sct = {}

        tid = threading.current_thread().ident
        if tid not in self._thread_sct:
            import mss as mss_module
            self._thread_sct[tid] = mss_module.mss()

        sct = self._thread_sct[tid]
        raw = sct.grab(self.monitor)
        frame = np.array(raw, dtype=np.uint8)[:, :, :3]  # BGRA → BGR
        return self._process(frame)

    # ------------------------------------------------------------------
    # Backend: grim (Wayland wlroots / Sway)
    # ------------------------------------------------------------------

    def _grab_grim(self) -> np.ndarray:
        m = self.monitor
        geometry = f"{m['left']},{m['top']} {m['width']}x{m['height']}"

        result = subprocess.run(
            ["grim", "-g", geometry, "-t", "ppm", "-"],
            capture_output=True, timeout=5
        )

        if result.returncode != 0:
            raise RuntimeError(f"grim échoué: {result.stderr.decode()}")

        # Décoder PPM depuis stdout (plus rapide que passer par un fichier PNG)
        frame = cv2.imdecode(
            np.frombuffer(result.stdout, dtype=np.uint8),
            cv2.IMREAD_COLOR
        )
        return self._process(frame)

    # ------------------------------------------------------------------
    # Backend: gnome-screenshot (Wayland GNOME)
    # ------------------------------------------------------------------

    def _grab_gnome(self) -> np.ndarray:
        m = self.monitor

        subprocess.run(
            ["gnome-screenshot", "-f", self._tmp_path, "-w"],
            capture_output=True, timeout=5
        )

        frame = cv2.imread(self._tmp_path)
        if frame is None:
            raise RuntimeError("gnome-screenshot: impossible de lire la capture")

        # Crop à la zone demandée
        frame = frame[
            m["top"]:m["top"] + m["height"],
            m["left"]:m["left"] + m["width"]
        ]
        return self._process(frame)

    # ------------------------------------------------------------------
    # Post-traitement commun
    # ------------------------------------------------------------------

    def _process(self, frame: np.ndarray) -> np.ndarray:
        """Resize + grayscale."""
        frame = cv2.resize(frame, (self.obs_width, self.obs_height),
                           interpolation=cv2.INTER_AREA)
        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = frame.reshape(self.obs_height, self.obs_width, 1)
        return frame

    def grab_raw(self) -> np.ndarray:
        """Capture brute sans transformation (debug)."""
        if self._backend == "mss":
            raw = self._sct.grab(self.monitor)
            return np.array(raw, dtype=np.uint8)[:, :, :3]
        else:
            # Pour Wayland, passer par grab() sans resize
            old_w, old_h, old_g = self.obs_width, self.obs_height, self.grayscale
            self.obs_width = self.monitor["width"]
            self.obs_height = self.monitor["height"]
            self.grayscale = False
            frame = self.grab()
            self.obs_width, self.obs_height, self.grayscale = old_w, old_h, old_g
            return frame

    # ------------------------------------------------------------------
    # Utilitaires
    # ------------------------------------------------------------------

    @staticmethod
    def _check_command(cmd: str) -> bool:
        """Vérifie si une commande est disponible."""
        try:
            subprocess.run(["which", cmd], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False