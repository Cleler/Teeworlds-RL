"""
Capture d'écran du client Teeworlds via mss.

mss est plus rapide que pyautogui.screenshot() car il utilise
directement les API natives de l'OS.
"""

import numpy as np
import cv2
import mss
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ScreenCapture:
    def __init__(self, monitor: dict, obs_width: int = 84,
                 obs_height: int = 84, grayscale: bool = True):
        """
        Args:
            monitor: Zone de capture {"top": 0, "left": 0, "width": 800, "height": 600}
            obs_width: Largeur de l'observation après resize
            obs_height: Hauteur de l'observation après resize
            grayscale: Convertir en niveaux de gris
        """
        self.monitor = monitor
        self.obs_width = obs_width
        self.obs_height = obs_height
        self.grayscale = grayscale
        self.sct: Optional[mss.mss] = None

    def start(self):
        """Initialise mss."""
        self.sct = mss.mss()
        logger.info(f"ScreenCapture initialisé: {self.monitor}")

    def stop(self):
        """Ferme mss."""
        if self.sct:
            self.sct.close()
            self.sct = None

    def grab(self) -> np.ndarray:
        """
        Capture l'écran et retourne l'observation.

        Returns:
            np.ndarray: Image (obs_height, obs_width, 1) si grayscale
                        ou (obs_height, obs_width, 3) si RGB.
                        dtype=np.uint8, valeurs 0-255.
        """
        if not self.sct:
            raise RuntimeError("ScreenCapture non démarré (appeler start())")

        # Capture brute (BGRA)
        raw = self.sct.grab(self.monitor)
        frame = np.array(raw, dtype=np.uint8)

        # Retirer le canal alpha : BGRA → BGR
        frame = frame[:, :, :3]

        # Resize
        frame = cv2.resize(frame, (self.obs_width, self.obs_height),
                           interpolation=cv2.INTER_AREA)

        # Grayscale
        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = frame.reshape(self.obs_height, self.obs_width, 1)

        return frame

    def grab_raw(self) -> np.ndarray:
        """Capture brute sans transformation (pour debug/visualisation)."""
        if not self.sct:
            raise RuntimeError("ScreenCapture non démarré")
        raw = self.sct.grab(self.monitor)
        return np.array(raw, dtype=np.uint8)[:, :, :3]

    @staticmethod
    def find_window(window_title: str = "Teeworlds") -> Optional[dict]:
        """
        Tente de trouver la fenêtre Teeworlds automatiquement.

        NOTE: Fonctionne uniquement sur Linux avec xdotool ou
        sur Windows avec pygetwindow. À adapter selon ton OS.

        Returns:
            dict monitor compatible mss, ou None si non trouvé.
        """
        try:
            import subprocess
            # Linux avec xdotool
            result = subprocess.run(
                ["xdotool", "search", "--name", window_title],
                capture_output=True, text=True
            )
            if result.stdout.strip():
                window_id = result.stdout.strip().split("\n")[0]
                geo = subprocess.run(
                    ["xdotool", "getwindowgeometry", "--shell", window_id],
                    capture_output=True, text=True
                )
                info = {}
                for line in geo.stdout.strip().split("\n"):
                    k, v = line.split("=")
                    info[k] = int(v)

                return {
                    "top": info.get("Y", 0),
                    "left": info.get("X", 0),
                    "width": info.get("WIDTH", 800),
                    "height": info.get("HEIGHT", 600),
                }
        except Exception as e:
            logger.warning(f"Impossible de trouver la fenêtre: {e}")

        return None
