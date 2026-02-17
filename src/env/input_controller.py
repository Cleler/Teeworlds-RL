"""
Injection d'inputs clavier/souris dans le client Teeworlds via pyautogui.

Gère le maintien des touches (keyDown/keyUp) et le déplacement de la souris
pour la visée.
"""

import subprocess
import math
import logging
from typing import Set

logger = logging.getLogger(__name__)

class InputController:
    def __init__(self, key_mapping: dict, aim_radius: int = 300,
                 screen_center: tuple[int, int] = (400, 300)):
        """
        Args:
            key_mapping: Mapping des actions vers les touches.
                         Ex: {"left": "a", "right": "d", "jump": "space", "hook": "shift"}
            aim_radius: Rayon en pixels pour la visée autour du centre.
            screen_center: Centre de la fenêtre de jeu (x, y) en pixels absolus.
        """
        self.keys = key_mapping
        self.aim_radius = aim_radius
        self.screen_center = screen_center
        self.held_keys: Set[str] = set()
        self.window_id = None

    def set_screen_center(self, center_x: int, center_y: int):
        """Met à jour le centre de l'écran (si la fenêtre bouge)."""
        self.screen_center = (center_x, center_y)

    # ------------------------------------------------------------------
    # Actions de haut niveau
    # ------------------------------------------------------------------

    def apply_action(self, direction: int, jump: int, fire: int,
                     hook: int, aim_x: float, aim_y: float):
        """
        Applique une action complète.

        Args:
            direction: 0=gauche, 1=neutre, 2=droite
            jump: 0 ou 1
            fire: 0 ou 1
            hook: 0 ou 1
            aim_x: direction de visée X, entre -1.0 et 1.0
            aim_y: direction de visée Y, entre -1.0 et 1.0
        """
        # Direction
        self._update_direction(direction)

        # Jump
        if jump:
            self._tap_key(self.keys["jump"])

        # Fire (clic souris)
        if fire:
            self._tap_key(self.keys["fire"])
            
        self._update_hold(self.keys["hook"], bool(hook))

        # Visée
        self._move_aim(aim_x, aim_y)

    def _update_direction(self, direction: int):
        """Gère les touches de déplacement gauche/droite."""
        key_left = self.keys["left"]
        key_right = self.keys["right"]

        if direction == 0:  # gauche
            self._hold_key(key_left)
            self._release_key(key_right)
        elif direction == 2:  # droite
            self._release_key(key_left)
            self._hold_key(key_right)
        else:  # neutre
            self._release_key(key_left)
            self._release_key(key_right)

    def _move_aim(self, aim_x: float, aim_y: float):
        """Déplace la souris pour viser dans une direction."""
        target_x = self.screen_center[0] + int(aim_x * self.aim_radius)
        target_y = self.screen_center[1] + int(aim_y * self.aim_radius)
        # pyautogui.moveTo(target_x, target_y, _pause=False)
        if self.window_id:
            # xdotool mousemove --window cible la position relative à la fenêtre
            subprocess.run(
                ["xdotool", "mousemove", "--window", str(self.window_id), str(target_x), str(target_y)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
    
    # ------------------------------------------------------------------
    # Gestion des touches
    # ------------------------------------------------------------------

    def _get_mouse_button(self, key: str) -> str:
        """Convertit le nom de la touche en ID de bouton xdotool."""
        if key == "mouse_left": return "1"
        if key == "mouse_right": return "3"
        return "1"
    
    def _hold_key(self, key: str):
        """Maintient une touche ou un bouton de souris enfoncé."""
        if key not in self.held_keys:
            if self.window_id:
                if key.startswith("mouse_"):
                    btn = self._get_mouse_button(key)
                    subprocess.run(["xdotool", "mousedown", "--window", str(self.window_id), btn], stdout=subprocess.DEVNULL)
                else:
                    subprocess.run(["xdotool", "keydown", "--window", str(self.window_id), key], stdout=subprocess.DEVNULL)
            self.held_keys.add(key)

    def _release_key(self, key: str):
        """Relâche une touche ou un bouton de souris."""
        if key in self.held_keys:
            if self.window_id:
                if key.startswith("mouse_"):
                    btn = self._get_mouse_button(key)
                    subprocess.run(["xdotool", "mouseup", "--window", str(self.window_id), btn], stdout=subprocess.DEVNULL)
                else:
                    subprocess.run(["xdotool", "keyup", "--window", str(self.window_id), key], stdout=subprocess.DEVNULL)
            self.held_keys.discard(key)

    def _tap_key(self, key: str):
        """Appuie et relâche un petit coup."""
        if self.window_id:
            if key.startswith("mouse_"):
                btn = self._get_mouse_button(key)
                subprocess.run(["xdotool", "click", "--window", str(self.window_id), btn], stdout=subprocess.DEVNULL)
            else:
                subprocess.run(["xdotool", "key", "--window", str(self.window_id), key], stdout=subprocess.DEVNULL)

    def _update_hold(self, key: str, pressed: bool):
        """Met à jour l'état maintenu d'une touche/souris."""
        if pressed:
            self._hold_key(key)
        else:
            self._release_key(key)

    def release_all(self):
        """Relâche tout (sécurité quand l'agent meurt)."""
        for key in list(self.held_keys):
            self._release_key(key)
        self.held_keys.clear()

    # ------------------------------------------------------------------
    # Utilitaires
    # ------------------------------------------------------------------

    @staticmethod
    def aim_angle_to_xy(angle_rad: float) -> tuple[float, float]:
        """Convertit un angle en coordonnées (x, y) normalisées."""
        return math.cos(angle_rad), math.sin(angle_rad)
