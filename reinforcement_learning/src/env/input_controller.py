# """
# Injection d'inputs clavier/souris dans le client Teeworlds via pyautogui.

# Gère le maintien des touches (keyDown/keyUp) et le déplacement de la souris
# pour la visée.
# """

# import os
# import subprocess
# import math
# import logging
# from typing import Set

# logger = logging.getLogger(__name__)

# class InputController:
#     def __init__(self, key_mapping: dict, aim_radius: int = 300,
#                  screen_center: tuple[int, int] = (400, 300)):
#         """
#         Args:
#             key_mapping: Mapping des actions vers les touches.
#                          Ex: {"left": "a", "right": "d", "jump": "space", "hook": "shift"}
#             aim_radius: Rayon en pixels pour la visée autour du centre.
#             screen_center: Centre de la fenêtre de jeu (x, y) en pixels absolus.
#         """
#         self.keys = key_mapping
#         self.aim_radius = aim_radius
#         self.screen_center = screen_center
#         self.held_keys: Set[str] = set()
#         self.display_id = None
    
#     def _run_xdotool(self, args: list):
#         """Exécute xdotool sur l'écran virtuel dédié à cet agent."""
#         if hasattr(self, 'display_id') and self.display_id is not None:
#             env = os.environ.copy()
#             env["DISPLAY"] = f":{self.display_id}"
#             subprocess.run(["xdotool"] + args, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
#     def set_screen_center(self, center_x: int, center_y: int):
#         """Met à jour le centre de l'écran (si la fenêtre bouge)."""
#         self.screen_center = (center_x, center_y)

#     # ------------------------------------------------------------------
#     # Actions de haut niveau
#     # ------------------------------------------------------------------

#     def apply_action(self, direction: int, jump: int, fire: int,
#                      hook: int, weapon_switch: int, aim_x: float, aim_y: float):
#         print("apply_action")
#         """
#         Applique une action complète.

#         Args:
#             direction: 0=gauche, 1=neutre, 2=droite
#             jump: 0 ou 1
#             fire: 0 ou 1
#             hook: 0 ou 1
#             aim_x: direction de visée X, entre -1.0 et 1.0
#             aim_y: direction de visée Y, entre -1.0 et 1.0
#         """
#         # Direction
#         self._update_direction(direction)

#         # Jump
#         if jump:
#             self._tap_key(self.keys["jump"])

#         # Fire (clic souris)
#         if fire:
#             self._tap_key(self.keys["fire"])
            
#         self._update_hold(self.keys["hook"], bool(hook))

#         # Weapon Switch
#         if weapon_switch == 1:
#             self._tap_key("mouse_scroll_up")    # Arme précédente (molette haut)
#         elif weapon_switch == 2:
#             self._tap_key("mouse_scroll_down")  # Arme suivante (molette bas)        
        
#         # Visée
#         self._move_aim(aim_x, aim_y)

#     def _update_direction(self, direction: int):
#         """Gère les touches de déplacement gauche/droite."""
#         print("update direction")
#         key_left = self.keys["left"]
#         key_right = self.keys["right"]

#         if direction == 0:  # gauche
#             print("direction = left")
#             self._hold_key(key_left)
#             self._release_key(key_right)
#         elif direction == 2:  # droite
#             print("direction = right")
#             self._release_key(key_left)
#             self._hold_key(key_right)
#         else:  # neutre
#             print("direction = none")
#             self._release_key(key_left)
#             self._release_key(key_right)

#     def _move_aim(self, aim_x: float, aim_y: float):
#         """Déplace la souris pour viser dans une direction."""
#         target_x = self.screen_center[0] + int(aim_x * self.aim_radius)
#         target_y = self.screen_center[1] + int(aim_y * self.aim_radius)
#         self._run_xdotool(["mousemove", str(target_x), str(target_y)])
    
#     # ------------------------------------------------------------------
#     # Gestion des touches
#     # ------------------------------------------------------------------

#     def _get_mouse_button(self, key: str) -> str:
#         """Convertit le nom de la touche en ID de bouton xdotool."""
#         if key == "mouse_left": return "1"
#         if key == "mouse_middle": return "2"
#         if key == "mouse_right": return "3"
#         if key == "mouse_scroll_up": return "4"
#         if key == "mouse_scroll_down": return "5"
#         return "1"
        
#     def _hold_key(self, key: str):
#         """Maintient une touche ou un bouton de souris enfoncé."""
#         if key not in self.held_keys:
#             if key.startswith("mouse_"):
#                 self._run_xdotool(["mousedown", self._get_mouse_button(key)])
#             else:
#                 self._run_xdotool(["keydown", key])
#             self.held_keys.add(key)

#     def _release_key(self, key: str):
#         """Relâche une touche ou un bouton de souris."""
#         if key in self.held_keys:
#             if key.startswith("mouse_"):
#                 btn = self._get_mouse_button(key)
#                 self._run_xdotool(["mouseup", self._get_mouse_button(key)])
#             else:
#                 self._run_xdotool(["keyup", key])
#             self.held_keys.discard(key)

#     def _tap_key(self, key: str):
#         """Appuie et relâche un petit coup."""
#         if key.startswith("mouse_"):
#             btn = self._get_mouse_button(key)
#             self._run_xdotool(["click", self._get_mouse_button(key)])
#         else:
#             self._run_xdotool(["key", key])

#     def _update_hold(self, key: str, pressed: bool):
#         """Met à jour l'état maintenu d'une touche/souris."""
#         if pressed:
#             self._hold_key(key)
#         else:
#             self._release_key(key)

#     def release_all(self):
#         """Relâche tout (sécurité quand l'agent meurt)."""
#         for key in list(self.held_keys):
#             self._release_key(key)
#         self.held_keys.clear()

#     # ------------------------------------------------------------------
#     # Utilitaires
#     # ------------------------------------------------------------------

#     @staticmethod
#     def aim_angle_to_xy(angle_rad: float) -> tuple[float, float]:
#         """Convertit un angle en coordonnées (x, y) normalisées."""
#         return math.cos(angle_rad), math.sin(angle_rad)


# """
# Injection d'inputs native via python-xlib.
# Remplace xdotool par des appels directs au serveur X11 en RAM.
# Zéro sous-processus bash, latence nulle, aucun conflit de threads.
# """

# import math
# import logging
# from typing import Set

# try:
#     from Xlib import X, display, XK
#     from Xlib.ext import xtest
# except ImportError:
#     raise ImportError("Veuillez installer python-xlib : pip install python-xlib")

# logger = logging.getLogger(__name__)

# class InputController:
#     def __init__(self, key_mapping: dict, aim_radius: int = 300,
#                  screen_center: tuple[int, int] = (400, 300)):
#         self.keys = key_mapping
#         self.aim_radius = aim_radius
#         self.screen_center = screen_center
#         self.held_keys: Set[str] = set()
        
#         # Variables système X11
#         self.display_id = None
#         self.disp = None
#         self.last_aim = None

#     def connect_display(self):
#         """Initialise la connexion directe au serveur X11 virtuel."""
#         display_name = f":{self.display_id}" if getattr(self, "display_id", None) is not None else None
#         self.disp = display.Display(display_name)

#     def set_screen_center(self, center_x: int, center_y: int):
#         self.screen_center = (center_x, center_y)

#     # ------------------------------------------------------------------
#     # Conversion des touches physiques
#     # ------------------------------------------------------------------
#     def _get_keycode(self, key_char: str):
#         """Convertit les strings ('a', 'space') en code matériel X11."""
#         if key_char == "space":
#             keysym = XK.XK_space
#         else:
#             keysym = XK.string_to_keysym(key_char)
#         return self.disp.keysym_to_keycode(keysym)

#     def _get_mouse_btn(self, key: str) -> int:
#         if key == "mouse_left": return 1
#         if key == "mouse_middle": return 2
#         if key == "mouse_right": return 3
#         if key == "mouse_scroll_up": return 4
#         if key == "mouse_scroll_down": return 5
#         return 1

#     # ------------------------------------------------------------------
#     # Exécution X11 native (xtest)
#     # ------------------------------------------------------------------
#     def _hold_key(self, key: str):
#         if key not in self.held_keys:
#             if key.startswith("mouse_"):
#                 xtest.fake_input(self.disp, X.ButtonPress, self._get_mouse_btn(key))
#             else:
#                 xtest.fake_input(self.disp, X.KeyPress, self._get_keycode(key))
#             self.held_keys.add(key)

#     def _release_key(self, key: str):
#         if key in self.held_keys:
#             if key.startswith("mouse_"):
#                 xtest.fake_input(self.disp, X.ButtonRelease, self._get_mouse_btn(key))
#             else:
#                 xtest.fake_input(self.disp, X.KeyRelease, self._get_keycode(key))
#             self.held_keys.discard(key)

#     def _tap_key(self, key: str):
#         if key.startswith("mouse_"):
#             btn = self._get_mouse_btn(key)
#             xtest.fake_input(self.disp, X.ButtonPress, btn)
#             xtest.fake_input(self.disp, X.ButtonRelease, btn)
#         else:
#             kc = self._get_keycode(key)
#             xtest.fake_input(self.disp, X.KeyPress, kc)
#             xtest.fake_input(self.disp, X.KeyRelease, kc)

#     def _update_hold(self, key: str, pressed: bool):
#         if pressed:
#             self._hold_key(key)
#         else:
#             self._release_key(key)

#     # ------------------------------------------------------------------
#     # Application de l'action du Réseau de Neurones
#     # ------------------------------------------------------------------
#     def apply_action(self, direction: int, jump: int, fire: int,
#                      hook: int, weapon_switch: int, aim_x: float, aim_y: float):
        
#         # Lazy loading de la connexion au display Xvfb
#         if not self.disp:
#             self.connect_display()

#         # 1. GAUCHE / DROITE -> Maintient (Hold)
#         key_left, key_right = self.keys["left"], self.keys["right"]
#         if direction == 0:  # gauche
#             self._hold_key(key_left)
#             self._release_key(key_right)
#         elif direction == 2:  # droite
#             self._release_key(key_left)
#             self._hold_key(key_right)
#         else:
#             self._release_key(key_left)
#             self._release_key(key_right)

#         # 2. HOOK -> Maintient (Hold)
#         self._update_hold(self.keys["hook"], bool(hook))

#         # 3. JUMP & FIRE -> Clic rapide (Tap)
#         # On simule un appui puis un relâchement instantané dans la même frame
#         if jump:
#             self._tap_key(self.keys["jump"])
            
#         if fire:
#             self._tap_key(self.keys["fire"])

#         # 4. CHANGEMENT D'ARME -> Roulette (Tap)
#         if weapon_switch == 1:
#             self._tap_key("mouse_scroll_up")
#         elif weapon_switch == 2:
#             self._tap_key("mouse_scroll_down")        
        
#         # 5. VISÉE DE LA SOURIS
#         target_x = self.screen_center[0] + int(aim_x * self.aim_radius)
#         target_y = self.screen_center[1] + int(aim_y * self.aim_radius)
#         if self.last_aim != (target_x, target_y):
#             # Déplace la souris sans avoir besoin de shell
#             xtest.fake_input(self.disp, X.MotionNotify, x=target_x, y=target_y)
#             self.last_aim = (target_x, target_y)

#         # /!\ CRITIQUE : Dit à X11 de traiter la file d'attente d'un coup
#         self.disp.sync()
        
#     def release_all(self):
#         """Relâche tout pour éviter le syndrome des touches fantômes."""
#         if not self.disp: return
#         for key in list(self.held_keys):
#             self._release_key(key)
#         self.disp.sync()
#         self.held_keys.clear()

#     @staticmethod
#     def aim_angle_to_xy(angle_rad: float) -> tuple[float, float]:
#         return math.cos(angle_rad), math.sin(angle_rad)




"""
Injection d'inputs native via python-xlib.
"""

import math
import logging
from typing import Set

try:
    from Xlib import X, display, XK
    from Xlib.ext import xtest
except ImportError:
    raise ImportError("Veuillez installer python-xlib : pip install python-xlib")

logger = logging.getLogger(__name__)


class InputController:
    def __init__(self, key_mapping: dict, aim_radius: int = 300,
                 screen_center: tuple[int, int] = (400, 300)):
        self.keys = key_mapping
        self.aim_radius = aim_radius
        self.screen_center = screen_center
        self.held_keys: Set[str] = set()

        self.display_id = None
        self.disp = None
        self.last_aim = None
        
        self._pending_release: Set[str] = set()

    def connect_display(self):
        """Initialise la connexion directe au serveur X11 virtuel."""
        display_name = f":{self.display_id}" if getattr(self, "display_id", None) is not None else None
        self.disp = display.Display(display_name)
        self.held_keys.clear()
        self.last_aim = None
        self._pending_release.clear()

    def set_screen_center(self, center_x: int, center_y: int):
        self.screen_center = (center_x, center_y)

    # ------------------------------------------------------------------
    # Conversion des touches
    # ------------------------------------------------------------------

    def _get_keycode(self, key_char: str):
        if key_char == "space":
            keysym = XK.XK_space
        else:
            keysym = XK.string_to_keysym(key_char)
        return self.disp.keysym_to_keycode(keysym)

    def _get_mouse_btn(self, key: str) -> int:
        if key == "mouse_left":        return 1
        if key == "mouse_middle":      return 2
        if key == "mouse_right":       return 3
        if key == "mouse_scroll_up":   return 4
        if key == "mouse_scroll_down": return 5
        return 1

    # ------------------------------------------------------------------
    # Primitives X11
    # ------------------------------------------------------------------

    def _hold_key(self, key: str):
        """Press uniquement si pas déjà maintenu — évite les repeat X11."""
        if key not in self.held_keys:
            if key.startswith("mouse_"):
                xtest.fake_input(self.disp, X.ButtonPress, self._get_mouse_btn(key))
            else:
                xtest.fake_input(self.disp, X.KeyPress, self._get_keycode(key))
            self.held_keys.add(key)

    def _release_key(self, key: str):
        """
        Release TOUJOURS envoyé à X11, que held_keys soit synchronisé ou non.
        Sans ça, si held_keys se vide (reset, exception...) sans que X11 reçoive
        le Release, la touche reste physiquement enfoncée côté Xvfb indéfiniment.
        """
        if key.startswith("mouse_"):
            xtest.fake_input(self.disp, X.ButtonRelease, self._get_mouse_btn(key))
        else:
            xtest.fake_input(self.disp, X.KeyRelease, self._get_keycode(key))
        self.held_keys.discard(key)

    def _tap_key(self, key: str):
        """
        Press + Release immédiat.
        Utilisé pour : fire (1 click = 1 balle), jump (1 press = 1 saut),
        weapon switch (scroll).
        Ces actions sont one-shot : le jeu réagit à l'événement Press lui-même,
        pas à la durée du maintien.
        """
        if key.startswith("mouse_"):
            btn = self._get_mouse_btn(key)
            xtest.fake_input(self.disp, X.ButtonPress, btn)
            xtest.fake_input(self.disp, X.ButtonRelease, btn)
        else:
            kc = self._get_keycode(key)
            xtest.fake_input(self.disp, X.KeyPress, kc)
            xtest.fake_input(self.disp, X.KeyRelease, kc)

    def _update_hold(self, key: str, pressed: bool):
        if pressed:
            self._hold_key(key)
        else:
            self._release_key(key)

    # ------------------------------------------------------------------
    # Application de l'action
    # ------------------------------------------------------------------

    def apply_action(self, direction: int, jump: int, fire: int,
                 hook: int, weapon_switch: int, aim_x: float, aim_y: float):

        if not self.disp:
            self.connect_display()

        # 1. Relâcher les touches one-shot du tick PRÉCÉDENT
        for key in list(self._pending_release):
            self._release_key(key)
        self._pending_release.clear()

        # 2. MOUVEMENT — hold until release
        key_left, key_right = self.keys["left"], self.keys["right"]
        if direction == 0:
            self._hold_key(key_left)
            self._release_key(key_right)
        elif direction == 2:
            self._release_key(key_left)
            self._hold_key(key_right)
        else:
            self._release_key(key_left)
            self._release_key(key_right)

        # 3. JUMP — one-shot : Press maintenant, Release au prochain tick
        if jump:
            self._hold_key(self.keys["jump"])
            self._pending_release.add(self.keys["jump"])

        # 4. FIRE — one-shot : Press maintenant, Release au prochain tick
        if fire:
            self._hold_key(self.keys["fire"])
            self._pending_release.add(self.keys["fire"])

        # 5. HOOK — hold until release
        self._update_hold(self.keys["hook"], bool(hook))

        # 6. WEAPON SWITCH — scroll (ces événements sont edge-triggered, pas level)
        if weapon_switch == 1:
            self._hold_key("mouse_scroll_up")
            self._pending_release.add("mouse_scroll_up")
        elif weapon_switch == 2:
            self._hold_key("mouse_scroll_down")
            self._pending_release.add("mouse_scroll_down")

        # 7. VISÉE
        target_x = self.screen_center[0] + int(aim_x * self.aim_radius)
        target_y = self.screen_center[1] + int(aim_y * self.aim_radius)
        if self.last_aim != (target_x, target_y):
            xtest.fake_input(self.disp, X.MotionNotify, x=target_x, y=target_y)
            self.last_aim = (target_x, target_y)

        self.disp.sync()


    # ------------------------------------------------------------------
    # Nettoyage
    # ------------------------------------------------------------------

    def release_all(self):
        """
        Relâche toutes les touches — appeler à chaque mort/reset.
        On force le Release de toutes les touches connues explicitement
        plutôt que de se fier au contenu de held_keys qui peut être désynchronisé.
        """
        if not self.disp:
            return
        for key in list(self._pending_release):
            self._release_key(key)
        self._pending_release.clear()
        for key in [
            self.keys.get("left"),
            self.keys.get("right"),
            self.keys.get("jump"),
            self.keys.get("fire"),
            self.keys.get("hook"),
        ]:
            if key:
                self._release_key(key)
        self.disp.sync()
        self.held_keys.clear()

    # ------------------------------------------------------------------

    @staticmethod
    def aim_angle_to_xy(angle_rad: float) -> tuple[float, float]:
        return math.cos(angle_rad), math.sin(angle_rad)