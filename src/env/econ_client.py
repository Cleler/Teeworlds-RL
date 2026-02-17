"""
Client econ (external console) pour communiquer avec le serveur Teeworlds.

Le protocole econ est un simple TCP text-based :
- Connexion TCP au port econ
- Envoi du password suivi de \n
- Envoi de commandes texte suivies de \n
- Réception de réponses texte ligne par ligne

Commandes utiles :
- "status"       → liste des joueurs connectés
- "say <msg>"    → message chat
- "kick <id>"    → kick un joueur
- "restart"      → restart la map
- "shutdown"     → éteindre le serveur
"""

import socket
import time
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class EconClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 8303,
                 password: str = "", read_timeout: float = 0.05):
        self.host = host
        self.port = port
        self.password = password
        self.read_timeout = read_timeout
        self.sock: Optional[socket.socket] = None

        # État du jeu (mis à jour à chaque poll)
        self.player_position = (0.0, 0.0)
        self.player_health = 10
        self.player_alive = True
        self.kills = 0
        self.deaths = 0
        self.prev_kills = 0
        self.prev_deaths = 0

    # ------------------------------------------------------------------
    # Connexion
    # ------------------------------------------------------------------

    def connect(self):
        """Se connecte au serveur econ et s'authentifie."""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(5.0)
            self.sock.connect((self.host, self.port))
            logger.info(f"Connecté à econ {self.host}:{self.port}")

            # Lire le message de bienvenue
            welcome = self._recv()
            logger.debug(f"Econ welcome: {welcome}")

            # Envoyer le password
            self._send(self.password)
            response = self._recv()
            if "authentication successful" in response.lower():
                logger.info("Authentification econ réussie")
            else:
                logger.warning(f"Réponse auth inattendue: {response}")

            # Passer en mode non-bloquant pour les lectures régulières
            self.sock.settimeout(self.read_timeout)
            return True

        except Exception as e:
            logger.error(f"Erreur connexion econ: {e}")
            return False

    def disconnect(self):
        """Ferme la connexion econ."""
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None
            logger.info("Déconnecté de econ")

    def reconnect(self, delay: float = 2.0):
        """Reconnexion avec délai."""
        self.disconnect()
        time.sleep(delay)
        return self.connect()

    # ------------------------------------------------------------------
    # Communication bas niveau
    # ------------------------------------------------------------------

    def _send(self, msg: str):
        """Envoie une commande au serveur."""
        if not self.sock:
            raise ConnectionError("Non connecté à econ")
        self.sock.sendall((msg + "\n").encode("utf-8"))

    def _recv(self) -> str:
        """Lit les données disponibles."""
        if not self.sock:
            return ""
        try:
            data = self.sock.recv(4096)
            return data.decode("utf-8", errors="replace")
        except socket.timeout:
            return ""
        except Exception as e:
            logger.error(f"Erreur réception econ: {e}")
            return ""

    def send_command(self, cmd: str) -> str:
        """Envoie une commande et retourne la réponse."""
        self._send(cmd)
        time.sleep(self.read_timeout)
        return self._recv()

    # ------------------------------------------------------------------
    # Lecture de l'état du jeu
    # ------------------------------------------------------------------

    def poll(self):
        """
        Lit les messages en attente et met à jour l'état interne.
        Appeler à chaque step de l'environnement.
        """
        data = self._recv()
        if data:
            self._parse_server_output(data)

    def _parse_server_output(self, data: str):
        """
        Parse la sortie serveur pour extraire les événements de jeu.

        Le serveur TW envoie des lignes comme :
        - "[game]: kill killer='0:joueur1' victim='1:joueur2' weapon=0"
        - "[game]: team_join player='0:nom' team=0"
        - Sorties de "status" avec les infos joueurs

        NOTE: Le format exact dépend de ta version de TW et de la config
        ec_output_level. Tu devras adapter les regex ci-dessous.
        """
        for line in data.strip().split("\n"):
            line = line.strip()
            if not line:
                continue

            logger.debug(f"econ: {line}")

            # Détecter un kill
            kill_match = re.search(r"kill.*killer='(\d+):.*victim='(\d+):", line)
            if kill_match:
                killer_id = int(kill_match.group(1))
                victim_id = int(kill_match.group(2))
                # On suppose que notre agent est le joueur 0
                if killer_id == 0:
                    self.kills += 1
                if victim_id == 0:
                    self.deaths += 1
                    self.player_alive = False

            # Détecter un respawn
            if "spawn" in line.lower() and "'0:" in line:
                self.player_alive = True

    def request_status(self) -> str:
        """Demande le status et retourne la réponse brute."""
        return self.send_command("status")

    def get_position(self) -> tuple[float, float]:
        """
        Retourne la position du joueur.

        NOTE IMPORTANTE: La commande "status" standard de TW ne retourne
        pas la position. Pour avoir la position, tu as 2 options :

        1. Utiliser un mod serveur qui expose la position
           (ex: ajouter une commande custom "player_pos <id>")

        2. Utiliser un serveur DDNet qui a des commandes étendues

        Pour l'instant on retourne la dernière position connue.
        Adapter _parse_server_output() selon ton serveur.
        """
        return self.player_position

    def get_score(self) -> tuple[int, int]:
        """Retourne (kills depuis dernier appel, deaths depuis dernier appel)."""
        k = self.kills - self.prev_kills
        d = self.deaths - self.prev_deaths
        self.prev_kills = self.kills
        self.prev_deaths = self.deaths
        return k, d

    def is_player_dead(self) -> bool:
        """Vérifie si le joueur est mort."""
        return not self.player_alive

    # ------------------------------------------------------------------
    # Commandes de contrôle
    # ------------------------------------------------------------------

    def restart_round(self):
        """Restart la map/round."""
        self.send_command("restart")
        self.player_alive = True
        self.kills = 0
        self.deaths = 0
        self.prev_kills = 0
        self.prev_deaths = 0

    def say(self, message: str):
        """Envoie un message dans le chat."""
        self.send_command(f'say "{message}"')

    def set_player_name(self, player_id: int, name: str):
        """Change le nom d'un joueur (admin)."""
        self.send_command(f'rename {player_id} "{name}"')
