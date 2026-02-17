"""
Client econ (external console) pour communiquer avec le serveur Teeworlds 0.7.

Spécificités TW 0.7 :
- Auth : le serveur envoie "Enter password:" puis attend le password + \n
- Les commandes comme "status" affichent leur résultat côté serveur,
  PAS sur le socket econ. On ne peut pas les lire.
- Avec ec_output_level 2, le serveur FORWARD les événements de jeu
  (kills, joins, spawns) sur le socket econ. C'est notre source de données.

On se base donc sur le flux d'événements pour tracker l'état du jeu.
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
        self._recv_buffer = ""

        # État du jeu
        self.player_position = (0.0, 0.0)
        self.player_alive = True
        self.kills = 0
        self.deaths = 0
        self.prev_kills = 0
        self.prev_deaths = 0

    # ------------------------------------------------------------------
    # Connexion
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Se connecte au serveur econ et s'authentifie."""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(5.0)
            self.sock.connect((self.host, self.port))
            logger.info(f"Connecté à econ {self.host}:{self.port}")

            # ---- Attendre le prompt "Enter password:" ----
            prompt = self._recv_until("Enter password:", timeout=5.0)
            if "Enter password:" not in prompt:
                logger.error(f"Prompt inattendu: {prompt!r}")
                return False
            logger.debug("Prompt password reçu")

            # ---- Envoyer le password ----
            self._send(self.password)

            # ---- Vérifier l'authentification ----
            response = self._recv_until("Authentication", timeout=5.0)
            if "Authentication successful" in response:
                logger.info("Authentification econ réussie")
            else:
                logger.error(f"Authentification échouée: {response!r}")
                return False

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

    def reconnect(self, delay: float = 2.0) -> bool:
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
        """Lit les données disponibles (non-bloquant si timeout court)."""
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

    def _recv_until(self, marker: str, timeout: float = 5.0) -> str:
        """Lit jusqu'à trouver un marqueur dans la réponse."""
        if not self.sock:
            return ""
        old_timeout = self.sock.gettimeout()
        self.sock.settimeout(0.5)

        accumulated = ""
        deadline = time.time() + timeout

        while time.time() < deadline:
            try:
                data = self.sock.recv(4096).decode("utf-8", errors="replace")
                accumulated += data
                if marker in accumulated:
                    break
            except socket.timeout:
                continue
            except Exception as e:
                logger.error(f"Erreur recv_until: {e}")
                break

        self.sock.settimeout(old_timeout)
        return accumulated

    def send_command(self, cmd: str):
        """Envoie une commande. Note: la réponse n'est PAS renvoyée sur econ pour la plupart des commandes TW 0.7."""
        self._send(cmd)

    # ------------------------------------------------------------------
    # Lecture de l'état du jeu (via flux d'événements)
    # ------------------------------------------------------------------

    def poll(self):
        """
        Lit les messages en attente et met à jour l'état interne.
        Appeler à chaque step de l'environnement.

        Avec ec_output_level 2, le serveur forward les logs de jeu.
        Format typique des lignes :
            [timestamp][game]: kill killer='0:PlayerName' victim='1:BotName' weapon=5
            [timestamp][game]: join player='0:PlayerName'
            [timestamp][game]: leave player='0:PlayerName'
            [timestamp][game]: start match type='DM' teamplay='0'
        """
        data = self._recv()
        if not data:
            return

        # Accumuler avec le buffer (on peut recevoir des lignes partielles)
        self._recv_buffer += data

        # Traiter les lignes complètes
        while "\n" in self._recv_buffer:
            line, self._recv_buffer = self._recv_buffer.split("\n", 1)
            line = line.strip()
            if line:
                self._parse_line(line)

    def _parse_line(self, line: str):
        """
        Parse une ligne de log du serveur.

        NOTE: Les formats exacts dépendent de ta version de TW.
        Adapte les regex si nécessaire en regardant les logs réels
        de ton serveur quand des événements se produisent.
        """
        logger.debug(f"econ: {line}")

        # --- Kill ---
        # Patterns possibles (à adapter selon ta version) :
        #   [time][game]: kill killer='0:Name' victim='1:Name' weapon=5
        #   [time][game]: kill killer_id=0 victim_id=1 weapon=5
        kill_match = re.search(
            r"kill.*killer[_=]'?(\d+)[:\s].*victim[_=]'?(\d+)[:\s]", line
        )
        if kill_match:
            killer_id = int(kill_match.group(1))
            victim_id = int(kill_match.group(2))
            logger.info(f"Kill détecté: {killer_id} → {victim_id}")
            # On suppose que notre agent est le joueur ID 0
            if killer_id == 0 and victim_id != 0:
                self.kills += 1
            if victim_id == 0:
                self.deaths += 1
                self.player_alive = False
            return

        # --- Format alternatif de kill (plus simple) ---
        # Certaines versions: "0:PlayerA killed 1:PlayerB with 5"
        kill_alt = re.search(r"(\d+):\S+\s+killed\s+(\d+):", line)
        if kill_alt:
            killer_id = int(kill_alt.group(1))
            victim_id = int(kill_alt.group(2))
            logger.info(f"Kill détecté (alt): {killer_id} → {victim_id}")
            if killer_id == 0 and victim_id != 0:
                self.kills += 1
            if victim_id == 0:
                self.deaths += 1
                self.player_alive = False
            return

        # --- Spawn / Respawn ---
        if re.search(r"spawn.*'?0:", line, re.IGNORECASE):
            self.player_alive = True

        # --- Match start (après restart) ---
        if "start match" in line:
            logger.info("Nouvelle partie détectée")
            self.player_alive = True

    # ------------------------------------------------------------------
    # Getters
    # ------------------------------------------------------------------

    def get_position(self) -> tuple[float, float]:
        """
        Retourne la position du joueur.

        LIMITATION: econ sur TW 0.7 ne donne PAS la position.
        Options pour l'obtenir :
        1. Modifier le serveur pour logger la position à chaque tick
        2. Parser les logs serveur (stdout) depuis un processus séparé
        3. Ne pas utiliser la position (se baser uniquement sur l'image)

        Pour l'instant retourne (0, 0). À implémenter selon ton approche.
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