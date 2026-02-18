#!/bin/bash
# ============================================
# Lance le serveur Teeworlds avec econ activé
# ============================================
#
# Prérequis:
#   - Teeworlds serveur installé
#   - Adapter TEEWORLDS_SERVER au chemin de ton binaire
#
# Usage:
#   bash scripts/launch_server.sh
#   bash scripts/launch_server.sh /chemin/vers/teeworlds_srv

set -e

# TEEWORLDS_SERVER="${1:-teeworlds_srv}"
# TEEWORLDS_SERVER="./teeworlds-game/build-server/teeworlds_srv"
TEEWORLDS_SERVER="${1:-"./teeworlds-game/build-server/teeworlds_srv"}"

# Vérifier que le binaire existe
if ! command -v "$TEEWORLDS_SERVER" &> /dev/null; then
    echo "Erreur: '$TEEWORLDS_SERVER' non trouvé."
    echo "Usage: $0 /chemin/vers/teeworlds_srv"
    exit 1
fi

# Créer un fichier de config temporaire pour le serveur
CFG="./reinforcement_learning/configs/server.cfg"

echo "Config serveur écrite dans: $CFG"
echo "Démarrage du serveur..."
echo "  Econ: 127.0.0.1:8303 (password: password)"
echo ""

$TEEWORLDS_SERVER -f "$CFG"
