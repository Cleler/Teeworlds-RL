#!/usr/bin/env python3
"""Point d'entrée pour l'évaluation d'un modèle entraîné."""

import argparse
import logging

from src.utils.config import load_config
from src.agent.train import load_and_play

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def main():
    parser = argparse.ArgumentParser(description="Teeworlds RL - Évaluation")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Chemin vers le fichier de configuration",
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Chemin vers le modèle (.zip)",
    )
    parser.add_argument(
        "--episodes", type=int, default=10,
        help="Nombre d'épisodes à jouer",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    load_and_play(config, args.model, args.episodes)


if __name__ == "__main__":
    main()
