#!/usr/bin/env python3
"""Point d'entrée pour l'entraînement."""

import argparse
import logging

from src.utils.config import load_config
from src.agent.train import train

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def main():
    parser = argparse.ArgumentParser(description="Teeworlds RL - Entraînement")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Chemin vers le fichier de configuration",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    train(config)


if __name__ == "__main__":
    main()
