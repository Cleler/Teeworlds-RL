"""Chargement et validation de la configuration."""

import yaml
import logging

logger = logging.getLogger(__name__)


def load_config(path: str = "reinforcement_learning/configs/default.yaml") -> dict:
    """Charge la configuration depuis un fichier YAML."""
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Configuration chargée depuis {path}")
    return config
