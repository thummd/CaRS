"""Configuration files for DS3M-FANTOM experiments."""

from pathlib import Path

CONFIG_DIR = Path(__file__).parent

def get_config_path(name: str) -> Path:
    """Get path to a configuration file."""
    return CONFIG_DIR / f"{name}.yaml"
