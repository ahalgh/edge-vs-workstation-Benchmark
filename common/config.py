"""Configuration loading with YAML files and CLI override support."""

import argparse
import copy
from pathlib import Path

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _set_nested(d: dict, dotted_key: str, value) -> None:
    """Set a value in a nested dict using dot notation (e.g. 'system.name')."""
    keys = dotted_key.split(".")
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def _parse_value(value: str):
    """Try to parse a CLI string as int, float, bool, list, or leave as string."""
    if value.lower() in ("true", "yes"):
        return True
    if value.lower() in ("false", "no"):
        return False
    if value.lower() in ("null", "none"):
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if "," in value:
        return [_parse_value(v.strip()) for v in value.split(",")]
    return value


def load_config(config_path: str = "config.yaml", cli_args: list[str] | None = None) -> dict:
    """Load YAML config and merge CLI overrides.

    CLI overrides use dot notation: --system.name jetson_thor --llm.batch_sizes 1,4,8
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path) as f:
        config = yaml.safe_load(f) or {}

    if cli_args:
        parser = argparse.ArgumentParser(add_help=False)
        # Parse unknown args as key=value pairs
        known, unknown = parser.parse_known_args(cli_args)
        i = 0
        while i < len(unknown):
            arg = unknown[i]
            if arg.startswith("--"):
                key = arg[2:]
                if i + 1 < len(unknown) and not unknown[i + 1].startswith("--"):
                    _set_nested(config, key, _parse_value(unknown[i + 1]))
                    i += 2
                else:
                    _set_nested(config, key, True)
                    i += 1
            else:
                i += 1

    return config


def get_benchmark_args() -> argparse.Namespace:
    """Standard CLI args shared by all benchmark runners."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args, remaining = parser.parse_known_args()
    args.remaining = remaining
    return args
