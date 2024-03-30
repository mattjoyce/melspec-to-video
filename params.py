import yaml
import json
import argparse
from typing import Any, Dict


class Params:
    def __init__(
        self, file_path: str, args: argparse.Namespace = None, file_type: str = "yaml"
    ):
        if file_type == "yaml":
            self.config = self.load_yaml_config(file_path)
        elif file_type == "json":
            self.config = self.load_json_config(file_path)
        else:
            raise ValueError("Unsupported file type. Please use 'yaml' or 'json'.")

        if args:
            self.overlay_args(args)

    def load_yaml_config(self, config_path: str) -> Dict[str, Any]:
        """Load the YAML configuration from the given path."""
        with open(config_path, "r") as file:
            return yaml.safe_load(file)

    def load_json_config(self, config_path: str) -> Dict[str, Any]:
        """Load the JSON configuration from the given path."""
        with open(config_path, "r") as file:
            return json.load(file)

    def overlay_args(self, args: argparse.Namespace):
        """Overlay command-line arguments on top of the configuration."""
        for key, value in vars(args).items():
            if value is not None:  # Only overlay provided args
                self.config[key] = value

    def __getitem__(self, key):
        """Enable dictionary-like access to configuration values."""
        return self.config[key]

    def __setitem__(self, key, value):
        """Allow setting configuration values, emulating dictionary behavior."""
        self.config[key] = value

    def to_json(self) -> str:
        """Convert the configuration to a JSON string."""
        return json.dumps(self.config, indent=4)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key with an optional default."""
        return self.config.get(key, default)

    def save_to_yaml(self, file_path: str):
        """Save the configuration to a YAML file."""
        with open(file_path, "w") as file:
            yaml.dump(self.config, file, default_flow_style=False)

    def save_to_json(self, file_path: str):
        """Save the configuration to a JSON file."""
        with open(file_path, "w") as file:
            json.dump(self.config, file, indent=4)
