import yaml
import json
import argparse
import os
from typing import Any, Dict, Optional


class Params:
    """
    A configuration manager that supports loading from YAML or JSON files,
    overlaying with command-line arguments, and providing dict-like access to configuration values.

    Attributes:
        config (Dict[str, Any]): The current configuration dictionary.

    Args:
        file_path (Optional[str]): Path to a YAML or JSON configuration file. Default is None.
        args (Optional[argparse.Namespace]): Command-line arguments object. Default is None.
        file_type (str): The type of the configuration file ('yaml' or 'json'). Default is 'yaml'.
        default_config (Optional[Dict[str, Any]]): A default configuration dictionary. Default is None.
    """

    def __init__(
        self,
        file_path: Optional[str] = None,
        args: Optional[argparse.Namespace] = None,
        file_type: str = "yaml",
        default_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initializes the Params object, loads configuration from a file if provided,
        and overlays with command-line arguments.
        """
        self.config = default_config or {}

        if file_path and os.path.exists(file_path):
            if file_type == "yaml":
                self.config = self.load_yaml_config(file_path)
            elif file_type == "json":
                self.config = self.load_json_config(file_path)
            else:
                raise ValueError("Unsupported file type. Please use 'yaml' or 'json'.")

        if args:
            self.overlay_args(args)

    def __repr__(self):
        # Convert the internal configuration dictionary to a pretty-printed string
        return json.dumps(self.config, indent=4, sort_keys=True)

    def load_yaml_config(self, config_path: str) -> Dict[str, Any]:
        """Loads a YAML configuration file."""
        with open(config_path, "r") as file:
            return yaml.safe_load(file)

    def load_json_config(self, config_path: str) -> Dict[str, Any]:
        """Loads a JSON configuration file."""
        with open(config_path, "r") as file:
            return json.load(file)

    def overlay_args(self, args: argparse.Namespace):
        """Overlays command-line arguments onto the existing configuration."""
        for key, value in vars(args).items():
            if value is not None:
                self.config[key] = value

    def __getitem__(self, key: str) -> Any:
        """Allows for dict-like retrieval of configuration values."""
        return self.config[key]

    def __setitem__(self, key: str, value: Any):
        """Allows for dict-like setting of configuration values."""
        self.config[key] = value

    def to_json(self) -> str:
        """Serializes the configuration to a JSON string."""
        return json.dumps(self.config, indent=4)

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieves a configuration value, returning a default if the key is not found."""
        return self.config.get(key, default)

    def save_to_yaml(self, file_path: str):
        """Saves the current configuration to a YAML file."""
        with open(file_path, "w") as file:
            yaml.dump(self.config, file, default_flow_style=False)

    def save_to_json(self, file_path: str):
        """Saves the current configuration to a JSON file."""
        with open(file_path, "w") as file:
            json.dump(self.config, file, indent=4)
