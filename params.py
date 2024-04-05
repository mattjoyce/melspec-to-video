""" class to help manage parameters and config """

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, List

import yaml


class Params:
    """
    A configuration manager that supports loading from YAML or JSON files,
    overlaying with command-line arguments,
    and providing dict-like access to configuration values.

    Attributes:
        config (Dict[str, Any]): The current configuration dictionary.

    Args:
      file_path (Optional[str]): Path to a YAML or JSON configuration file. Default is None.
      args (Optional[argparse.Namespace]): Command-line arguments object. Default is None.
      file_type (str): The type of the configuration file ('yaml' or 'json'). Default is 'yaml'.
      default_config (Optional[Dict[str, Any]]): A default config dictionary. Default is None.
    """

    def __init__(
        self,
        file_path: Optional[str] = None,
        args: Optional[argparse.Namespace] = None,
        file_type: str = "yaml",
        default_config: Optional[Dict[str, Any]] = None,
    ):
        self._config = (
            default_config or {}
        )  # Use a private attribute to store the configuration dictionary
        self._load_config(file_path, file_type)
        if args:
            self.overlay_args(args)
        self._set_dynamic_attributes()
        self._exclusions: List[str] = []  # List to hold keys to exclude during save

    def _load_config(self, file_path: Optional[str], file_type: str):
        if file_path:
            path = Path(file_path)
            if path.exists():
                if file_type == "yaml":
                    self._config = self.load_yaml_config(file_path)
                elif file_type == "json":
                    self._config = self.load_json_config(file_path)
                else:
                    raise ValueError(
                        "Unsupported file type. Please use 'yaml' or 'json'."
                    )

    def _set_dynamic_attributes(self):
        """internal setter for key in the internal config list"""
        for key, value in self._config.items():
            setattr(self, key, value)

    def __getattr__(self, name: str) -> Any:
        """Allows getting configuration values as attributes."""
        try:
            return self._config[name]
        except KeyError as e:
            raise AttributeError(f"'Params' object has no attribute '{name}'") from e

    def __setattr__(self, name: str, value: Any):
        """Allows setting configuration values as attributes, distinguishing between
        special attributes and configuration keys."""
        if name in ["_exclusions"]:
            # Directly handle special attributes; they should not be added to _config
            object.__setattr__(self, name, value)
        elif name == "_config" or name.startswith("_"):
            # Handle private attributes, including _config itself, normally
            super().__setattr__(name, value)
        else:
            # All other attributes are treated as configuration keys
            self._config[name] = value
            super().__setattr__(name, value)

    def __repr__(self):
        # Convert the internal configuration dictionary to a pretty-printed string
        return json.dumps(self._config, indent=4, sort_keys=True)

    def set_exclusions(self, keys: List[str]):
        """Sets the list of configuration keys to exclude during save operations.

        Args:
            keys (List[str]): A list of keys to exclude.
        """
        self._exclusions = keys

    def load_yaml_config(self, config_path: str) -> Dict[str, Any]:
        """Loads a YAML configuration file using pathlib for enhanced path handling.

        Args:
            config_path (str): The path to the YAML configuration file.

        Returns:
            Dict[str, Any]: The loaded configuration dictionary.

        Raises:
            ValueError: If there are issues parsing the YAML file or the file cannot be found.
        """
        path = Path(config_path)
        try:
            content = path.read_text(encoding="utf-8")
            return yaml.safe_load(content)
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file at {config_path}: {e}") from e
        except FileNotFoundError:
            raise ValueError(
                f"YAML configuration file not found at {config_path}"
            ) from e

    def load_json_config(self, config_path: str) -> Dict[str, Any]:
        """Loads a JSON configuration file using pathlib for enhanced path handling.

        Args:
            config_path (str): The path to the JSON configuration file.

        Returns:
            Dict[str, Any]: The loaded configuration dictionary.

        Raises:
            ValueError: If there are issues parsing the JSON file or the file cannot be found.
        """
        path = Path(config_path)
        try:
            content = path.read_text(encoding="utf-8")
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error parsing JSON file at {config_path}: {e}") from e
        except FileNotFoundError:
            raise ValueError(
                f"JSON configuration file not found at {config_path}"
            ) from e

    def overlay_args(self, args: argparse.Namespace):
        """Overlays command-line arguments onto the existing configuration."""
        for key, value in vars(args).items():
            if value is not None:
                self._config[key] = value

    def __getitem__(self, key: str) -> Any:
        """Allows for dict-like retrieval of configuration values."""
        return self._config[key]

    def __setitem__(self, key: str, value: Any):
        """Allows for dict-like setting of configuration values."""
        self._config[key] = value

    def to_json(self) -> str:
        """Serializes the configuration to a JSON string."""
        return json.dumps(self._config, indent=4)

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieves a configuration value, returning a default if the key is not found."""
        return self._config.get(key, default)

    def save_to_yaml(self, file_path: str):
        """Saves the current configuration to a YAML file, respecting any set exclusions."""
        config_to_save = {
            key: value
            for key, value in self._config.items()
            if key not in self._exclusions
        }
        path = Path(file_path)
        content = yaml.dump(config_to_save, default_flow_style=False)
        path.write_text(content, encoding="utf-8")

    def save_to_json(self, file_path: str):
        """Saves the current configuration to a JSON file, respecting any set exclusions."""
        config_to_save = {
            key: value
            for key, value in self._config.items()
            if key not in self.exclusions
        }
        path = Path(file_path)
        content = json.dumps(config_to_save, indent=4)
        path.write_text(content, encoding="utf-8")
