import yaml
import json
import datetime
import os
from typing import Any, Dict, Optional
import logging

# Setup basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_config(config_path: str) -> Optional[Dict[str, Any]]:
    """
    Load a YAML configuration file.

    Args:
        config_path (str): The path to the configuration file.

    Returns:
        Optional[dict]: A dictionary containing the configuration loaded from the YAML file, or None if an error occurs.
    """
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
            if not isinstance(config, dict):
                logging.error(
                    f"The configuration file does not contain a valid YAML dictionary: {config_path}"
                )
                return None
            return config
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file {config_path}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error loading configuration: {e}")
    return None


def generate_project_folder_name(base_name: Optional[str]) -> str:
    """
    Generate a project folder name using a base name and the current timestamp.

    Args:
        base_name (Optional[str]): The base name to prepend to the timestamp. If None or empty,
                                   a default prefix 'project' is used.

    Returns:
        str: A string representing the folder name, which includes the base name (or a default prefix)
             and the current timestamp.

    Raises:
        ValueError: If the base_name contains characters not allowed in file paths.
    """
    # Use a default prefix if base_name is None or empty
    if not base_name:
        base_name = "project"

    # Validate that base_name does not contain characters illegal in file paths
    if any(char in base_name for char in '\\/*?:"<>|'):
        raise ValueError(
            f"base_name contains characters not allowed in file paths: {base_name}"
        )

    # Get the current timestamp in year-month-day-hour-minute-second format
    timestamp = datetime.datetime.now().strftime("%y%m%d%H%M%S")

    # Concatenate the base name and timestamp to form the folder name
    folder_name = f"{base_name}-{timestamp}"
    return folder_name


def create_folder(base_path: str, folder_name: str) -> Optional[str]:
    """
    Create a new folder at the specified base path with the given folder name.

    Args:
        base_path (str): The path where the new folder should be created.
        folder_name (str): The name of the folder to create.

    Returns:
        Optional[str]: The full path to the newly created folder, or None if an error occurred.
    """
    # Combine the base path and folder name to form the full project path
    project_path = os.path.join(base_path, folder_name)

    try:
        # Use exist_ok=True to avoid raising an error if the directory already exists
        os.makedirs(project_path, exist_ok=True)
        return project_path
    except Exception as e:
        logging.error(f"Failed to create folder '{project_path}': {e}")
        return None


def initialize_project_data(project_folder: str, fullpath: str) -> Dict[str, Any]:
    """
    Initialize the project data in a JSON file within the project folder.

    Args:
        project_folder (str): The path to the project folder where the JSON file will be created.
        fullpath (str): The full path to the source audio file.

    Returns:
        dict: A dictionary representing the initial project data.
    """
    # Extract the directory path and filename from the fullpath argument
    directory_path = os.path.dirname(fullpath)
    filename = os.path.basename(fullpath)
    # Construct the path to the JSON file within the project folder
    json_filename = os.path.join(project_folder, "project_data.json")

    # Define the initial structure of the project data
    initial_data = {
        "audio_metadata": {
            "source_audio_path": directory_path,
            "source_audio_filename": filename,
            "max_power": None,
            "sample_rate": None,
            "sample_count": None,
        },
        "images_metadata": [],
    }
    # Write the initial project data to the JSON file
    with open(json_filename, "w") as json_file:
        json.dump(initial_data, json_file, indent=4)

    return initial_data


def load_project_data(project_folder: str) -> Optional[Dict[str, Any]]:
    """
    Load project data from a JSON file within the specified project folder.

    Args:
        project_folder (str): The path to the project folder.

    Returns:
        Optional[Dict[str, Any]]: The loaded project data as a dictionary, or None if the file does not exist or an error occurs.
    """
    json_filename = os.path.join(project_folder, "project_data.json")

    if os.path.exists(json_filename):
        try:
            with open(json_filename, "r") as json_file:
                return json.load(json_file)
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON from {json_filename}: {e}")
        except Exception as e:
            logging.error(f"Unexpected error loading project data: {e}")
    else:
        logging.error("Project data file not found.")

    return None


def update_project_data(
    project_folder: str, key: str, new_data: Any
) -> Optional[Dict[str, Any]]:
    """
    Update a specific key in the project data JSON file with new data.

    Args:
        project_folder (str): The path to the project folder.
        key (str): The key within the project data to update.
        new_data (Any): The new data to insert or replace.

    Returns:
        Optional[Dict[str, Any]]: The updated project data as a dictionary, or None if an error occurs.
    """
    json_filename = os.path.join(project_folder, "project_data.json")

    if not os.path.exists(json_filename):
        logging.error(
            f"Project data file not found at {json_filename}. Initial data setup may be required."
        )
        return None

    try:
        with open(json_filename, "r") as json_file:
            project_data = json.load(json_file)
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from {json_filename}: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error reading project data: {e}")
        return None

    # Determine whether to update or append new data
    if key in project_data:
        if isinstance(project_data[key], list):
            project_data[key].append(new_data)  # Append if the existing data is a list
        else:
            project_data[key] = new_data  # Replace existing data
    else:
        project_data[key] = new_data  # Insert new key-value pair

    try:
        with open(json_filename, "w") as json_file:
            json.dump(project_data, json_file, indent=4)
    except Exception as e:
        logging.error(f"Unexpected error updating project data: {e}")
        return None

    return project_data
