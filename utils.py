import yaml
import json
from datetime import datetime
import os
from typing import Any


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def generate_project_folder_name(base_name):
    timestamp = datetime.now().strftime("%y%m%d%H%M%S")
    folder_name = f"{base_name}-{timestamp}"
    return folder_name


def create_folder(base_path, folder_name):
    project_path = os.path.join(base_path, folder_name)
    if not os.path.exists(project_path):
        os.makedirs(project_path)
    return project_path


def initialize_project_data(project_folder, fullpath):
    directory_path = os.path.dirname(fullpath)
    filename = os.path.basename(fullpath)
    json_filename = os.path.join(project_folder, "project_data.json")

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
    # Write the project data to the JSON file
    with open(json_filename, "w") as json_file:
        json.dump(initial_data, json_file, indent=4)

    return initial_data


def load_project_data(project_folder):
    json_filename = os.path.join(project_folder, "project_data.json")

    if os.path.exists(json_filename):
        with open(json_filename, "r") as json_file:
            return json.load(json_file)
    else:
        print("Project data file not found.")
        return None


def update_project_data(project_folder, key, new_data):
    json_filename = os.path.join(project_folder, "project_data.json")
    # Ensure the project data file exists
    if not os.path.exists(json_filename):
        print("Project data file not found. Initial data setup may be required.")
        return

    # Load the existing project data
    with open(json_filename, "r") as json_file:
        project_data = json.load(json_file)
        print(f"current project data : {project_data}")
    # Check if the key exists and if it points to a list, append the new data
    if key in project_data and isinstance(project_data[key], list):
        project_data[key].append(new_data)
    else:
        # For non-list data or new keys, update or set the value directly
        project_data[key] = new_data

    print(f"updated project data : {project_data}")
    # Write the updated project data back to the file
    with open(json_filename, "w") as json_file:
        json.dump(project_data, json_file, indent=4)
    return project_data
