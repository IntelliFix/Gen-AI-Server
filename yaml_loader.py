import yaml
from flask import Blueprint


def open_yaml(filename):
    with open(filename, "r") as file:
        configuration = yaml.load(file, Loader=yaml.FullLoader)
    return configuration


def blueprint_yaml(collection_name, blueprint_module):
    blueprint_configuration = open_yaml("config.yaml")

    blueprint_module_data = blueprint_configuration[collection_name][blueprint_module]
    data_blueprint = Blueprint(blueprint_module_data["root_path"], __name__)

    return blueprint_module_data["routes"], data_blueprint