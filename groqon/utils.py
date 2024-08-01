import inspect
import json
import time
from datetime import datetime
from functools import wraps
from typing import Any, Dict

import yaml
from termcolor import colored


def get_current_time_str():
    now = datetime.now()
    time_str = now.strftime("%Y-%m-%d %H:%M:%S")
    return time_str


def log_function_call(func):

    if inspect.iscoroutinefunction(func):

        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            cc(
                f"Starting function '{func.__name__}' at {start_time}",
                "green",
                "on_white",
            )

            result = await func(*args, **kwargs)

            end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            cc(f"Exiting function '{func.__name__}' at {end_time}", "red", "on_white")

            return result

    else:

        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            cc(
                f"Starting function '{func.__name__}' at {start_time}",
                "green",
                "on_white",
            )

            result = func(*args, **kwargs)

            end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            cc(f"Exiting function '{func.__name__}' at {end_time}", "red", "on_white")

            return result

    return wrapper


def cc(x, *args, end=None, **kwargs):
    """
    Available text colors:
    black, red, green, yellow, blue, magenta, cyan, white,
    light_grey, dark_grey, light_red, light_green, light_yellow, light_blue,
    light_magenta, light_cyan.

    Available text highlights:
        on_black, on_red, on_green, on_yellow, on_blue, on_magenta, on_cyan, on_white,
        on_light_grey, on_dark_grey, on_light_red, on_light_green, on_light_yellow,
        on_light_blue, on_light_magenta, on_light_cyan.

    Available attributes:
        bold, dark, underline, blink, reverse, concealed.

    Example:
        colored('Hello, World!', 'red', 'on_black', ['bold', 'blink'])
        colored('Hello, World!', 'green')
    """
    print(colored(x, *args, **kwargs), end=end)


# function to read config.yaml
def load_config(config_file):
    """
    Reads the config.yaml file and returns the config dictionary.

    Args:
        config_file: The path to the config.yaml file.

    Returns:
        dict: The config dictionary. The config dictionary.
    """
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config


# function to save dict to config.yaml
def save_config(config: dict, config_file):
    """
    Saves the config dictionary to the config.yaml file.

    Args:
        config: The config dictionary.
        config_file: The path to the config.yaml file.

    Returns:
        None
    """
    with open(config_file, "w") as f:
        yaml.dump(config, f)

    return


def get_model_ids(models_file, exclude: list = None):
    """
    Reads the model IDs from the models.json file and populates the modelindex list.

    Returns:
        None
    """
    modelindex = []
    try:

        with open(models_file, "r") as f:
            models_data = json.load(f)

        modelindex.extend(
            model["id"] for model in models_data["data"] if model.get("active")
        )
    except Exception as e:
        print(e)

    if exclude:
        modelindex = [model for model in modelindex if model not in exclude]

    return modelindex
