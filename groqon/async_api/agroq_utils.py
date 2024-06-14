# groq_utils.py

import json
import re
from pathlib import Path
from typing import Any
from termcolor import colored

from tqdm import tqdm

from ..groq_config import modelindex, DEFAULT_MODEL
from ..logger import get_logger
from typing import Dict, Union
import aiofiles
from functools import cache
from datetime import datetime
from .schema import ResponseModel
logger = get_logger(__name__)

async def save_dict_to_json(output_dict: Dict[str, Any], save_dir: str, file_name: str):
    """Save output dictionary to a JSON file."""

    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)

    file_name = "".join(c for c in file_name if c.isalnum() or c.isspace()).strip()
    file_name_split = file_name.split(" ")

    if len(file_name_split) > 10:
        file_name = " ".join(file_name_split[:10])

    json_file_path = save_dir_path / f"{file_name}.json"

    async with aiofiles.open(json_file_path, "w") as f:
        await f.write(json.dumps(output_dict, indent=4))

async def write_json(data: Union[Dict, str], filename: str):
    """Write dictionary or JSON string to a file."""

    async with aiofiles.open(filename, "w") as f:
        await f.write(json.dumps(data, indent=4))


def now() -> str:
    """Returns the current time as a string."""
    return datetime.now().strftime("%H:%M:%S")

@cache
def get_model_from_name(model: str) -> str:
    """Get the full model name from the given partial name."""

    model = model.lower().strip()
    for model_name in modelindex:
        if model in model_name.lower().strip():
            return model_name

    print("Available models: ", modelindex)
    print("Using default model: ", DEFAULT_MODEL)
    return DEFAULT_MODEL


def extract_rate_limit_info(data):
    json_str = json.dumps(data) if isinstance(data, dict) else data
    # Define regular expressions
    model_regex = r"model\s`(.*?)`"
    model_limit_regex = r"Limit\s(\d+)"
    wait_time_regex = r"try again in\s(\d+)ms"
    type_regex = r"type\":\s*\"(.*?)\""
    code_regex = r"code\":\s*\"(.*?)\""

    # Extract values using regular expressions
    model = re.search(model_regex, json_str)
    model_limit = re.search(model_limit_regex, json_str)
    wait_time = re.search(wait_time_regex, json_str)
    type_val = re.search(type_regex, json_str)
    code_val = re.search(code_regex, json_str)

    # Create a dictionary with extracted values
    return {
        "model": model.group(1) if model else None,
        "model_limit (RPM)": int(model_limit.group(1)) if model_limit else None,
        "wait_time (ms)": int(wait_time.group(1)) if wait_time else None,
        "type": type_val.group(1) if type_val else None,
        "code": code_val.group(1) if code_val else None,
    }


def print_model_response(response: ResponseModel):
    """Print the model response."""
    print(colored(f"query : {response.query}", "green"))
    print(
        colored(
            f"response : {response.response_text}",
            "yellow",
        )
    )
    print(
        colored(
            f"Model : {response.model}", "magenta"
        )
    )
    print(
        colored(
            f"Speed : {response.tokens_per_second:.2f} T/s", "magenta"
        )
    )
    if response.status_code != 200:
    
        print(
            colored(
                f"status code: {response.status_code}",
                'red'
            )
        )
    print()


async def save_cookie(cookie: dict | str | list[dict], file_path: str) -> None:
    file_path = Path(file_path)
    with open(file_path, "w") as f:
        json.dump(cookie, f)


def get_cookie(file_path: str) -> Any:
    file_path = Path(file_path)
    with open(file_path, "r") as f:
        return json.load(f)


def file_exists(file_path: str=None) -> bool:
    if file_path is not None:
        return Path(file_path).is_file()


def show_progress_bar(iterable, desc: str) -> Any:
    return tqdm(iterable, desc=desc)
