# groq_utils.py

import json
import math
from datetime import datetime
from functools import cache
from pathlib import Path

# from pprint import pprint
from typing import Any, Dict, Iterable

import aiofiles
from termcolor import colored
from tqdm import tqdm

from ..groq_config import DEFAULT_MODEL, modelindex
from ..logger import get_logger
from .schema import ChatCompletionChunk, ErrorResponse

logger = get_logger(__name__)


def make_raw_streamed_reponse(chunks: list[ChatCompletionChunk]) -> str:
    raw = ""
    for i in chunks:
        raw += "data: " + i.model_dump_json() + "\n\n"

    raw += "data: [DONE]"
    return raw


async def async_generator_from_iterables(*iterables: Iterable):
    """
    Generates tuples of corresponding elements from multiple iterables.

    Args:
        *iterables: A variable number of iterables to iterate over.

    Yields:
        A tuple containing the corresponding elements from each iterable.
    """

    # Find the shortest iterable to determine the number of iterations
    min_length = min(len(iterable) for iterable in iterables)

    # Iterate over the shortest iterable
    for i in range(min_length):
        yield tuple(iterable[i] for iterable in iterables)


def clean_name(name: str):
    """remove non-alphanumeric characters from string"""
    return "".join(c for c in name if c.isalnum() or c.isspace()).strip()


def shorten_file_name(file_name: str) -> str:
    """Shorten the file name to 10 characters."""
    file_name_split = file_name.split(" ")
    if len(file_name_split) > 10:
        file_name = " ".join(file_name_split[:10])
    return file_name


async def write_dict_to_json(
    output_dict: Dict[str, Any], directory: str | Path, file_name: str | Path
):
    """Save output dictionary to a JSON file."""

    if isinstance(file_name, str):
        file_name = Path(file_name)

    if file_name.suffix == "":
        file_name = file_name.stem + ".json"

    await write_str_to_file(json.dumps(output_dict, indent=4), directory, file_name)


async def write_str_to_file(content: str, directory: str | Path, file_name: str | Path):
    """saves content to given file name in given directory asynchronously"""

    if isinstance(directory, str):
        directory = Path(directory)
    if isinstance(file_name, str):
        file_name = Path(file_name)

    directory.mkdir(parents=True, exist_ok=True)
    file_name, suffix = file_name.stem, file_name.suffix
    file_name = clean_name(file_name)
    file_name = shorten_file_name(file_name)

    # check if file name has some extension and add .txt if not
    if suffix == "":
        suffix = ".txt"

    full_name = file_name + suffix
    file_path = directory / full_name

    async with aiofiles.open(file_path, "w") as f:
        await f.write(content)


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


def x_eq_len_of_y(x: list[str], y: list[str]) -> list[str]:
    x *= math.ceil(len(y) / len(x))
    return x


async def save_cookie(cookie: dict | list[dict], file_path: str) -> None:
    file_path = Path(file_path)
    with open(file_path, "w") as f:
        json.dump(cookie, f)


def get_cookie(file_path: str) -> Any:
    file_path = Path(file_path)
    with open(file_path, "r") as f:
        return json.load(f)


def file_exists(file_path: str = None) -> bool:
    if file_path is not None:
        return Path(file_path).is_file()


def show_progress_bar(iterable, desc: str) -> Any:
    return tqdm(iterable, desc=desc)


def calculate_tokens_per_second(usage: Dict[str, Any]) -> float:
    """Calculate tokens per second from usage statistics."""
    try:
        return usage["completion_tokens"] / usage["completion_time"]
    except (KeyError, ZeroDivisionError):
        return 0


def ccc(x, *args, end=None, **kwargs):
    print(colored(x, *args, **kwargs), end=end)


def strip_newlines(x: str) -> str:
    return x.strip("\n\n").strip("\n").strip()


def print_model_response(response: dict):
    """Print the model response."""

    if "error" in response.keys():
        response = ErrorResponse(**response)
        ccc(f"Query:    {response.query}", "green")
        ccc(f"Error: {response.error.message}", "red")
        ccc(f"Model: {response.error.type}", "magenta")
        ccc(f"Code: {response.error.code}", "magenta")
    else:
        ccc(f"Query:    {response.get('query')}", "green")
        ccc(
            f"Response: {strip_newlines(response.get('choices')[0].get('message').get('content'))}",
            "yellow",
        )
        ccc(f"Model: {response.get('model')}", "magenta")
        ccc(
            f"TOK/s: {calculate_tokens_per_second(response.get('usage'))}",
            "magenta",
        )

    print()
