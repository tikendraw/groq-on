import json
import os
from pathlib import Path

from .logger import get_logger

logger = get_logger(__name__)

home_dir = os.path.expanduser("~")
groq_config_folder = os.path.join(home_dir, ".groqon")
GROQ_COOKIE_FILE = Path(groq_config_folder) / "groq_cookie.json"
MODEL_JSON_FILE = Path(groq_config_folder) / "models.json"


URL = "https://groq.com/"
API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_URL = "https://api.groq.com/openai/v1/models"
AUTHENTICATION_URL = "https://web.stytch.com/sdk/v1/sessions/authenticate"

GLOBAL_LIMITS = {
    "gemma-7b-it": {"requests_per_minute": 30, "calls": 18},
    "llama3-70b-8192": {"requests_per_minute": 30, "calls": 16},
    "llama3-8b-8192": {"requests_per_minute": 30, "calls": 16},
    "mixtral-8x7b-32768": {"requests_per_minute": 30, "calls": 18},
}


def get_model_ids(models_file=MODEL_JSON_FILE):
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
        logger.exception("Error reading models.json ", exc_info=e)
        modelindex = [
            "gemma-7b-it",
            "llama3-70b-8192",
            "llama3-8b-8192",
            "mixtral-8x7b-32768",
        ]

    return modelindex


modelindex = get_model_ids()

DEFAULT_MODEL = modelindex[2]  # Llama3-8b
