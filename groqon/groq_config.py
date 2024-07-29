import os
from pathlib import Path

from .utils import get_model_ids, save_config

home_dir = os.path.expanduser("~")
groq_config_folder = Path(os.path.join(home_dir, ".groqon"))
groq_config_folder.mkdir(exist_ok=True)
groq_error_folder = groq_config_folder / "errors"
groq_error_folder.mkdir(exist_ok=True)

GROQON_CONFIG_FILE = Path(groq_config_folder) / "config.yaml"
GROQ_COOKIE_FILE = Path(groq_config_folder) / "groq_cookie.json"
MODEL_LIST_FILE = Path(groq_config_folder) / "models.json"

TOP_P = 1
STREAM = True
TEMPERATURE = 0.1
MAX_TOKENS = 2048
SYSTEM_PROMPT = "Please try to provide useful, helpful and actionable answers."

URL = "https://groq.com/"
API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_URL = "https://api.groq.com/openai/v1/models"
AUTHENTICATION_URL = "https://web.stytch.com/sdk/v1/oauth/authenticate"
ENDTOKEN = "<|stopgroqontoken|>"
PORT = 8888


# there is a whisper model in the models list in groq website (filter that)
EXCLUDED_MODELS = ["whisper-large-v3"]
modelindex = (
    get_model_ids(models_file=MODEL_LIST_FILE, exclude=EXCLUDED_MODELS)
    if MODEL_LIST_FILE.exists()
    else [
        "gemma2-9b-it",
        "gemma-7b-it",
        "llama-3.1-70b-versatile",
        "llama-3.1-8b-instant",
        "llama3-70b-8192",
        "llama3-8b-8192",
        "llama3-groq-70b-8192-tool-use-preview",
        "llama3-groq-8b-8192-tool-use-preview",
        "mixtral-8x7b-32768",
    ]
)
DEFAULT_MODEL = modelindex[3]  # Llama3-8b

if not GROQON_CONFIG_FILE.exists():
    config = {
        "defaults": {
            "cookie_file": str(GROQ_COOKIE_FILE.absolute()),
            "models": modelindex,
            "headless": True,
            "n_workers": 2,
            "reset_login": False,
            "verbose": False,
            "print_output": True,
        },
        "client": {
            "system_prompt": SYSTEM_PROMPT,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "top_p": TOP_P,
            "stream": STREAM,
        },
    }

    save_config(config, GROQON_CONFIG_FILE)
