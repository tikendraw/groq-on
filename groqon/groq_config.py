import os
from pathlib import Path

from .utils import get_model_ids, save_config

home_dir = os.path.expanduser("~")
groq_config_folder = os.path.join(home_dir, ".groqon")
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
AUTHENTICATION_URL = 'https://web.stytch.com/sdk/v1/oauth/authenticate'
ENDTOKEN='<|endtoken|>'
PORT = 8888

modelindex = get_model_ids(models_file=MODEL_LIST_FILE)
DEFAULT_MODEL = modelindex[2]  # Llama3-8b

if not GROQON_CONFIG_FILE.exists():
    config = {
        'defaults': {
            'cookie_file': str(GROQ_COOKIE_FILE.absolute()),
            'models': modelindex,
            'headless': True,
            'n_workers': 2,
            'reset_login': False,
            'server_model_configs': str(MODEL_LIST_FILE.absolute()),
            'verbose': False,
            'print_output': True
        },
        'client':{
            'system_prompt': SYSTEM_PROMPT,
            'temperature': TEMPERATURE,
            'max_tokens': MAX_TOKENS,
            'top_p': TOP_P,
            'stream': STREAM
        }
    }
    
    save_config(config, GROQON_CONFIG_FILE)
    

