import os
from pathlib import Path
import os
from pathlib import Path
import json


home_dir = os.path.expanduser('~')  
groq_config_folder = os.path.join(home_dir, '.groqon')
GROQ_COOKIE_FILE= Path(groq_config_folder) / "groq_cookie.json"
MODEL_JSON_FILE = Path(groq_config_folder) / "models.json"


URL = "https://groq.com/"
API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_URL = "https://api.groq.com/openai/v1/models"


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


        for model in models_data["data"]:
            model_id = model["id"]
            modelindex.append(model_id)
    except Exception as e:
        print(f"Error reading models.json: {e}")

    return modelindex

modelindex = get_model_ids()  
print(modelindex)


