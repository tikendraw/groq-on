import os

from .async_api.agroq import agroq
from .groq_config import groq_config_folder
from .logger import get_logger

os.makedirs(groq_config_folder, exist_ok=True)


__all__ = ["get_logger", 'agroq']
