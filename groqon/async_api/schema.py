
from pydantic import BaseModel
from pathlib import Path
from typing import List, Union, Dict, Any
from ..groq_config import GROQ_COOKIE_FILE, DEFAULT_MODEL, MODEL_LIST_FILE
import uuid

TOP_P = 1
STREAM = True
TEMPERATURE = 0.1
MAX_TOKENS = 2048
SYSTEM_PROMPT = "Please try to provide useful, helpful and actionable answers."


class AgroqConfig(BaseModel):
    query_list: List[str] = None
    cookie_file: str = GROQ_COOKIE_FILE
    models: List[str] = [DEFAULT_MODEL]
    headless: bool = True
    save_dir: Union[str, Path] = None
    system_prompt: str = SYSTEM_PROMPT
    print_output: bool = True
    temperature: float = TEMPERATURE
    max_tokens: int = MAX_TOKENS
    n_workers: int = 2
    top_p:int = TOP_P
    stream: bool = STREAM
    reset_login: bool = False
    server_model_configs: str = MODEL_LIST_FILE
    verbose:bool = True
    keep_running:bool = False

class MessageModel(BaseModel):
    content: str
    role: str

class RequestModel(BaseModel):
    id: str = uuid.uuid4().hex
    query: str
    model: str = DEFAULT_MODEL
    system_prompt: str = SYSTEM_PROMPT
    temperature: float = TEMPERATURE
    max_tokens: int = MAX_TOKENS
    top_p: int = TOP_P
    stream: bool = STREAM
    messages: List[MessageModel] = []
    
    def make_message(self, role: str, content: str):
        return MessageModel(role=role, content=content)
    
    def add_message(self, role: str, content: str):
        self.messages.append(self.make_message(role=role, content=content))
    
    def add_messages(self, messages: List[MessageModel]):
        self.messages.extend(messages)
        
    def add_history(self, response: 'ResponseModel'):
        if response.query and response.response_text:
            last_message = self.messages[-1]
            if not (last_message.role == "user" and last_message.content==response.query):
                self.add_message(role="user", content=response.query)
            self.add_message(role="assistant", content=response.response_text)

class APIRequestModel(BaseModel):
    model:str
    messages: List[MessageModel]
    temperature: float
    max_tokens: int
    top_p: int
    stream: bool
    
    class Config:
        arbitrary_types_allowed = False
    
    @classmethod
    def from_request_model(self, request_model: RequestModel):
        return APIRequestModel(
            model=request_model.model,
            messages=request_model.messages,
            temperature=request_model.temperature,
            max_tokens=request_model.max_tokens,
            top_p=request_model.top_p,
            stream=request_model.stream
        )


class ResponseModel(BaseModel):
    id: str
    query: str | None = None
    response_text: str | None = None
    raw_response: str | Dict = None
    model: str | None = None
    tokens_per_second: float | int | None = 0
    stats: Dict[str, Any] | str | None = {}
    status_code: int | None = None
    
    def __repr__(self):
        return f"ResponseModel(id={self.id}) \nquery={self.query} \nresponse_text={self.response_text} \nmodel={self.model} \ntokens_per_second={self.tokens_per_second}"
