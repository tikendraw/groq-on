
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel

from ..groq_config import (
    DEFAULT_MODEL,
    GROQ_COOKIE_FILE,
    MAX_TOKENS,
    STREAM,
    SYSTEM_PROMPT,
    TEMPERATURE,
    TOP_P,
)


class ErrorModel(BaseModel):
    message: str | None = None
    type: str|None = None
    code: str|None = None


class ErrorResponse(BaseModel):
    error: ErrorModel
    query:str|None = None
    status_code:int|None = None
        
class AgroqServerConfig(BaseModel):
    cookie_file: str = GROQ_COOKIE_FILE
    models: List[str] = [DEFAULT_MODEL]
    headless: bool = True
    n_workers: int = 2
    reset_login: bool = False
    verbose:bool = True


class AgroqClientConfig(BaseModel):
    models: List[str] = [DEFAULT_MODEL]
    save_dir: Union[str, Path, None] = None
    system_prompt: str = SYSTEM_PROMPT
    print_output: bool = True
    temperature: float = TEMPERATURE
    max_tokens: int = MAX_TOKENS
    top_p:int = TOP_P
    stream: bool = STREAM
    stop_server:bool = False
    
    def __init__(self, **data):
        super().__init__(**data)
        if self.save_dir is not None:
            self.save_dir = Path(self.save_dir)
            self.save_dir.mkdir(parents=True, exist_ok=True)


class MessageModel(BaseModel):
    content: str
    role: str


class APIRequestModel(BaseModel):
    local_id:str|None =None
    query:str|None = None
    model:str = DEFAULT_MODEL
    system_prompt:str = SYSTEM_PROMPT
    messages: List[MessageModel] | None = None
    temperature: float = TEMPERATURE
    max_tokens: int = MAX_TOKENS
    top_p: int = TOP_P
    stream: bool = STREAM
    
    class Config:
        arbitrary_types_allowed = False
    
    def __init__(self, **data):
        super().__init__(**data)
        if self.messages is None:
            self.messages = []
            self.add_initial_message()
    
    def get_query(self):
        if self.query is None:
            user_messages = [message for message in self.messages if message.role == "user"]
            if user_messages:
                self.query = user_messages[-1].content
                return self.query
        return self.query
    
    def add_initial_message(self):        
        self.add_message(content=self.system_prompt, role="system")
        self.add_message(content=self.query, role="user")

    def make_message(self, role: str, content: str):
        return MessageModel(role=role, content=content)
    
    def add_message(self, role: str, content: str):
        self.messages.append(self.make_message(role=role, content=content))
    
    def add_messages(self, messages: List[MessageModel]):
        self.messages.extend(messages)
    
    # this doesn't work
    # def add_history(self, response: 'APIResponseModel'):
    #     if response.query and response.response_text:
    #         last_message = self.messages[-1] if self.messages else None
    #         if not (last_message and last_message.role == "user" and last_message.content == response.query):
    #             self.add_message(role="user", content=response.query)
    #         self.add_message(role="assistant", content=response.response_text)
            
    def model_dump(self):
        return {
            "model":self.model,
            "messages":[
                {
                    "role":message.role,
                    "content":message.content
                } for message in self.messages
            ],
            "temperature":self.temperature,
            "max_tokens":self.max_tokens,
            "top_p":self.top_p,
            "stream":self.stream
        }
    
    def model_dump_json(self):
        return json.dumps(self.model_dump())


class ChoiceModel(BaseModel):
    index: int
    message: MessageModel
    logprobs: Any = None
    finish_reason: str

class Xgroq(BaseModel):
    id:str

class APIResponseModel(BaseModel):
    local_id:str|None =None
    query:str|None = None
    id: str 
    object: str
    created: int
    model: str
    choices: List[ChoiceModel]
    usage: Dict[str, Any]
    system_fingerprint: str | None =None
    x_groq: Xgroq

    
def set_local_id_and_query(api_request:APIRequestModel, api_response:APIResponseModel | ErrorResponse):
    
    if isinstance(api_response, APIResponseModel):
        api_response.local_id  = api_request.local_id 
    
    api_response.query  = api_request.get_query()
    return api_response

