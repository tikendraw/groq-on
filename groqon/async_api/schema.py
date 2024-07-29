import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Union

from groq.types.chat.chat_completion import (
    ChatCompletion,
    Choice,
    ChatCompletionMessage,
)
from groq.types.chat.chat_completion_assistant_message_param import (
    ChatCompletionAssistantMessageParam,
)
from groq.types.chat.chat_completion_chunk import ChatCompletionChunk
from groq.types.chat.chat_completion_function_message_param import (
    ChatCompletionFunctionMessageParam,
)
from groq.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from groq.types.chat.chat_completion_system_message_param import (
    ChatCompletionSystemMessageParam,
)
from groq.types.chat.chat_completion_tool_message_param import (
    ChatCompletionToolMessageParam,
)
from groq.types.chat.chat_completion_user_message_param import (
    ChatCompletionUserMessageParam,
)
from pydantic import BaseModel, Field, root_validator

from ..groq_config import (
    DEFAULT_MODEL,
    GROQ_COOKIE_FILE,
    MAX_TOKENS,
    STREAM,
    SYSTEM_PROMPT,
    TEMPERATURE,
    TOP_P,
)


class ChatCompletion(ChatCompletion):
    query: str | None = None
    local_id: str | None = None


class ChatCompletionChunk(ChatCompletionChunk):
    query: str | None = None
    local_id: str | None = None


class ErrorModel(BaseModel):
    message: str | None = None
    type: str | None = None
    code: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorModel
    query: str | None = None
    status_code: int | None = None


class AgroqServerConfig(BaseModel):
    cookie_file: str = GROQ_COOKIE_FILE
    models: List[str] = [DEFAULT_MODEL]
    headless: bool = True
    n_workers: int = 2
    reset_login: bool = False
    verbose: bool = True


class AgroqClientConfig(BaseModel):
    models: List[str] = [DEFAULT_MODEL]
    save_dir: Union[str, Path, None] = None
    system_prompt: str = SYSTEM_PROMPT
    print_output: bool = True
    temperature: float = TEMPERATURE
    max_tokens: int = MAX_TOKENS
    top_p: int = TOP_P
    stream: bool = STREAM
    stop_server: bool = False

    def __init__(self, **data):
        super().__init__(**data)
        if self.save_dir is not None:
            self.save_dir = Path(self.save_dir)
            self.save_dir.mkdir(parents=True, exist_ok=True)


class FunctionModel(BaseModel):
    name: str | None = None
    arguments: dict | None = None


class ToolCallsModel(BaseModel):
    id: str | None = None
    type: str | None = None
    function: FunctionModel | None = None


class MessageModel(BaseModel):
    content: str | None = None
    role: str | None = None
    tool_calls: list[ToolCallsModel] | None = None


class APIRequestModel(BaseModel):
    local_id: str | None = None
    query: str | None = None
    model: str = DEFAULT_MODEL
    system_prompt: str = SYSTEM_PROMPT
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
            user_messages = [
                message for message in self.messages if message.role == "user"
            ]
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

    def model_dump(self):
        return {
            "model": self.model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in self.messages
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "stream": self.stream,
        }

    def model_dump_json(self):
        return json.dumps(self.model_dump())


class ChoiceModel(BaseModel):
    index: int
    message: MessageModel
    logprobs: Any = None
    finish_reason: str


class Xgroq(BaseModel):
    id: str


class APIResponseModel(BaseModel, extra="allow"):
    local_id: str | None = None
    query: str | None = None
    id: str
    object: str
    created: int
    model: str
    choices: List[Choice]
    usage: Dict[str, Any]
    system_fingerprint: str | None = None
    x_groq: Xgroq


def set_local_id_and_query(
    api_request: APIRequestModel, api_response: APIResponseModel | ErrorResponse
):
    if isinstance(api_response, APIResponseModel) and isinstance(
        api_request, APIResponseModel
    ):
        api_response.local_id = api_request.local_id

    api_response.query = api_request.get_query()
    return api_response


NOT_GIVEN = None


class GROQ_APIRequest(BaseModel):
    # change list to iterables if validation error
    messages: List[ChatCompletionMessageParam]
    model: str
    max_tokens: Optional[int] = Field(default=MAX_TOKENS)
    stream: Optional[bool] = Field(default=False)
    top_p: Optional[float] = Field(default=TOP_P)
    temperature: Optional[float] = Field(default=TEMPERATURE)
    frequency_penalty: Optional[float] = Field(default=NOT_GIVEN)
    function_call: Optional[Any] = Field(default=NOT_GIVEN)
    functions: Optional[List[Any]] = Field(default=NOT_GIVEN)
    logit_bias: Optional[Dict[str, int]] = Field(default=NOT_GIVEN)
    logprobs: Optional[bool] = Field(default=NOT_GIVEN)
    n: Optional[int] = Field(default=NOT_GIVEN)
    parallel_tool_calls: Optional[bool] = Field(default=NOT_GIVEN)
    presence_penalty: Optional[float] = Field(default=NOT_GIVEN)
    response_format: Optional[Any] = Field(default=NOT_GIVEN)
    seed: Optional[int] = Field(default=NOT_GIVEN)
    stop: Optional[Union[str, List[str]]] = Field(default=NOT_GIVEN)
    tool_choice: Optional[str] = Field(default=NOT_GIVEN)
    tools: Optional[List[Dict[str, Any]]] = Field(default=NOT_GIVEN)
    top_logprobs: Optional[int] = Field(default=NOT_GIVEN)

    # class Config:
    # allow_unknown_fields = True
    # arbitrary_types_allowed = True
    class Config:
        json_encoders = {"SerializationIterator": lambda v: list(v)}

    def __init__(self, **data):
        super().__init__(**data)
        self._remove_not_given()

    def _remove_not_given(self):
        for field in list(self.__dict__.keys()):
            if getattr(self, field) == NOT_GIVEN:
                delattr(self, field)

    def get_query(self):
        for message in self.messages[::-1]:  # iterating reversed list
            if not isinstance(message, dict):
                if message.role == "user":
                    return message.content

            elif isinstance(message, dict):
                if message.get("role") == "user":
                    return message.get("content")

        return None

    @root_validator(pre=True)
    def parse_messages(cls, values):
        if "messages" in values:
            parsed_messages = []
            for message in values["messages"]:
                if isinstance(message, dict):
                    role = message.get("role")
                    if role == "system":
                        parsed_messages.append(
                            ChatCompletionSystemMessageParam(**message)
                        )
                    elif role == "user":
                        parsed_messages.append(
                            ChatCompletionUserMessageParam(**message)
                        )
                    elif role == "assistant":
                        parsed_messages.append(
                            ChatCompletionAssistantMessageParam(**message)
                        )
                    elif role == "tool":
                        parsed_messages.append(
                            ChatCompletionToolMessageParam(**message)
                        )
                    elif role == "function":
                        parsed_messages.append(
                            ChatCompletionFunctionMessageParam(**message)
                        )
                    else:
                        raise ValueError(f"Invalid message format: {message}")
                else:
                    parsed_messages.append(message)
            values["messages"] = parsed_messages
        return values


ResponseModels = Union[ChatCompletion | list[ChatCompletionChunk] | ErrorResponse]


class PartialChatCompletionChunk(BaseModel):
    id: Optional[str] = None
    choices: Optional[List[Choice]] = None
    created: Optional[int] = None
    model: Optional[str] = None
    object: Optional[Literal["chat.completion.chunk"]] = None
    system_fingerprint: Optional[str] = None
    usage: Optional[dict] = None
    x_groq: Optional[dict] = None

    class Config:
        extra = "allow"


def parse_chat_completion_chunks(
    chunks: List[PartialChatCompletionChunk],
) -> ChatCompletion:
    if not chunks:
        raise ValueError("No chunks provided")

    # Initialize with data from the first chunk
    first_chunk = chunks[0]
    last_chunk = chunks[-1]
    combined_data = {
        "id": first_chunk.id,
        "created": first_chunk.created,
        "model": first_chunk.model,
        "object": "chat.completion",
        "system_fingerprint": first_chunk.system_fingerprint,
        "choices": [],
        "usage": last_chunk.usage,
        "x_groq": last_chunk.x_groq,  # Use x_groq from the last chunk
    }

    # Prepare to collect message content for each choice
    choice_messages: Dict[int, Dict[str, Any]] = {}

    for chunk in chunks:
        if chunk.choices:
            for choice in chunk.choices:
                if choice.index not in choice_messages:
                    choice_messages[choice.index] = {
                        "role": choice.delta.role or "assistant",
                        "content": "",
                        "function_call": None,
                        "tool_calls": [],
                    }

                delta = choice.delta

                # Accumulate content
                if delta.content:
                    choice_messages[choice.index]["content"] += delta.content

                # Handle function calls (if present)
                if delta.function_call:
                    if choice_messages[choice.index]["function_call"] is None:
                        choice_messages[choice.index]["function_call"] = {
                            "name": "",
                            "arguments": "",
                        }
                    choice_messages[choice.index]["function_call"]["name"] += (
                        delta.function_call.name or ""
                    )
                    choice_messages[choice.index]["function_call"]["arguments"] += (
                        delta.function_call.arguments or ""
                    )

                # Handle tool calls (if present)
                if delta.tool_calls:
                    for tool_call in delta.tool_calls:
                        existing_tool_call = next(
                            (
                                t
                                for t in choice_messages[choice.index]["tool_calls"]
                                if t.id == tool_call.id
                            ),
                            None,
                        )
                        if existing_tool_call:
                            existing_tool_call.function.name += (
                                tool_call.function.name or ""
                            )
                            existing_tool_call.function.arguments += (
                                tool_call.function.arguments or ""
                            )
                        else:
                            choice_messages[choice.index]["tool_calls"].append(
                                tool_call
                            )

    # Construct final choices
    for index, message_data in choice_messages.items():
        choice = Choice(
            index=index,
            message=ChatCompletionMessage(**message_data),
            finish_reason=last_chunk.choices[-1].finish_reason
            if last_chunk.choices
            else None,
            logprobs=last_chunk.choices[-1].logprobs if last_chunk.choices else None,
        )
        combined_data["choices"].append(choice)

    # Create and return the ChatCompletion object
    return ChatCompletion(**combined_data)
