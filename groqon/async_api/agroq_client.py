import asyncio
import json

# import uuid
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import httpx
from groq.types.chat.chat_completion_message_param import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from pydantic import BaseModel

from ..groq_config import ENDTOKEN, PORT, modelindex
from ..logger import get_logger
from ..utils import cc, log_function_call
from ._utils import (
    get_model_from_name,
    print_model_response,
    write_dict_to_json,
    x_eq_len_of_y,
)
from .schema import (
    GROQ_APIRequest,
    GroqonClientConfig,
)

logger = get_logger(__name__)


class GroqonClient(BaseModel):
    config: GroqonClientConfig
    _PORT: Optional[int] = PORT
    tools: list[dict] | None = None
    _base_url: str = f"http://localhost:{PORT}/openai/v1/chat/completions"

    # @log_function_call
    async def multi_query_async(
        self,
        query: Union[str, List[str]] = None,
        model: Union[str, List[str]] = None,
        system_prompt: Union[str, List[str]] = None,
        temperature: Union[float, List[float]] = None,
        max_tokens: Union[int, List[int]] = None,
        top_p: Union[int, List[int]] = None,
        stop_server: bool = False,
        tools: list[dict] = None,
    ) -> list[Dict[str, Any]]:
        if tools:
            self.tools = tools

        if query is None and stop_server:
            query = [ENDTOKEN]

        # Convert single string inputs to lists
        query = [query] if isinstance(query, str) else query
        model = (
            [model]
            if isinstance(model, str)
            else (model or self.config.models or modelindex)
        )
        system_prompt = (
            [system_prompt]
            if isinstance(system_prompt, str)
            else (system_prompt or [self.config.system_prompt])
        )
        temperature = (
            [temperature]
            if isinstance(temperature, (int, float))
            else (temperature or [self.config.temperature])
        )
        max_tokens = (
            [max_tokens]
            if isinstance(max_tokens, int)
            else (max_tokens or [self.config.max_tokens])
        )
        top_p = [top_p] if isinstance(top_p, int) else (top_p or [self.config.top_p])

        model = [get_model_from_name(name) for name in model]

        model = x_eq_len_of_y(x=model, y=query)
        system_prompt = x_eq_len_of_y(x=system_prompt, y=query)
        temperature = x_eq_len_of_y(x=temperature, y=query)
        max_tokens = x_eq_len_of_y(x=max_tokens, y=query)

        print(f"got {len(query)} queries")

        responses = await self.make_request(
            query=query,
            model=model,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            save_dir=self.config.save_dir,
            print_output=self.config.print_output,
        )

        if stop_server:
            _ = await self.make_request(
                query=[ENDTOKEN],
                model=model,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return

        return responses

    async def make_request(
        self,
        query,
        model,
        system_prompt,
        temperature,
        max_tokens,
        save_dir=None,
        print_output=False,
        **kwargs,
    ):
        async def process_single_request(q, m, sp, t, mt, **kwargs):
            try:
                request_model = self.make_request_model(
                    query=q,
                    model=m,
                    system_prompt=sp,
                    temperature=t,
                    max_tokens=mt,
                    **kwargs,
                )
                output = await self.send_requestmodel(request_model)

                output["query"] = q
                if print_output:
                    print_model_response(output)

                if save_dir:
                    await write_dict_to_json(
                        output, save_dir, f"{datetime.now().isoformat()} {q}.json"
                    )

                return output
            except Exception as e:
                logger.error(f"Error processing query '{q}': {e}")
                return {"error": str(e), "query": q}

        request_params = zip(query, model, system_prompt, temperature, max_tokens)
        tasks = [
            process_single_request(q, m, sp, t, mt)
            for q, m, sp, t, mt in request_params
        ]

        responses = await asyncio.gather(*tasks)
        return [r for r in responses if r is not None]

    # @log_function_call
    async def send_requestmodel(self, request: GROQ_APIRequest):
        request_json = request.model_dump_json()
        # cc(f"request model dump: {request.model_dump()}", "red", "on_white")
        return await self.http_client(request_json)

    # @log_function_call
    async def http_client(
        self, message: str
    ) -> Union[dict, AsyncGenerator[dict, None]]:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self._base_url}",
                    content=message,
                    headers={"Content-Type": "application/json"},
                    timeout=30.0,  # Increase timeout
                )
                response.raise_for_status()

                return response.json()

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error: {e}")
            raise e
        except Exception as e:
            logger.error(f"Error in HTTP client: {e}")
            raise e

    async def process_stream_response(
        self, response: httpx.Response
    ) -> AsyncGenerator[dict, None]:
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                data = line.lstrip("data:").strip()
                if data == "[DONE]":
                    break
                try:
                    yield json.loads(data)
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse JSON: {data}")

    # @log_function_call
    def make_request_model(self, query: str, **kwargs) -> GROQ_APIRequest:
        return GROQ_APIRequest(
            messages=[
                ChatCompletionSystemMessageParam(
                    role="system",
                    content=kwargs.get("system_prompt", self.config.system_prompt),
                ),
                ChatCompletionUserMessageParam(role="user", content=query),
            ],
            model=kwargs.get("model", self.config.models[0]),
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            stream=False,  # always false
            top_p=kwargs.get("top_p", self.config.top_p),
        )

    def multi_query(
        self,
        query: Union[str, List[str]] = None,
        model: Union[str, List[str]] = None,
        system_prompt: Union[str, List[str]] = None,
        temperature: Union[float, List[float]] = None,
        max_tokens: Union[int, List[int]] = None,
        top_p: Union[int, List[int]] = None,
        stop_server: bool = False,
    ) -> list[Dict[str, Any]]:
        return asyncio.run(
            self.multi_query_async(
                query=query,
                model=model,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stop_server=stop_server,
            )
        )
