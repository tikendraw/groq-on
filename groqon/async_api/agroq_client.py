import asyncio
import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Union
from pydantic import BaseModel
from ..groq_config import PORT, modelindex
from ..logger import get_logger
from ..utils import cc, log_function_call
from .agroq_utils import (
    async_generator_from_iterables,
    get_model_from_name,
    print_model_response,
    write_dict_to_json,
    x_eq_len_of_y,
)
from .schema import AgroqClientConfig, RequestModel

# from line_profiler import profile
logger = get_logger(__name__)


class AgroqClient(BaseModel):
    """Checkout this url to get help with model selection based on tasks: https://artificialanalysis.ai/providers/groq """
    config: AgroqClientConfig
    _PORT: int = PORT

    # @log_function_call
    async def multi_query_async(
        self,
        query: Union[str, List[str]],
        model: Union[str, List[str]] = None, 
        system_prompt: Union[str, List[str]] = None, 
        temperature: Union[float, List[float]] = None,
        max_tokens: Union[int, List[int]] = None,
        top_p: Union[int, List[int]] = None,
        stream: bool = True,
    ) -> list[Dict[str, Any]]:

        # Convert single string inputs to lists
        query = [query] if isinstance(query, str) else query
        model = [model] if isinstance(model, str) else (model or modelindex)
        system_prompt = [system_prompt] if isinstance(system_prompt, str) else (system_prompt or [self.config.system_prompt])
        temperature = [temperature] if isinstance(temperature, (int, float)) else (temperature or [self.config.temperature])
        max_tokens = [max_tokens] if isinstance(max_tokens, int) else (max_tokens or [self.config.max_tokens])
        top_p = [top_p] if isinstance(top_p, int) else (top_p or [self.config.top_p])
        
        

        model = [get_model_from_name(name) for name in model]

        model = x_eq_len_of_y(x=model, y=query)
        system_prompt = x_eq_len_of_y(x=system_prompt, y=query)
        temperature = x_eq_len_of_y(x=temperature, y=query)
        max_tokens = x_eq_len_of_y(x=max_tokens, y=query)
        
        print(f'got {len(query)} queries')


        responses = []


        async for q, m, sp, t, mt in async_generator_from_iterables(query, model, system_prompt, temperature, max_tokens):
            try:
                output = await self.send_requestmodel(
                    self.make_request_model(
                        query=q,
                        model=m,
                        system_prompt=sp,
                        temperature=t,
                        max_tokens=mt,
                        stream=stream
                    )
                )
                if self.config.print_output:
                    cc(f"Query: {q}", 'green')
                    print_model_response(output)
                    
                if self.config.save_dir:
                    await write_dict_to_json(output, self.config.save_dir , f'{datetime.now().isoformat()} {q}.json')
                    
                responses.append(output)
            except Exception as e:
                logger.error(f"Error processing query '{q}': {e}")
                

        return responses

    # @log_function_call
    async def send_requestmodel(self, request: RequestModel):
        request_byte = request.model_dump_json()
        return await self.tcp_client(request_byte)

    # @log_function_call
    async def tcp_client(self, message: str) -> dict:
        try:
            reader, writer = await asyncio.open_connection('127.0.0.1', self._PORT)

            writer.write(message.encode())
            await writer.drain()

            data = await reader.read(99999)

            writer.close()
            await writer.wait_closed()
            return json.loads(data.decode())
        
        except Exception as e:
            logger.error(f"Error in TCP client: {e}")

    # @log_function_call
    def make_request_model(self, query: str, **kwargs) -> RequestModel:
        return RequestModel(
            id=kwargs.get('id', str(uuid.uuid4())),
            query=query,
            model=kwargs.get('model', self.config.models[0]),
            system_prompt=kwargs.get('system_prompt', self.config.system_prompt),
            temperature=kwargs.get('temperature', self.config.temperature),
            max_tokens=kwargs.get('max_tokens', self.config.max_tokens),
            stream = kwargs.get('stream', self.config.stream)
        )