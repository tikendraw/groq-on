import asyncio
import json
import uuid
from datetime import datetime
from typing import Any, Dict, List

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


class AgroqClient:
    """Checkout this url to get help with model selection based on tasks: https://artificialanalysis.ai/providers/groq """
    def __init__(self, config: AgroqClientConfig):
        self.config = config
        self._PORT = PORT

    # @log_function_call
    async def multi_query_async(
        self,
        query_list: List[str],
        model_list: List[str] = None, 
        save: bool = False,
        system_prompt_list: List[str] = None, 
        temperature_list: List[float] = None,
        max_tokens_list: List[int] = None,
        stream: bool = True,
    ) -> list[Dict[str, Any]]:

        if model_list is None:
            model_list = modelindex
        if system_prompt_list is None:
            system_prompt_list = [self.config.system_prompt]
        if temperature_list is None:
            temperature_list = [self.config.temperature]
        if max_tokens_list is None:
            max_tokens_list = [self.config.max_tokens]
        if isinstance(model_list, str):
            model_list = [model_list]
            
        stream = [stream]* len(query_list)

        model_list = [get_model_from_name(name) for name in model_list]

        model_list = x_eq_len_of_y(x=model_list, y=query_list)
        system_prompt_list = x_eq_len_of_y(x=system_prompt_list, y=query_list)
        temperature_list = x_eq_len_of_y(x=temperature_list, y=query_list)
        max_tokens_list = x_eq_len_of_y(x=max_tokens_list, y=query_list)
        print(f'got {len(query_list)} queries')


        responses = []


        async for q, m, sp, t, mt, st in async_generator_from_iterables(query_list, model_list, system_prompt_list, temperature_list, max_tokens_list, stream):
            try:
                output = await self.send_requestmodel(
                    self.make_request_model(
                        query=q,
                        model=m,
                        system_prompt=sp,
                        temperature=t,
                        max_tokens=mt,
                        stream=st
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