import asyncio
import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel

from ..groq_config import ENDTOKEN, PORT, modelindex
from ..logger import get_logger
from ..utils import cc, log_function_call
from ..utils import get_current_time_str as now
from .agroq_utils import (
    async_generator_from_iterables,
    get_model_from_name,
    print_model_response,
    write_dict_to_json,
    x_eq_len_of_y,
)
from .schema import AgroqClientConfig, APIRequestModel

# from line_profiler import profile
logger = get_logger(__name__)


class AgroqClient(BaseModel):
    """Checkout this url to get help with model selection based on tasks: https://artificialanalysis.ai/providers/groq """
    config: AgroqClientConfig
    _PORT: Optional[int] = PORT

    @log_function_call
    async def multi_query_async(
        self,
        query: Union[str, List[str]]=None,
        model: Union[str, List[str]] = None, 
        system_prompt: Union[str, List[str]] = None, 
        temperature: Union[float, List[float]] = None,
        max_tokens: Union[int, List[int]] = None,
        top_p: Union[int, List[int]] = None,
        stream: bool = True,
        stop_server:bool = False
    ) -> list[Dict[str, Any]]:

        if query is None and stop_server:
            query = [ENDTOKEN]
                        
        # Convert single string inputs to lists
        query = [query] if isinstance(query, str) else query
        model = [model] if isinstance(model, str) else (model or self.config.models or modelindex)
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


        responses = await self.make_request(
            query=query,
            model=model,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            save_dir=self.config.save_dir,
            print_output=self.config.print_output
        )
        
        if stop_server:
            _ = await self.make_request(
                query=[ENDTOKEN],
                model=model,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream
            )
            return 
            
        return responses


    async def make_request(self, query, model, system_prompt, temperature, max_tokens, stream=True, save_dir=None, print_output=False):
        async def process_single_request(q, m, sp, t, mt):
            try:
                request_model = self.make_request_model(
                    query=q,
                    model=m,
                    system_prompt=sp,
                    temperature=t,
                    max_tokens=mt,
                    stream=stream
                )
                output = await self.send_requestmodel(request_model)
                
                if print_output:
                    print_model_response(output)
                
                if save_dir:
                    await write_dict_to_json(output, save_dir, f'{datetime.now().isoformat()} {q}.json')
                
                return output
            except Exception as e:
                logger.error(f"Error processing query '{q}': {e}")
                return {"error": str(e), "query": q}

        request_params = zip(query, model, system_prompt, temperature, max_tokens)
        tasks = [process_single_request(q, m, sp, t, mt) for q, m, sp, t, mt in request_params]
        
        responses = await asyncio.gather(*tasks)
        return [r for r in responses if r is not None]
    
    @log_function_call
    async def send_requestmodel(self, request: APIRequestModel):
        request_byte = request.model_dump_json()
        return await self.tcp_client(request_byte)

    @log_function_call
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
            raise e

    @log_function_call
    def make_request_model(self, query: str, **kwargs) -> APIRequestModel:
        return APIRequestModel(
            local_id=kwargs.get('id', str(uuid.uuid4())),
            query=query,
            model=kwargs.get('model', self.config.models[0]),
            system_prompt=kwargs.get('system_prompt', self.config.system_prompt),
            temperature=kwargs.get('temperature', self.config.temperature),
            max_tokens=kwargs.get('max_tokens', self.config.max_tokens),
            stream = kwargs.get('stream', self.config.stream)
        )