import argparse
import asyncio
from asyncio import QueueEmpty
import itertools
import json
import math
import os
import uuid
from contextlib import asynccontextmanager
from line_profiler import profile
from typing import Any, Dict, List
import time
from datetime import datetime
from icecream import ic
from playwright.async_api import BrowserContext, Page, Response, Route, async_playwright, Request
from termcolor import colored

from ..element_selectors import (
    CHAT_INPUT_SELECTOR,
    QUERY_SUBMIT_SELECTOR,
    SIGNIN_BUTTON_SELECTOR,
    SIGNIN_BUTTON_TEXT,
)
from ..groq_config import (
    URL,
    modelindex,
)
from .schema import APIRequestModel, RequestModel, ResponseModel, AgroqConfig
from ..logger import get_logger
from .agroq_utils import (
    file_exists,
    get_cookie,
    get_model_from_name,
    now,
    print_model_response,
    save_cookie,
    save_dict_to_json,
)

logger = get_logger(__name__)
ic.disable()


def print_function_name_in_red(func):
    def wrapper(*args, **kwargs):
        start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        print(colored(f"{func.__name__} started at {start_time}", 'red'))
        result = func(*args, **kwargs)
        return result
    return wrapper


def cc(x, color):
    print(colored(x,color))
    
class Agroq:
    def __init__(self, config: AgroqConfig):
        self.got_models = False
        self.config = config
        self.url = URL
        if self.config.reset_login:
            self.reset_login()
        self.cookie = None
        self.context = None
        self.request_queue = asyncio.Queue()
        self.response_queue = asyncio.Queue()
        self.output_queue = asyncio.Queue()

        self.task = None
        self.loop = asyncio.get_event_loop()
        self.hehe=0

 
    @print_function_name_in_red
    @profile
    def make_request_model(self, query: str, **kwargs) -> RequestModel:
        return RequestModel(
            id=str(datetime.now()),
            query=query,
            model=kwargs.get('model', self.config.models[0]),
            system_prompt=kwargs.get('system_prompt', self.config.system_prompt),
            temperature=kwargs.get('temperature', self.config.temperature),
            max_tokens=kwargs.get('max_tokens', self.config.max_tokens),
            
        )
        
    @print_function_name_in_red
    @profile
    def reset_login(self):
        try:
            os.remove(self.config.cookie_file)
        except Exception as e:
            logger.critical(
                f"exception while deleting cookie file, delete manually to reset login. {self.config.cookie_file} ",
                exc_info=e,
            )
        
    @print_function_name_in_red
    @profile
    async def login_user(self) -> list:
        async with async_playwright() as p:
            browser = await p.firefox.launch(headless=False)
            context = await browser.new_context()
            page = await context.new_page()

            await page.goto(self.url, timeout=60_000)

            sec = 100
            while sec > 0:
                await page.wait_for_timeout(1000)
                sec -= 1
                button_text = "Sign in to Groq"

                if "groq.com" in page.url:
                    try:
                        button_text = page.locator(SIGNIN_BUTTON_SELECTOR)
                        button_text = await button_text.inner_text()
                    except Exception as e:
                        logger.debug("Into google login page ", exc_info=e)

                if (
                    SIGNIN_BUTTON_TEXT.lower().strip()
                    not in button_text.lower().strip()
                ):
                    sec = 0

            cookie = await context.cookies()
            logger.debug("login page closed!")
            await page.close()
        return cookie
        
    @print_function_name_in_red
    @profile
    async def get_cookie_or_login(self, url: str, cookie_file: str) -> List[Dict]:
        cookie = get_cookie(cookie_file) if file_exists(cookie_file) else None
        if not cookie:
            logger.warning(
                f"""
    Cookie not loaded!!!
    LOGIN REQUIRED!!!
    You have 100 seconds to login to groq.com, Make it quick!!! 
    page will close in 100 seconds, Your time started: {now()}
    .""",
            )
            cookie = await self.login_user()
            await save_cookie(cookie, cookie_file)
        return cookie
            
    # @print_function_name_in_red
    # @asynccontextmanager
    @profile
    async def groq_context(self, cookie: list[dict] = None):
        
        async with async_playwright() as p:
            browser = await p.firefox.launch(headless=self.config.headless)
            context = await browser.new_context()

            if cookie:
                await context.add_cookies(cookie)
                logger.debug("Cookie loaded!!!")


            try: 
                yield context
            except Exception as e:
                raise e
            
            finally:                
                cc('finally started', 'green')
                cookies = await context.cookies()

                if cookies:
                    await save_cookie(cookies, self.config.cookie_file)
                    cc('cookies saved', 'green')

                await self.response_queue.join()
                
                # time.sleep(10)
                await browser.close()
                logger.debug("Browser closed!!!")

        
    async def process_queries(self, context:BrowserContext):
        while self.config.keep_running:
            # get all the queries
            requests_models = []
            while not self.request_queue.empty():
                cc('in while loop', 'red')
                requests_models.append(await self.request_queue.get())
                self.request_queue.task_done()
            
            
            # if len(requests_models) > 0:
            #     query_splits = [requests_models[i::self.config.n_workers] for i in range(self.config.n_workers)]

            worker_tasks = []
            for query_split in query_splits:
                worker_task = asyncio.create_task(self.worker_task(context=context, request_models=query_split))
                worker_tasks.append(worker_task)
                
            await asyncio.gather(*worker_tasks)
            cc('after asyncio gather', 'blue')

        
    @print_function_name_in_red
    @profile
    async def do_query(self, page: Page, request_model: RequestModel, ) -> None:
        try:
            textarea = await page.wait_for_selector(CHAT_INPUT_SELECTOR)
            n_try = 20
            is_disabled = await textarea.is_disabled()

            while is_disabled:
                if n_try > 0:
                    await page.wait_for_timeout(100)
                    is_disabled = await textarea.is_disabled()
                    n_try -= 1
                else:
                    raise TimeoutError(f"Chat is temporarily disabled, Query: ({request_model.query}) not submitted.")
                    break
            if not is_disabled:
                
                await textarea.fill('a')
                await page.locator(QUERY_SUBMIT_SELECTOR).click()

        except Exception as e:
            logger.exception("Exception occurred: func - do_query ", exc_info=e)
            await self.response_queue.put(
                ResponseModel(
                    id=request_model.id,
                    query=request_model.query,
                    response_text="chat is disabled, no query submitted",
                    model=request_model.model
                )
            )
            

    @print_function_name_in_red
    @profile
    async def worker_task(self, context: BrowserContext, request_models: List[RequestModel]):

        page = await context.new_page()

        await page.route("**/**/*.woff2", self.abort_fonts)
        await page.route("**/**/*.woff", self.abort_fonts)
        await page.route("**/**/*.css", self.abort_css)
        await page.route("**/**/web/metrics", self.abort_metrics)

        cc('opening tab...', 'blue')
        await page.goto(URL, timeout=60 * 1000)
        
        for request_model in request_models:
            cc(f'Quering {request_model.query}...', 'green')
            
            async def wrapper_handle_chat_completions(route: Route, request: Request):
                await self.handle_chat_completions(route, request, request_model, self.response_queue)

            # Update route with the new wrapper function
            await page.route("**/**/chat/completions", wrapper_handle_chat_completions)

            await self.do_query(page, request_model=request_model)
            
            response = await self.response_queue.get() # check if any response in queue

            if response:
                print_model_response(response)
                await self.output_queue.put(response)
            self.response_queue.task_done()

        cc('exiting worker_task...', 'blue')
        # await page.wait_for_timeout(1000*30)

                
    @print_function_name_in_red
    @profile
    async def get_response_model_from_id(self, request_model_id: str) -> ResponseModel:
        for response in self.response_queue:
            if response.id == request_model_id:
                return response
        return None

    @profile
    async def multi_query_async(
        self,
        query_list: List[str],
        model_list: List[str], 
        save: bool = False,
        system_prompt_list: List[str] = None, 
        temperature_list:list[float]=None,
        max_tokens_list:list[int]=None
        ) -> Dict[str, Any]:
        
        if system_prompt_list is None:
            system_prompt_list = [self.config.system_prompt]
        if temperature_list is None:
            temperature_list = [self.config.temperature]
        if max_tokens_list is None:
            max_tokens_list = [self.config.max_tokens]
        
        model_list      *= math.ceil(len(query_list) / len(model_list))
        system_prompt_list   *= math.ceil(len(query_list) / len(system_prompt_list))
        temperature_list     *= math.ceil(len(query_list) / len(temperature_list))
        max_tokens_list      *= math.ceil(len(query_list) / len(max_tokens_list))
        
        for q, m, sp, t, mt in zip(query_list, model_list, system_prompt_list, temperature_list, max_tokens_list):
            self.request_queue.put_nowait(
                self.make_request_model(
                    query=q,
                    model=get_model_from_name(m),
                    system_prompt=sp,
                    temperature=t,
                    max_tokens=mt
                    )
                )
        # self.request_queue.put_nowait(None)
        cookie = await self.get_cookie_or_login(self.url, self.config.cookie_file)
        print('len cookie: ',len(cookie))
        await self.groq_context(cookie=cookie)   
        
        responses = []
        while not self.output_queue.empty():
            try:
                responses.append( self.output_queue.get_nowait())
            except QueueEmpty:
                return responses
        return responses
        
    # @print_function_name_in_red
    @profile
    async def handle_chat_completions(self, route: Route, request:Request, request_model: RequestModel, queue: asyncio.Queue) -> None:
        cc("in handle_chat_completions", "green")
        request = route.request

        if request.method == "POST":
            cc('method is post', 'blue')
            data = request.post_data_json
            
            if data:
                
                request_model.add_message(
                    content=request_model.system_prompt,
                    role="system",
                )
                
                request_model.add_message(
                    content=request_model.query,
                    role="user",
                )
                
                api_payload = APIRequestModel.from_request_model(request_model)
                api_payload_dict = api_payload.model_dump_json()

                cc(api_payload_dict, 'green')
                await route.continue_(
                    post_data=api_payload_dict,
                    # headers={**route.request.headers, 'Content-Type': 'application/json'},
                )
            else:
                route.continue_()

        else:
            route.continue_()
            
        response = await route.fetch()
        response = await self.handle_streamed_response(response, request_model.query)
            
        await queue.put(
            ResponseModel(
                id=request_model.id,
                **response                
                )
            )
        cc('out handle_chat_completions', 'green')

    
    async def abort_fonts(self, route: Route):
        await route.abort()

    async def abort_css(self, route: Route):
        await route.abort()

    async def abort_metrics(self, route: Route):
        await route.abort()
        
    @print_function_name_in_red
    async def get_models_from_api_response(self, route: Route, *args, **kwargs):
        response = await route.fetch()
        json_response = await response.json()

        if not self.got_models:
            await save_dict_to_json(json_response, self.config.server_model_configs, f"server_models_{now()}")
            self.got_models = True
        await route.continue_()

    @profile
    def multi_query(self, query_list: List[str], save: bool = False, return_dict:bool=True, **kwargs) -> Dict[str, Any]:
        responses = asyncio.run(self.multi_query_async(query_list, model_list=kwargs.get('model_list')))
                    
        if return_dict:
            return [response.model_dump() for response in responses]
        else:
            return responses
    
    @profile
    async def handle_streamed_response(self, response: Response, query: str) -> Dict[str, Any]:
        """Handle streamed response from the server."""
        # ic("in handle_streamed_response func  : ", now())
        accumulated_content = ""
        stats = {}
        model = None
        tokens_per_second = 0
        body_str=''
        status_code=response.status
        try:
            body_bytes = await response.body()
            body_str = body_bytes.decode("utf-8")
            
            lines = body_str.split("\n\n")

            error = False
            for line in lines:
                if line.strip() == "data: [DONE]":
                    break
                if line.startswith("data: "):
                    json_content = line[len("data: ") :]
                    try:
                        chunk = json.loads(json_content)
                        accumulated_content += chunk["choices"][0]["delta"].get(
                            "content", ""
                        )
                        model = chunk.get("model")

                        if "x_groq" in chunk:
                            stats = chunk["x_groq"].get("usage", {})
                    except json.JSONDecodeError as e:
                        logger.exception(
                            colored(f"Failed to decode JSON: {e}", "red"), exc_info=True
                        )
                elif line.startswith('{"error"'):
                    error = True

                    error_json_ = json.loads(line)
                    error_json = error_json_.get('error')
                    accumulated_content += error_json.get("message", "Error")
                    stats = error_json_

                    if error_json.get("code") == "model_not_active":
                        accumulated_content += f"\nChoose Models from : {modelindex} "
                        
                    elif error_json.get('code') == 'rate_limit_exceeded':
                        accumulated_content += "\nWait kar le thoda."
                        
                    else:   
                        accumulated_content += "\nUnknown Error"


                    logger.error("Error while handling streamed response from server", exc_info=error_json)
                    
                else:
                    error = True
                    logger.error("Unknown Error while handling streamed response from server", exc_info=body_str )
                    accumulated_content += "\n" + str(line)

            if stats and not error:
                try:
                    tokens_per_second = (
                        stats["completion_tokens"] / stats["completion_time"]
                    )
                except Exception as e:
                    logger.error("failed to calculate tokens per second", exc_info=e)

        except Exception as e:
            logger.exception("Exception in handle_streamed_response", exc_info=e)

        finally:
            out= {
                "query": query,
                "response_text": accumulated_content,
                "stats": stats,
                "tokens_per_second": tokens_per_second,
                "model": model,
                "raw_response": body_str,
                "status_code": status_code
            }
            # cc(out, 'green')
            return out

        #)

