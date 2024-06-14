import argparse
import asyncio
import itertools
import json
import math
import os
import uuid
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Union

from icecream import ic
from playwright.async_api import BrowserContext, Page, Response, Route, async_playwright
from pydantic import BaseModel
from termcolor import colored
from typing_extensions import Annotated

from ..element_selectors import (
    CHAT_INPUT_SELECTOR,
    QUERY_SUBMIT_SELECTOR,
    SIGNIN_BUTTON_SELECTOR,
    SIGNIN_BUTTON_TEXT,
)
from ..groq_config import (
    DEFAULT_MODEL,
    GROQ_COOKIE_FILE,
    MODEL_LIST_FILE,
    URL,
    modelindex,
)
from ..logger import get_logger
from .agroq_utils import (
    file_exists,
    get_cookie,
    get_model_from_name,
    now,
    print_model_response,
    save_cookie,
    save_dict_to_json,
    write_json,
)

logger = get_logger(__name__)
# ic.disable()

class AgroqConfig(BaseModel):
    query_list: List[str] = None
    cookie_file: str = GROQ_COOKIE_FILE
    models: List[str] = [DEFAULT_MODEL]
    headless: bool = False
    save_dir: Union[str, Path] = None
    system_prompt: str = "Please try to provide useful, helpful and actionable answers."
    print_output: bool = True
    temperature: float = 0.1
    max_tokens: int = 2048
    n_workers: int = 2
    reset_login: bool = False
    server_model_configs: str = MODEL_LIST_FILE


class RequestModel(BaseModel):
    id: str = uuid.uuid4().hex
    query: str
    model: str = DEFAULT_MODEL
    system_prompt: str = "Please try to provide useful, helpful and actionable answers."
    temperature: float = 0.1
    max_tokens: int = 2048
    top_p: int = 1
    stream: bool = True
    

class ResponseModel(BaseModel):
    query: str = None
    response_text: str = None
    model: str = None
    tokens_per_second: float = 0
    stats: Dict[str, Any] = {}



class Agroq:


    def __init__(self, config: AgroqConfig):
        self.got_models = False
        self.config=config
        self.url = URL
        if self.config.reset_login:
            self.reset_login()


    def init(self):        
        self.cookie = self.get_cookie_or_login(url=URL, cookie_file=self.config.cookie_file)
        self.context = self.groq_context(cookie=self.cookie)
        self.request_queue = asyncio.Queue()
        self.response_queue = asyncio.Queue()

        
        
    def make_request_model(self, query: str, **kwargs) -> RequestModel:
        return RequestModel(
            query=query,
            model=kwargs.get('model',self.config.models[0]),
            system_prompt=kwargs.get('system_prompt',self.config.system_prompt),
            temperature=kwargs.get('temperature', self.config.temperature),
            max_tokens=kwargs.get('max_tokens', self.config.max_tokens),
        )

    def reset_login(self):
        try:
            os.remove(self.config.cookie_file)
        except Exception as e:
            logger.critical(
                f"exception while deleting cookie file, delete manually to reset login. {self.config.cookie_file} ",
                exc_info=e,
            )

    async def login_user(self) -> dict:
        with async_playwright() as p:
            # ic("in login_user func  : ", now())
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
            # ic("in login_user func  : ", now())
            await page.close()
        return cookie

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
            cookie = await self.login_user(url=url, cookie_file=cookie_file)
        return cookie
    
    @asynccontextmanager
    async def groq_context(
        self,
        cookie: list[dict] = None,
    ):

        async with async_playwright() as p:
            browser = await p.firefox.launch(headless=self.config.headless)
            context = await browser.new_context()

            if cookie:
                await context.add_cookies(cookie)
                logger.debug("Cookie loaded!!!")

            try:
                yield context
            finally:
                cookies = await context.cookies()
                if cookies:
                    await save_cookie(cookies, self.config.cookie_file)
                # await context.close()
                await browser.close()
                logger.debug("Browser closed!!!")

    
    async def do_query(self, page: Page, request_model: RequestModel, queue: asyncio.Queue) -> None:
        """Perform the query on the page."""
        # ic("in do_query func  : ", now())

        try:
            textarea = await page.wait_for_selector(CHAT_INPUT_SELECTOR)

            n_try = 20
            # Check if the textarea is disabled
            is_disabled = await textarea.is_disabled()

            while is_disabled:
                # print("chat is disabled!!")
                if n_try > 0:
                    await page.wait_for_timeout(100)
                    is_disabled = await textarea.is_disabled()

                    # print(n_try, " try...")
                    n_try -= 1
                else:
                    # print(f"breaking loop tries reached!!! try: {n_try}")
                    await self.queue.put(
                        ResponseModel(   
                            query=request_model.query,
                            response_text="chat is disabled, no query submitted",
                            model=request_model.model
                        )
                    )

                    return

            if not is_disabled:
                await textarea.fill(request_model.query)
                await page.locator(QUERY_SUBMIT_SELECTOR).click()

            return

        except Exception as e:
            logger.exception("Exception occurred: func - do_query ", exc_info=e)
            await self.queue.put(
                        ResponseModel(   
                            query=request_model.query,
                            response_text="chat is disabled, no query submitted",
                            model=request_model.model
                        )            
                    )

    async def worker_task(
        self,
        model: str,
        context: BrowserContext,
        queries: List[str],
        queue: asyncio.Queue,
        config: Any,
        reset_login: bool = False,
    ):
        """Worker task to process a subset of queries."""
        # ic("in worker_task func  : ", now())


        responses = []
        page = await self.context.new_page()

        partial_handle = partial(
            self.handle_chat_completions, request_model=None, queue=self.queue
        )

        # page.on('request', lambda x: print(colored(f">>>: {x.url} {x.method} {x.post_data_json}", "cyan")))
        # page.on('response', lambda x: print(colored(f"<<<: {x.url} {x.status}", "red")))

        # await page.route("**/**/static/**", abort_media)
        await page.route("**/**/*.woff2", lambda x: x.abort())
        await page.route("**/**/*.woff", lambda x: x.abort())
        await page.route("**/**/*.css", lambda x: x.abort())
        await page.route("**/**/web/metrics", lambda x: x.abort())

        await page.route("**/**/chat/completions", partial_handle)
        await page.route("**/**/models", self.get_models_from_api_response)

        request_model = RequestModel(**{
            "model": get_model_from_name(model),
            "query": None,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": 1,
            "system_prompt": config.system_prompt,
        })

        await page.goto(URL, timeout=60 * 1000)
        # print("start query: ", now())

        for query in queries:
            request_model.query = query

            partial_handle.keywords["request_model"] = request_model

            await self.do_query(page, request_model=request_model, queue=self.queue)

            response = await self.queue.get()
            responses.append(response)

            if self.config.print_output:
                print_model_response(response)

            if "rate_limit_exceeded" in response["response_text"]:
                await page.close()
                return responses
            if response["response_text"] == "timeout chat is disabled":
                await page.close()
                return responses

            if self.config.save_dir:
                await save_dict_to_json(response, self.config.save_dir, query)

        # await page.wait_for_timeout(1000*300) # to see the problem occuring
        await page.close()
        return responses

    async def _agroq(
        self,
        query_list: List[str],
        n_workers: int = 2,
        reset_login: bool = False,
    ) -> List[Dict[str, Any]]:
        """Main function to perform queries and process responses."""


        # Split the queries among the workers
        query_splits = [query_list[i::n_workers] for i in range(n_workers)]

        # repeating the model names
        self.config.model_list *= math.ceil(len(query_list) / len(self.config.model_list))


        # Create worker tasks
        worker_tasks = [
            self.worker_task(
                queries=query_splits[i],
                model=self.config.model_list[i],
                queue=self.queue,
                reset_login=reset_login,
            )
            for i in range(n_workers)
        ]

        # Run worker tasks concurrently
        responses = await asyncio.gather(*worker_tasks)
        responses = list(itertools.chain(*responses))
        return responses

    def query(self, query: str, system_prompt: str, temperature: float = 0.1, max_tokens: int = 2048, model: str = DEFAULT_MODEL ) -> List[Dict[str, Any]]:
        """Perform a single query."""
        
        # return self._agroq([query], n_workers=1)
        return self.make_request_model(query=query, model=model, system_prompt=system_prompt, temperature=temperature, max_tokens=max_tokens)
    
    async def get_models_from_api_response(self, route: Route):
        """Fetch and save models from API response."""
        if not self.got_models:
            if "/openai/v1/models" in route.request.url:
                response = await route.fetch()
                data = await response.json()
                await write_json(data, filename=MODEL_LIST_FILE)
                self.got_models = True

        await route.continue_()

    async def handle_response(self, response: Response):
        """Handle HTTP response."""
        # ic("in handle_response func  : ", now())

        try:
            if "chat/completions" in response.url:
                await self.handle_streamed_response(response)
        except Exception as e:
            logger.exception("Exception in handle_response", exc_info=e)

    async def handle_streamed_response(
        self, response: Response, query: str
    ) -> Dict[str, Any]:
        """Handle streamed response from the server."""
        # ic("in handle_streamed_response func  : ", now())
        accumulated_content = ""
        stats = {}
        model = None
        tokens_per_second = 0

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
                    error_json = error_json_.get("error")
                    accumulated_content += error_json.get("message", "Error")
                    stats = error_json_

                    if error_json.get("code") == "model_not_active":
                        accumulated_content += f"\nChoose Models from : {modelindex} "

                    elif error_json.get("code") == "rate_limit_exceeded":
                        accumulated_content += "\nWait kar le thoda."

                    else:
                        accumulated_content += "\nUnknown Error"

                    logger.error(
                        "Error while handling streamed response from server",
                        exc_info=error_json,
                    )

                else:
                    error = True
                    logger.error(
                        "Unknown Error while handling streamed response from server",
                        exc_info=body_str,
                    )
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
            return ResponseModel(**{
                "query": query,
                "response_text": accumulated_content,
                "stats": stats,
                "tokens_per_second": tokens_per_second,
                "model": model,
            })

    async def abort_images(self, route: Route):
        """Abort media requests to save bandwidth."""
        if route.request.resource_type == "image":
            await route.abort()
        else:
            await route.continue_()

    async def abort_fonts(self, route: Route):
        if route.request.url.endswith(".woff2") or route.request.url.endswith(".woff"):
            await route.abort()
        else:
            await route.continue_()

    async def abort_metics(self, route: Route):
        if route.request.url.endswith("/web/metrics"):
            await route.abort()
        else:
            await route.continue_()

    async def abort_css(self, route: Route):
        if route.request.url.endswith(".css"):
            await route.abort()
        else:
            await route.continue_()

    async def handle_chat_completions(self, route: Route, *args, **kwargs):
        """Handle chat completions route."""
        # ic("in handle_chat_completions func  : ", now())
        request = route.request

        request_model = kwargs.get("request_model", {})

        if request.method == "POST" and "chat/completions" in request.url:
            data = request.post_data_json
            if data:
                modified_data = {
                    "model": request_model.model,
                    "temperature": request_model.temperature,
                    "max_tokens": request_model.max_tokens,
                    "stream": request_model.stream,
                    "messages": [
                        {
                            "content": request_model.system_prompt,
                            "role": "system",
                        },
                        {
                            "content": request_model.query, 
                            "role": "user"
                        },
                    ],
                }
                # print("start post data : ", now())
                # print(colored(f"***** >>> this is going as post data, {modified_data}", "green"))
                await route.continue_(post_data=json.dumps(modified_data))
            else:
                await route.continue_()
        else:
            await route.continue_()

        response = await self.get_query_response(route, request_model.query)

        await self.queue.put(response)

    async def get_query_response(self, route: Route, query: str):
        """Fetch the response for the query."""
        # ic("in get_query_response func  : ", now())

        if "chat/completions" in route.request.url:
            response = await route.fetch()
            return await self.handle_streamed_response(response, query)

    async def get_page(self, context) -> Page:
        """Get a new page from the context."""
        return await context.new_page()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "queries",
        nargs="+",
        help="one or more quoted string like 'what is the purpose of life?' 'why everyone hates meg griffin?'",
    )
    parser.add_argument(
        "--model_list",
        nargs="+",
        default=DEFAULT_MODEL,
        help=f"Available models are {' '.join(modelindex)}, e.g enter as --model_list 'llama3-8b' 'gemma-7b' ",
    )
    parser.add_argument(
        "--cookie_file",
        type=str,
        default=GROQ_COOKIE_FILE,
        help="looks in current directory by default for groq_cookie.json. If not found, You will have to login when the browswer opens under 120 seconds. It's one time thing",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default="Please try to provide useful, helpful and actionable answers.",
        help="System prompt to be given to the llm model. Its like 'you are samuel l jackson as my assistant'. Default is None.",
    )
    parser.add_argument(
        "--headless", type=str, default="True", help="set true to not see the browser"
    )
    parser.add_argument(
        "--output_dir",
        default=os.getcwd(),
        help="Path to save the output file. Defaults to current working directory.",
    )

    parser.add_argument(
        "--temperature", type=float, default=0.1, help="Temperature value"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2048,
        help="Maximum number of tokens (upper limit)",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=2,
        help="Number of browers instances to work simultaneously. Keep between 1-8 (eats up ram)",
    )

    # a flag to reset the login
    parser.add_argument(
        "--reset_login",
        action="store_true",
        help="If true, will delete the groq_cookie.json file and login again",
    )

    args = parser.parse_args()

    def clean(x):
        return x.lower().strip() == "true"

    headless = clean(args.headless)

    out = Agroq().agroq(
        cookie_file=args.cookie_file,
        model_list=args.model_list,
        headless=headless,
        query_list=args.queries,
        save_dir=args.output_dir,
        system_prompt=args.system_prompt,
        print_output=True,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        n_workers=args.n_workers,
        reset_login=args.reset_login,
    )
