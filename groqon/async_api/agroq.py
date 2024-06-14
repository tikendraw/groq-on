import argparse
import asyncio
import json
import math
import itertools
import os
import re
from contextlib import asynccontextmanager
from datetime import datetime
from functools import partial, cache
from pprint import pprint
from pathlib import Path
from typing import Any, Dict, List, Union
from line_profiler import profile
import aiofiles
from icecream import ic
from playwright.async_api import Page, Response, Route, async_playwright
from termcolor import colored

from ..element_selectors import (
    QUERY_SUBMIT_SELECTOR,
    SIGNIN_BUTTON_SELECTOR,
    CHAT_INPUT_SELECTOR,
)
from ..groq_config import (
    DEFAULT_MODEL,
    GROQ_COOKIE_FILE,
    MODEL_LIST_FILE,
    URL,
    modelindex,
)
from ..logger import get_logger
from .agroq_utils import file_exists, get_cookie, save_cookie

logger = get_logger(__name__)
# ic.disable()
GOT_MODELS = False


def now() -> str:
    """Returns the current time as a string."""
    return datetime.now().strftime("%H:%M:%S")


# @profile
async def login_user(p: async_playwright, url: str, cookie_file: str) -> dict:
    # ic("in login_user func  : ", now())
    browser = await p.firefox.launch(headless=False)
    context = await browser.new_context()
    page = await context.new_page()

    await page.goto(url, timeout=60_000)

    sec = 100
    while sec > 0:
        await page.wait_for_timeout(1000)
        sec -= 1
        button_text = "Sign in to Groq"
        
        if 'groq.com' in page.url:
            try:
                button_text = page.locator(SIGNIN_BUTTON_SELECTOR)
                button_text = await button_text.inner_text()
            except Exception as e:
                logger.debug("Into google login page ", exc_info=e)
            
        if "sign in" not in button_text.lower().strip():
            # text says : Sign in to Groq
            sec = 0

    cookie = await context.cookies()
    await save_cookie(cookie, cookie_file)
    logger.debug("login page closed!")
    # ic("in login_user func  : ", now())
    await page.close()
    return cookie


# @profile
@asynccontextmanager
async def groq_context(
    cookie_file: str = GROQ_COOKIE_FILE,
    model: str = "llama3-8b",
    headless: bool = False,
    system_prompt: str = None,
):
    """Async context manager for Playwright context setup and teardown."""
    # ic("in groq_context func  : ", now())
    url = URL

    cookie = get_cookie(cookie_file) if file_exists(cookie_file) else None

    async with async_playwright() as p:
        if not cookie:
            logger.warning(
                f"""get_model_from_name
Cookie not loaded!!!
You have 100 seconds to login to groq.com, Make it quick!!! 
page will close in 100 seconds, Your time started: {now()}
just a one time thing.""",
            )
            cookie = await login_user(p=p, url=url, cookie_file=cookie_file)

        browser = await p.firefox.launch(headless=headless)
        context = await browser.new_context()

        if cookie:
            await context.add_cookies(cookie)
            logger.debug("Cookie loaded!!!")

        try:
            yield context
        finally:
            cookies = await context.cookies()
            if cookies:
                await save_cookie(cookies, cookie_file)
            # await context.close()
            await browser.close()
            logger.debug("Browser closed!!!")


# @profile
async def save_output(output_dict: Dict[str, Any], save_dir: str, file_name: str):
    """Save output dictionary to a JSON file."""

    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)

    file_name = "".join(c for c in file_name if c.isalnum() or c.isspace()).strip()
    file_name_split = file_name.split(" ")

    if len(file_name_split) > 10:
        file_name = " ".join(file_name_split[:10])

    json_file_path = save_dir_path / f"{file_name}.json"

    async with aiofiles.open(json_file_path, "w") as f:
        await f.write(json.dumps(output_dict, indent=4))


# @profile
async def do_query(page: Page, user_dict: dict, queue: asyncio.Queue) -> None:
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
                await queue.put(
                    {
                        "query": user_dict.get("query"),
                        "response_text": "chat is disabled, no query submitted",
                        "time_stats": {},
                        "tokens_per_second": 0,
                        "model": user_dict.get("model", DEFAULT_MODEL),
                    }
                )
                
                return

        if not is_disabled:
            await textarea.fill(user_dict.get("query"))
            await page.locator(QUERY_SUBMIT_SELECTOR).click()
            # await page.wait_for_event("response")
            # await page.screenshot(path="after_submit.png", full_page=True)

        return

    except Exception as e:
        logger.exception("Exception occurred: func - do_query ", exc_info=e)
        await queue.put(
            {
                "query": user_dict.get("query"),
                "response_text": "timeout chat is disabled",
                "time_stats": {},
                "tokens_per_second": 0,
                "model": user_dict.get("model", DEFAULT_MODEL),
            }
        )

@cache
#@profile
def get_model_from_name(model: str) -> str:
    """Get the full model name from the given partial name."""

    model = model.lower().strip()
    for model_name in modelindex:
        if model in model_name.lower().strip():
            return model_name

    print("Available models: ", modelindex)
    print("Using default model: ", DEFAULT_MODEL)
    return DEFAULT_MODEL


# @profile
async def worker_task(
    worker_id: int,
    queries: List[str],
    cookie_file: str,
    model: str,
    headless: bool,
    save_dir: Union[str, Path],
    system_prompt: str,
    print_output: bool,
    temperature: float,
    max_tokens: int,
    queue: asyncio.Queue,
    reset_login: bool = False,
):
    """Worker task to process a subset of queries."""
    # ic("in worker_task func  : ", now())

    if reset_login:
        try:
            os.remove(cookie_file)
        except Exception as e:
            logger.critical(f"exception while deleting cookie file, delete manually to reset login. {cookie_file} ", exc_info=e)


    responses = []
    async with groq_context(cookie_file=cookie_file, headless=headless) as context:
        page = await context.new_page()

        partial_handle = partial(handle_chat_completions, user_dict=None, queue=queue)

        # page.on('request', lambda x: print(colored(f">>>: {x.url} {x.method} {x.post_data_json}", "cyan")))
        # page.on('response', lambda x: print(colored(f"<<<: {x.url} {x.status}", "red")))

        # await page.route("**/**/static/**", abort_media)
        await page.route("**/**/*.woff2", lambda x: x.abort())
        await page.route("**/**/*.woff", lambda x: x.abort())
        await page.route("**/**/*.css", lambda x: x.abort())
        await page.route("**/**/web/metrics", lambda x: x.abort())

        await page.route("**/**/chat/completions", partial_handle)
        await page.route("**/**/models", get_models_from_api_response)

        original_user_dict = {
            "model": get_model_from_name(model),
            "query": None,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 1,
            "system_prompt": system_prompt,
        }

        await page.goto(URL, timeout=60 * 1000)
        # print("start query: ", now())

        for query in queries:
            user_dict = original_user_dict.copy()
            user_dict["query"] = query

            partial_handle.keywords["user_dict"] = user_dict

            await do_query(page, user_dict=user_dict, queue=queue)

            response = await queue.get()
            responses.append(response)

            if print_output:
                print_model_response(response)

            if "rate_limit_exceeded" in response["response_text"]:
                await page.close()
                return responses
            if "timeout chat is disabled" == response["response_text"]:
                await page.close()
                return responses

            if save_dir:
                await save_output(response, save_dir, query)

        # await page.wait_for_timeout(1000*300) # to see the problem occuring
        await page.close()
        return responses


# @profile
async def _agroq(
    query_list: List[str],
    cookie_file: str = GROQ_COOKIE_FILE,
    model_list: List[str] = "llama3-8b",
    headless: bool = False,
    save_dir: Union[str, Path] = None,
    system_prompt: str = "Please try to provide useful, helpful and actionable answers.",
    print_output: bool = True,
    temperature: float = 0.1,
    max_tokens: int = 2048,
    n_workers: int = 2,
    reset_login: bool = False,
) -> List[Dict[str, Any]]:
    """Main function to perform queries and process responses."""
    # ic("in _agroq func", now())

    if isinstance(query_list, str):
        query_list = [query_list]

    n_workers = min(n_workers, len(query_list))

    if isinstance(model_list, str):
        model_list = [model_list]

    # Split the queries among the workers
    query_splits = [query_list[i::n_workers] for i in range(n_workers)]

    # repeating the model names
    model_list *= math.ceil(len(query_list) / len(model_list))

    queue = asyncio.Queue()

    # Create worker tasks
    worker_tasks = [
        worker_task(
            worker_id=i,
            queries=query_splits[i],
            cookie_file=cookie_file,
            model=model_list[i],
            headless=headless,
            save_dir=save_dir,
            system_prompt=system_prompt,
            print_output=print_output,
            temperature=temperature,
            max_tokens=max_tokens,
            queue=queue,
            reset_login=reset_login,
        )
        for i in range(n_workers)
    ]

    # Run worker tasks concurrently
    responses = await asyncio.gather(*worker_tasks)
    responses = list(itertools.chain(*responses))
    return responses


# @profile
def agroq(
    query_list: List[str],
    cookie_file: str = GROQ_COOKIE_FILE,
    model_list: List[str] = "llama3-8b",
    headless: bool = False,
    save_dir: Union[str, Path] = None,
    system_prompt: str = "Please try to provide useful, helpful and actionable answers.",
    print_output: bool = True,
    temperature: float = 0.1,
    max_tokens: int = 2048,
    n_workers: int = 2,
    reset_login: bool = False,
) -> List[Dict[str, Any]]:
    """Main function to perform queries and process responses."""
    # ic("in agroq func", now())
    # print("start ", now())
    return asyncio.run(
        _agroq(
            query_list=query_list,
            cookie_file=cookie_file,
            model_list=model_list,
            headless=headless,
            save_dir=save_dir,
            system_prompt=system_prompt,
            print_output=print_output,
            temperature=temperature,
            max_tokens=max_tokens,
            n_workers=n_workers,
            reset_login=reset_login,
        )
    )


def print_model_response(model_response: Dict[str, Any]):
    """Print the model response."""
    print(colored(f"query : {model_response.get('query', '')}", "green"))
    print(
        colored(
            f"response : {model_response.get('response_text', 'response not found, try again')}",
            "yellow",
        )
    )
    print(
        colored(
            f"Model : {model_response.get('model', 0)}", "magenta"
        )
    )
    print(
        colored(
            f"Speed : {model_response.get('tokens_per_second', 0):.2f} T/s", "magenta"
        )
    )
    print()


# @profile
async def write_json(data: Union[Dict, str], filename: str):
    """Write dictionary or JSON string to a file."""

    async with aiofiles.open(filename, "w") as f:
        await f.write(json.dumps(data, indent=4))


# @profile
async def get_models_from_api_response(route: Route):
    """Fetch and save models from API response."""
    global GOT_MODELS
    if not GOT_MODELS:
        if "/openai/v1/models" in route.request.url:
            response = await route.fetch()
            data = await response.json()
            await write_json(data, filename=MODEL_LIST_FILE)
            GOT_MODELS = True

    await route.continue_()


# @profile
async def handle_response(response: Response):
    """Handle HTTP response."""
    # ic("in handle_response func  : ", now())

    try:
        if "chat/completions" in response.url:
            await handle_streamed_response(response)
    except Exception as e:
        logger.exception("Exception in handle_response", exc_info=e)


# @profile
async def handle_streamed_response(response: Response, query: str) -> Dict[str, Any]:
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
        return {
            "query": query,
            "response_text": accumulated_content,
            "time_stats": stats,
            "tokens_per_second": tokens_per_second,
            "model": model,
        }


def extract_rate_limit_info(data):
    json_str = json.dumps(data) if isinstance(data, dict) else data
    # Define regular expressions
    model_regex = r"model\s`(.*?)`"
    model_limit_regex = r"Limit\s(\d+)"
    wait_time_regex = r"try again in\s(\d+)ms"
    type_regex = r"type\":\s*\"(.*?)\""
    code_regex = r"code\":\s*\"(.*?)\""

    # Extract values using regular expressions
    model = re.search(model_regex, json_str)
    model_limit = re.search(model_limit_regex, json_str)
    wait_time = re.search(wait_time_regex, json_str)
    type_val = re.search(type_regex, json_str)
    code_val = re.search(code_regex, json_str)

    # Create a dictionary with extracted values
    return {
        "model": model.group(1) if model else None,
        "model_limit (RPM)": int(model_limit.group(1)) if model_limit else None,
        "wait_time (ms)": int(wait_time.group(1)) if wait_time else None,
        "type": type_val.group(1) if type_val else None,
        "code": code_val.group(1) if code_val else None,
    }


async def abort_images(route: Route):
    """Abort media requests to save bandwidth."""
    if route.request.resource_type == "image":
        await route.abort()
    else:
        await route.continue_()


async def abort_fonts(route: Route):
    if route.request.url.endswith(".woff2") or route.request.url.endswith(".woff"):
        await route.abort()
    else:
        await route.continue_()


async def abort_metics(route: Route):
    if route.request.url.endswith("/web/metrics"):
        await route.abort()
    else:
        await route.continue_()


async def abort_css(route: Route):
    if route.request.url.endswith(".css"):
        await route.abort()
    else:
        await route.continue_()


# @profile
async def handle_chat_completions(route: Route, *args, **kwargs):
    """Handle chat completions route."""
    # ic("in handle_chat_completions func  : ", now())
    request = route.request

    user_dict = kwargs.get("user_dict", {})
    queue = kwargs.get("queue")

    if not user_dict:
        raise ValueError("user_dict is required")

    if request.method == "POST" and "chat/completions" in request.url:
        data = request.post_data_json
        if data:
            modified_data = {
                "model": user_dict.get("model", DEFAULT_MODEL),
                "temperature": user_dict.get("temperature", 0.2),
                "max_tokens": user_dict.get("max_tokens", 2048),
                "stream": True,
                "messages": [
                    {
                        "content": user_dict.get(
                            "system_prompt",
                            "Please try to provide useful, helpful and actionable answers.",
                        ),
                        "role": "system",
                    },
                    {"content": user_dict.get("query", "hi"), "role": "user"},
                ],
            }
            # print("start post data : ", now())
            # print(colored(f"***** >>> this is going as post data, {modified_data}", "green"))
            await route.continue_(post_data=json.dumps(modified_data))
        else:
            await route.continue_()
    else:
        await route.continue_()

    response = await get_query_response(route, user_dict.get("query"))

    await queue.put(response)


# @profile
async def get_query_response(route: Route, query: str):
    """Fetch the response for the query."""
    # ic("in get_query_response func  : ", now())

    if "chat/completions" in route.request.url:
        response = await route.fetch()
        return await handle_streamed_response(response, query)


async def get_page(context) -> Page:
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

    out = agroq(
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
            reset_login=args.reset_login
        )
    
