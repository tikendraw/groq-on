# groq.py

import argparse
import json
from termcolor import colored
import asyncio
from typing import List, Optional, Union
import datetime
from pprint import pprint
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from printt import printt
from playwright.async_api import async_playwright, TimeoutError, Page, Route, Response, Request
from ..utils import get_current_time_str
from ..element_selectors import (
    END_TEXT_SELECTOR,
    QUERY_INPUT_SELECTOR,
    QUERY_SUBMIT_SELECTOR,
    COPY_BUTTON_SELECTOR
)
from ..groq_config import modelindex, URL, GROQ_COOKIE_FILE, API_URL, MODEL_JSON_FILE
from .agroq_utils import (
    check_element_and_get_text,
    clear_chat,
    file_exists,
    get_content_text,
    get_cookie,
    get_query_text,
    save_cookie,
    set_system_prompt,
    get_model_name
)
from ..logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def groq_context(
    cookie_file: str = GROQ_COOKIE_FILE,
    model: str = "llama3-8b",
    headless: bool = False,
    system_prompt: str = None,
):
    """
    Context manager for creating a Groq context.

    Args:
        cookie_file (str, optional): The path to the cookie file. Defaults to 'groq_cookie.json'.
        model (str, optional): The model to use. Defaults to 'llama3-70b'.
        headless (bool, optional): Whether to run the browser in headless mode. Defaults to False.
        system_prompt (str, optional): The system prompt to set. Defaults to None.

    Yields:
        BrowserContext: The browser context object for interacting with the browser.

    Raises:
        None
    """

    url = URL

    if file_exists(cookie_file):
        cookie = get_cookie(cookie_file)
    else:
        headless = False
        cookie = None

    async with async_playwright() as p:
        browser = await p.firefox.launch(headless=headless)
        context = await browser.new_context()
        
        

        if cookie:
            await context.add_cookies(cookie)
            logger.debug("Cookie loaded!!!")
        else:
            logger.warning(
                "Cookie not loaded!!!",
                "You have 120 seconds to login to groq.com, Make it quick!!! page will close in 120 seconds,  HEADLESS is set to False, just a one time thing.",
            )
            page = await context.new_page()
            await page.goto(url, timeout=60_000)  # wait_until="networkidle"
            await page.wait_for_timeout(1000 * 120)  # 120 sec to login
            await page.close()
            logger.debug("login page closed!")

            
        try:
            yield context

        finally:
            # Save cookie if not None
            cookies = await context.cookies()
            if cookies:
                await save_cookie(cookies, cookie_file)
            await context.close()
            await browser.close()
            logger.debug("Browser closed!!!")
            
async def do_query(
    page:Page, 
    query: str, 
) -> Page:
    """
    Retrieves the response to a given query using the specified page object.

    Args:
        page (Page): The page object for interacting with the browser.
        query (str): The query string to be filled in the input field.

    Returns:
        page (Page): the used page object.

    """

    # Add query
    await page.locator(QUERY_INPUT_SELECTOR).fill(query)
    # Submit query
    await page.locator(QUERY_SUBMIT_SELECTOR).click()
    

    # Check if generation finished if not wait till end and get token/s
    async def is_generation_finished(page):
        x,y = await check_element_and_get_text(page, END_TEXT_SELECTOR)
        return x, y

    try:
        is_present, token_count = await is_generation_finished(page)
        logger.debug('Generation complete!')
    except TimeoutError as e:
        logger.exception("Couldn't find finish token, ", exc_info=e)
        return

    if not is_present:
        screenshot_name = f"{get_current_time_str}-screenshot_generation_not_finished.png"
        screenshot_save_dir = Path('.').parent.parent /screenshot_name
        
        await page.screenshot(path=screenshot_save_dir, full_page=True)
        logger.warning(f"Generation not finished!!!, screenshot: {screenshot_save_dir}")
        return

    # Get query and text generated
    query_text, raw_query_html = await get_query_text(page)
    response_content, raw_response_html = await get_content_text(page)

    # Screenshot the content (for no reason)
    # await page.screenshot(path=f"./saved_responses/{get_current_time_str()}-screenshot2.png", full_page=True)
    await clear_chat(page)
    

    output_dict = {
        "query": query_text,
        "response": response_content,
        "token/s": token_count,
    }
            
    return output_dict, page


def save_output(save_dir: str, query_text: str, output_dict: dict, save_output: bool = False):
    if save_output:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        json_file = save_dir / f"{query_text}.json"
        with open(json_file, "w") as f:
            json.dump(output_dict, f)
        logger.debug(f"output saved! {json_file}")

async def mocking_api(
    page:Page, 
    query: str, 
    save_output: bool = False, 
    save_dir: str = os.getcwd(),
    model: str = "llama3-70b",
    system_prompt: str = None
    ):
    async def handle(route: Route):
        response = await route.fetch()
        json = await response.json()
        json.append({ "name": "Loquat", "id": 100})
        # Fulfill using the original response, while patching the response body
        # with the given JSON object.
        await route.fulfill(response=response, json=json)

        await page.route("https://api.groq.com/openai/v1/chat/completions", handle)

async def agroq(
    query_list: list[str],
    cookie_file: str = GROQ_COOKIE_FILE,
    model: str = "llama3-70b",
    headless: bool = False,
    save_output: bool = True,
    save_dir: str = os.getcwd(),
    system_prompt: str = None,
    print_output=True,
) -> list[dict]:
    """
    Executes a list of Groq queries and returns a list of dictionaries containing the query, response, and token/s value.

    Args:
        query_list (list[str]): A list of Groq queries to execute.
        cookie_file (str, optional): The path to the cookie file. Defaults to 'groq_cookie.json'.
        model (str, optional): The model to use. Defaults to 'llama3-70b'.
        headless (bool, optional): Whether to run the browser in headless mode. Defaults to False.
        save_output (bool, optional): A flag indicating whether to save the output as a JSON file. Defaults to True.
        save_dir (str, optional): The directory path to save the output JSON file. Defaults to the current working directory.
        print_output (bool, optional): A flag indicating whether to print the query, response, and token/s value. Defaults to True.
        system_prompt (str, optional): The system prompt to set. Defaults to None.

    Returns:
        list[dict]: A list of dictionaries containing the query, response, and token/s value for each executed query.
    """

    if isinstance(query_list, str):
        query_list = [query_list]

    async with groq_context(
        cookie_file=cookie_file,
        headless=headless,
    ) as context:
        response_list = []
        page = await get_response(context)

        for query in query_list:
            response, page = await do_query(
                page, query, save_output=save_output, save_dir=save_dir, model=model, system_prompt=system_prompt
            )
            response_list.append(response)

            if print_output:
                print("Query", ":", response.get("query", None))
                print("Response", ":", response.get("response", None))
                print("token/s", ":", response.get("token/s", None))
            
            
        await page.close()
        return response_list



async def write_models_to_json(models, filename=MODEL_JSON_FILE):
    with open(filename, 'w') as f:
        json.dump(models, f)
        
    
async def get_models(route:Route):
    if "/openai/v1/models" in route.request.url:
        response = await route.fetch()
        jsonn = await response.json()
        # save the json 
        await write_models_to_json(jsonn)
        await route.continue_()
    else:
        await route.continue_()
    

async def post_query_data(route:Route):
    
    if 'chat/completions' in route.request.url and route.request.method == 'POST':
        print("POST RAN*********************")
        request = route.request
        data = request.post_data_json  # Get existing payload (if POST request)
        if data:
            modified_data = data
            # Modify the data here (e.g., add/remove fields, change values)
            # modified_data["new_field"] = "new_value"
            print(colored(f'++++++: {request.url} {modified_data}', 'yellow'))
            await route.continue_(post_data=json.dumps(modified_data))
    else:
        # print("ELSE RAN*********************")
        await route.continue_()  # Don't modify non-POST requests
  
  
async def handle_streamed_response(response: Response):
    """
    Handle the streamed JSON data from the response.
    Collect and process the chunks to form the complete response.
    """
    body_bytes = await response.body()
    body_str = body_bytes.decode('utf-8')
    lines = body_str.split("\n\n")

    accumulated_content = ""
    stats=None,

    for line in lines:
        if line.strip() == "data: [DONE]":
            break
        if line.startswith("data: "):
            json_content = line[len("data: "):]
            try:
                chunk = json.loads(json_content)
                accumulated_content += chunk['choices'][0]['delta'].get('content', '')

                # Update stats
                if "x_groq" in chunk:
                    stats = chunk["x_groq"].get('usage', {})
                    
            except json.JSONDecodeError as e:
                logger.exception(colored(f"Failed to decode JSON: {e}", 'red'),exc_info=True)

    
    if stats:
        try:
            tokenspersecond = stats['completion_tokens'] / stats['completion_time']
        except Exception as e:
            tokenspersecond = 0

    reponse= {
        "response_text": accumulated_content,
        "time_stats": stats,
        "tokenspersecond": tokenspersecond
    }

    pprint(reponse, indent=4)
    return reponse
    





async def abort_media(route:Route):
    if route.request.resource_type == 'image':
        await route.abort()
    elif route.request.url.endswith(".woff2") or route.request.url.endswith(".woff"):
        await route.abort()
    else:
        await route.continue_()


async def get_query_response(route:Route):
    if 'chat/completions' in route.request.url:
        response = await route.fetch()
        response=await handle_streamed_response(response)
    
    await route.continue_()
    
    return response
    
        
async def get_response(context, url=URL) -> Page:

    page = await context.new_page()

    # page.on('request',  lambda  request: print(colored(f'<<<<<: {request.url} {request.method}', 'red')))
    # page.on('response', lambda response: print(colored(f'<<<<<: {response.url} {response.status}', 'green')))

    await page.route("**/**/chat/completions", post_query_data)
    await page.route("**/**/static/**", abort_media)
    await page.route("**/**/models", get_models)
    response = await page.route("**/**/chat/completions", get_query_response)
    # await page.route("**/**/chat/completions", get_query_response)


    await page.goto(url, timeout=60*1000)

    return page


    
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "queries",
        nargs="+",
        help="one or more quoted string like 'what is the purpose of life?' 'why everyone hates meg griffin?'",
    )
    parser.add_argument(
        "--model",
        default="llama3-70b",
        help=f"Available models are {' '.join(modelindex.keys())}",
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
        help="System prompt to be given to the llm model. Its like 'you are samuel l jackson as my assistant'. Default is None.",
    )
    parser.add_argument(
        "--headless", 
        type=str,
        default='True',
        help="set true to not see the browser"
    )
    parser.add_argument(
        "--save_output",
        action="store_true",
        help="set true to save the groq output with its query name.json",
    )
    parser.add_argument(
        "--output_dir",
        default=os.getcwd(),
        help="Path to save the output file. Defaults to current working directory.",
    )

    args = parser.parse_args()
    
    def check_headless(x):
        if x.lower().strip() == "true":
            return True
        else:
            return False


    headless = check_headless(args.headless)
    

    import asyncio
    asyncio.run(
        agroq(
            cookie_file=args.cookie_file,
            model=args.model,
            headless=headless,
            query_list=args.queries,
            save_output=args.save_output,
            save_dir=args.output_dir,
            system_prompt=args.system_prompt,
            print_output=True,
        )
    )

