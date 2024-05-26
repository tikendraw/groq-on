import argparse
import json
import os
from contextlib import contextmanager
from pathlib import Path

from playwright._impl._errors import TimeoutError
from playwright.sync_api import Page, sync_playwright

from ..element_selectors import (
    END_TEXT_SELECTOR,
    QUERY_INPUT_SELECTOR,
    QUERY_SUBMIT_SELECTOR,
)
from ..groq_config import modelindex, URL, GROQ_COOKIE_FILE
from .groq_utils import (
    check_element_and_get_text,
    clear_chat,
    file_exists,
    get_content_text,
    get_cookie,
    get_query_text,
    save_cookie,
    select_model,
    set_system_prompt,
)


@contextmanager
def groq_context(
    cookie_file: str = GROQ_COOKIE_FILE,
    model: str = "llama3-8b",
    headless: bool = False,
    system_prompt: str = None,
    reset_cookies=False
) -> Page:
    """
    Context manager for creating a Groq context.

    Args:
        cookie_file (str, optional): The path to the cookie file. Defaults to 'groq_cookie.json'.
        model (str, optional): The model to use. Defaults to 'llama3-70b'.
        headless (bool, optional): Whether to run the browser in headless mode. Defaults to False.
        system_prompt (str, optional): The system prompt to set. Defaults to None.

    Yields:
        Page: The page object for interacting with the browser.

    Raises:
        None
    """

    url = URL
    
    if reset_cookies:
        os.remove(GROQ_COOKIE_FILE)

    if file_exists(cookie_file):
        cookie = get_cookie(cookie_file)
    else:
        headless = False
        cookie = None

    with sync_playwright() as p:
        browser = p.firefox.launch(headless=headless)
        context = browser.new_context()
        page = context.new_page()

        if cookie:
            context.add_cookies(cookie)
            print("Cookie loaded!!!")
        else:
            print(
                "Cookie not loaded!!!",
                "You have 120 seconds to login to groq.com, Make it quick!!! HEADLESS is set to False, just a one time thing.",
            )

        page.goto(url, timeout=60_000) #wait_until="networkidle")

        if not cookie:
            page.wait_for_timeout(1000 * 120)  # 120 sec to login

        page.screenshot(path="screenshot1.png", full_page=True)

        # Set model
        select_model(page, model_choice=model)

        # Set system prompt
        if system_prompt:
            set_system_prompt(page, system_prompt=system_prompt)
            
        try:
            yield page
        finally:
            # Save cookie
            save_cookie(context.cookies(), cookie_file)
            page.close()
            browser.close()
            print("Browser closed!!!")


def get_groq_response(
    page: Page, query: str, save_output: bool = False, save_dir: str = os.getcwd()
) -> dict:
    """
    Retrieves the response to a given query using the specified page object.

    Args:
        page (Page): The page object for interacting with the browser.
        query (str): The query string to be filled in the input field.
        save_output (bool, optional): A flag indicating whether to save the output as a JSON file. Defaults to False.
        save_dir (str, optional): The directory path to save the output JSON file. Defaults to the current working directory.

    Returns:
        dict: A dictionary containing the extracted query, response content, and token/s value.
    """
    # Add query
    page.locator(QUERY_INPUT_SELECTOR).fill(query)
    # Submit query
    page.locator(QUERY_SUBMIT_SELECTOR).click()

    # Check if generation finished if not wait till end and get token/s
    try: 
        is_present, end_text = check_element_and_get_text(page, END_TEXT_SELECTOR)
    except TimeoutError as e:
        print("Couldn't find finish token, ",e)
        return
    
    if not is_present:
        print("Generation not finished!!!")
        page.screenshot(path="screenshot_generation_not_finished.png", full_page=True)
        return

    # Get query and text generated
    query_text, raw_query_html = get_query_text(page)
    response_content, raw_response_html = get_content_text(page)

    # Screenshot the content (for no reason)
    page.screenshot(path="screenshot2.png", full_page=True)
    clear_chat(page)

    output_dict = {
        "query": query_text,
        "response": response_content,
        # "query_html":raw_query_html,
        # "response_html":raw_response_html,
        "token/s": end_text,
    }
    if save_output:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        json_file = save_dir / f"{query_text}.json"

        with open(json_file, "w") as f:
            json.dump(output_dict, f)

    return output_dict


def groq(
    query_list: list[str],
    cookie_file: str = GROQ_COOKIE_FILE,
    model: str = "llama3-70b",
    headless: bool = False,
    save_output: bool = True,
    save_dir: str = os.getcwd(),
    system_prompt: str = None,
    print_output=True,
    reset_cookies=False
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

    with groq_context(
        cookie_file=cookie_file,
        model=model,
        headless=headless,
        system_prompt=system_prompt,
        reset_cookies=reset_cookies
    ) as page:
        response_list = []
        for query in query_list:
            response = get_groq_response(
                page, query, save_output=save_output, save_dir=save_dir
            )
            response_list.append(response)

            if print_output:
                print("Query", ":", response.get("query", None))
                print("Response", ":", response.get("response", None))
                print("token/s", ":", response.get("token/s", None))
        return response_list

                

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

    parser.add_argument(
        "--reset_cookies",
        action="store_true", 
        help="deletes the old cookie , need to login again"    
    )

    args = parser.parse_args()
    def check_headless(x):
        if x.lower().strip() == "true":
            return True
        else:
            return False


    headless = check_headless(args.headless)
    
    groq(
        cookie_file=args.cookie_file,
        model=args.model,
        headless=headless,
        query_list=args.queries,
        save_output=args.save_output,
        save_dir=args.output_dir,
        system_prompt=args.system_prompt,
        print_output=True,
        reset_cookie=args.reset_cookies
    )


