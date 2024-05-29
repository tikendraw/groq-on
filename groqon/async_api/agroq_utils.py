# groq_utils.py

import json
import re
import time
from html import unescape
from pathlib import Path
from typing import Any
from printt import printt, disable_printt

disable_printt()
from playwright.async_api import Page
from tqdm import tqdm

from ..element_selectors import (
    CLEAR_CHAT_BUTTON,
    END_TEXT_SELECTOR,
    MODEL_DROPDOWN_SELECTOR,
    QUERY_INPUT_SELECTOR,
    SYSTEM_PROMPT_BUTTON,
    SYSTEM_PROMPT_TEXTAREA,
    QUERY_SELECTOR,
    RESPONSE_SELECTOR,
    MODEL_NAME_SELECTOR,
)
from ..groq_config import modelindex
from ..parsing import extract_to_markdown
from ..logger import get_logger

logger = get_logger(__name__)


async def get_text_by_selector(page: Page, selector) -> str:
    """
    Retrieves the text content from the element specified by the given selector on the given page.

    Args:
        page (Page): The page object for interacting with the browser.
        selector (str): The CSS selector of the element to retrieve the text from.

    Returns:
        tuple: A tuple containing two elements:
                - The extracted text from the element, converted to markdown format.
                - The raw HTML content of the element.
    """
    try:
        html = await page.query_selector(selector)
        html = await html.inner_html()
        return extract_to_markdown(html), html
    except Exception as e:
        logger.exception(
            f"Exception occured while getting text from {selector}, ", exc_info=e
        )
        raise e


async def get_query_text(page: Page) -> tuple[str, str]:
    """
    Extracts the query text from the input field on the specified page object.

    Args:
        page (Page): The page object for interacting with the browser.

    Returns:
        tuple[str, str]: A tuple containing the extracted query text and its raw HTML representation.
    """
    query_text, query_html = await get_text_by_selector(page, QUERY_SELECTOR)
    return query_text, query_html


async def get_content_text(page: Page) -> tuple[str, str]:
    """
    Extracts the response content text from the specified page object.

    Args:
        page (Page): The page object for interacting with the browser.

    Returns:
        tuple[str, str]: A tuple containing the extracted response content text and its raw HTML representation.
    """
    content_text, content_html = await get_text_by_selector(page, RESPONSE_SELECTOR)
    return content_text, content_html


async def check_element_and_get_text(
    page: Page,
    selector: str,
    timeout: int = 1000 * 120,
    hidden=False,
    max_retries=100,
    retry_delay=1,
) -> tuple[bool, str]:
    """
    Checks if an element with the specified selector is present on the page and retrieves its text content.

    Args:
        page (Page): The page object for interacting with the browser.
        selector (str): The CSS selector for the element to check.
        timeout (int, optional): The timeout value in milliseconds. Defaults to 120000 (2 minutes)
        hidden (bool, optional): Whether to check for hidden elements. Defaults to False.
        max_retries (int, optional): The maximum number of times to retry checking the element. Defaults to 100.
        retry_delay (int, optional): The delay in seconds between retries. Defaults to 1.

    Returns:
        tuple: A tuple containing a boolean indicating if the element was found and its inner text.
               If the element is not found after the maximum number of retries, returns (False, None).
    """
    retries = 0
    while retries < max_retries:
        try:
            element = await page.wait_for_selector(
                selector, state="attached", timeout=retry_delay * 1000
            )
            if element:
                text = await element.inner_text()
                return True, text
        except Exception as e:
            pass
        retries += 1
        logger.debug("Element not found, Waiting 1 second...")
        await page.wait_for_timeout(1000 * retry_delay)
    return False, None


async def get_hidden_text(page, selector):
    # Evaluate JavaScript in the context of the page to extract the text content of the hidden element
    await page.hover(selector)
    return await page.evaluate(f'document.querySelector("{selector}").innerText')


async def select_model(page: Page, model_choice: str = "llama3-8b") -> Page:
    """
    Selects the specified model from the dropdown menu on the given page.

    Args:
        page (Page): The page object for interacting with the browser.
        model_choice (str, optional): The model to select. Defaults to "llama3-8b".

    Returns:
        None
    """
    try:
        a = await page.wait_for_selector(MODEL_DROPDOWN_SELECTOR)
        a.click()
        downpress = list(modelindex.keys()).index(model_choice)

        for _ in range(downpress):
            await page.keyboard.press("ArrowDown")
        await page.keyboard.press("Enter")
        logger.critical(f"Model set to : {model_choice}")
    except Exception as e:
        logger.exception("Exception occured while selecting model:: ", exc_info=e)
        pass

    return page


async def save_cookie(cookie: Any, file_path: str) -> None:
    """
    Saves the provided cookie data to a JSON file.

    Args:
        cookie (Any): The cookie data to save.
        file_path (str): The path to the file where the cookie data will be saved.

    Returns:
        None
    """
    file_path = Path(file_path)
    with open(file_path, "w") as f:
        json.dump(cookie, f)


def get_cookie(file_path: str) -> Any:
    """
    Loads the cookie data from a JSON file.

    Args:
        file_path (str): The path to the file containing the cookie data.

    Returns:
        Any: The loaded cookie data.
    """
    file_path = Path(file_path)
    with open(file_path, "r") as f:
        return json.load(f)


def file_exists(file_path: str) -> bool:
    """
    Checks if a file exists at the specified path.

    Args:
        file_path (str): The path to the file to check.

    Returns:
        bool: True if the file exists, False otherwise.
    """
    return Path(file_path).is_file()


async def clear_chat(page: Page) -> None:
    """
    Clears the chat history on the specified page.

    Args:
        page (Page): The page object for interacting with the browser.

    Returns:
        None
    """
    try:
        await page.locator(CLEAR_CHAT_BUTTON).click()
        await page.wait_for_timeout(1000 * 3)
        logger.debug("Chat cleared!!!")
    except Exception as e:
        logger.exception("Exception occured while clearing chat, ", exc_info=e)
        pass


async def set_system_prompt(page: Page, system_prompt: str) -> None:
    """
    Sets the system prompt on the specified page.

    Args:
        page (Page): The page object for interacting with the browser.
        system_prompt (str): The system prompt text to set.

    Returns:
        None
    """
    try:
        await page.locator(SYSTEM_PROMPT_BUTTON).click()
        await page.locator(SYSTEM_PROMPT_TEXTAREA).fill(system_prompt)
        await page.wait_for_timeout(1000 * 3)
        logger.debug("Set system prompt!!!")
    except Exception as e:
        logger.exception("Exception occured while setting system prompt, ", exc_info=e)
        pass


async def get_model_name(page: Page):
    """
    returns name of the model being used
    """
    new_model = None
    try:
        model_name_value, _ = await get_text_by_selector(page, MODEL_NAME_SELECTOR)

        if model_name_value in list(modelindex.values()):
            model_name = [k for k, v in modelindex.items() if v == model_name_value][0]
            return model_name
        else:
            logger.critical(f"New Model found: {model_name_value} ")
            new_model = model_name_value
            return None

    except Exception as e:
        logger.exception(
            f"Failed to get model name, New model found {new_model}", exc_info=e
        )
        pass


def show_progress_bar(iterable, desc: str) -> Any:
    """
    Displays a progress bar for the specified iterable.

    Args:
        iterable (iterable): The iterable to display the progress bar for.
        desc (str): A description for the progress bar.

    Returns:
        Any: The wrapped iterable with progress bar.
    """
    return tqdm(iterable, desc=desc)
