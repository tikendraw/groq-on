import json
import re
from html import unescape
from pathlib import Path

from playwright.sync_api import Page
from typing import Any
from .groq_config import modelindex

from .element_selectors import(
    CLEAR_CHAT_BUTTON, 
    DROPDOWN_BUTTON, 
    QUERY_SELECTOR, 
    RESPONSE_SELECTOR, 
    PROMPT_SETTER_SELETOR, 
    SYSTEM_PROMPT_TEXTAREA, 
    PROMPT_SAVE_BUTTON_SELECTOR, 
    PROFILE_BUTTON_SELECTOR,
    ADVANCED_SETTING_SELECTOR,
    SYSTEM_PROMPT_BUTTON
)

# setting system prompt is removed
def set_system_prompt(page: Page, system_prompt: str) -> None:
    """
    Sets the system prompt on the page for interaction.

    Args:
        page (Page): The page object for interacting with the browser.
        system_prompt (str): The system prompt text to set.

    Returns:
        None
    """
    try:
        profile = page.locator(PROFILE_BUTTON_SELECTOR)
        profile.click()
        
        page.locator(ADVANCED_SETTING_SELECTOR).click()
        page.locator(SYSTEM_PROMPT_BUTTON).click()
        
        text_area = page.locator(SYSTEM_PROMPT_TEXTAREA)
        text_area.fill(system_prompt)

        page.locator(PROMPT_SAVE_BUTTON_SELECTOR).click()
        print("Set System prompt!")
    except Exception as e:
        print("Exception occured while setting System Prompt:: ", e)


def select_model(page: Page, model_choice) -> None:
    """
    Selects a model from a dropdown button on a page.

    Args:
        page (Page): The page object for interacting with the browser.
        model_choice (str): The name of the model to select.

    Returns:
        None

    Raises:
        Exception: If an error occurs while selecting the model.

    Description:
        This function clicks on the dropdown button on the page, finds the index of the model_choice
        in the modelindex list, and then presses the ArrowDown key that many times. Finally, it presses
        the Enter key to select the model. If an error occurs during the process, it prints an error message.
    """
    try:
        # dropdown_button
        page.locator(DROPDOWN_BUTTON).click()

        downpress = list(modelindex.keys()).index(model_choice)

        for _ in range(downpress):
            page.keyboard.press("ArrowDown")
        page.keyboard.press("Enter")

        print(f"Model set to : {model_choice}")
    except Exception as e:
        print("Exception occured while selecting model:: ", e)


def check_element_and_get_text(
    page: Page, selector, max_retries=100, retry_delay=1
) -> [bool, str | None]:
    """
    Checks if an element with the given selector exists on the page and retrieves its inner text.

    Args:
        page (Page): The page object for interacting with the browser.
        selector (str): The CSS selector of the element to check.
        max_retries (int, optional): The maximum number of times to retry checking the element. Defaults to 100.
        retry_delay (int, optional): The delay in seconds between retries. Defaults to 1.

    Returns:
        tuple: A tuple containing a boolean indicating if the element was found and its inner text.
               If the element is not found after the maximum number of retries, returns (False, None).
    """
    retries = 0
    while retries < max_retries:
        element = page.locator(selector)
        if element:
            return True, element.inner_text()
        retries += 1
        print("Waiting 1 second...")
        page.wait_for_timeout(1000 * retry_delay)
    return False, None


def clear_chat(page: Page) -> None:
    """
    Clears the chat on the given page.

    Args:
        page (Page): The page object representing the chat interface.

    Returns:
        None

    Raises:
        Exception: If an error occurs while clearing the chat.
    """
    try:
        page.locator(CLEAR_CHAT_BUTTON).click()
        print("Chat cleared")
    except Exception as e:
        print("Exception occured while clearing chat:: ", e)


def get_text_by_selector(page: Page, selector) -> str:
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
    html = page.query_selector(selector).inner_html()
    return extract_to_markdown(html), html


# Get the text from the desired elements
def get_query_text(page: Page) -> str:
    """
    Retrieves the text content from the element specified by the given selector on the given page.

    Args:
        page (Page): The page object for interacting with the browser.

    Returns:
        tuple: A tuple containing two elements:
                - The extracted text from the element, converted to markdown format.
                - The raw HTML content of the element.
    """
    return get_text_by_selector(page, QUERY_SELECTOR)


def get_content_text(page: Page) -> str:
    """
    Retrieves the content from the specified page using the given selector.

    Args:
        page (Page): The page object for interacting with the browser.

    Returns:
        tuple: A tuple containing two elements:
                - The extracted content from the element, converted to markdown format.
                - The raw HTML content of the element.
    """
    return get_text_by_selector(page, RESPONSE_SELECTOR)


# load the cookie
def get_cookie(file_name: str) -> dict | None:
    """
    Retrieves the cookie from the specified file.

    Parameters:
        file_name (str): The name of the file containing the cookie.

    Returns:
        dict: The cookie data loaded from the file.

    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    with open(file_name) as f:
        return json.load(f)
    return None


def file_exists(file: str) -> bool:
    """
    Checks if a file exists using pathlib.

    Args:
        file (str): The path of the file to check.

    Returns:
        bool: True if the file exists, False otherwise.
    """
    path = Path(file)
    return path.exists()


# save the cookie
def save_cookie(cookie: Any, filename: str = "cookies.json") -> None:
    """
    saves the cookie
    Parameters:
        cookie: The cookie data to be saved.
        filename (optional): The name of the file to save the cookie to. Defaults to 'cookies.json'.
    Returns:
        None
    """
    with open(filename, "w") as output:
        json.dump(cookie, output)


def extract_to_markdown(html_source: str) -> str:
    """
    Extracts markdown content from the given HTML source.

    Args:
        html_source (str): The HTML source code from which to extract markdown content.

    Returns:
        str: The extracted markdown content from the HTML source.
    """
    output = []
    in_table = False
    code_block = []
    for line in html_source.split("\n"):
        line = line.strip()
        if line.startswith("<pre>"):
            
            output.append("```\n")
            code_block = []
        elif line.startswith("</pre>"):
            code_block.append(line.replace("</pre>", ""))
            code = unescape("\n".join(code_block))
            output.append(code + "\n")
            output.append("```\n\n")
            code_block = []
        elif line.startswith("<table"):
            table_lines = [line]
            in_table = True
        elif in_table and line.startswith("</table>"):
            table_lines.append(line)
            table = "\n".join(table_lines)
            output.append(table + "\n\n")
            in_table = False
        elif in_table:
            table_lines.append(line)
        else:
            text = re.sub(r"<[^>]+>", "", line)
            if text:
                output.append(text + "\n")
    return "".join(output).strip().strip(r"\n")
