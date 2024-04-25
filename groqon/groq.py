import argparse
import json
import os
from contextlib import contextmanager

from playwright.sync_api import Page, sync_playwright

from .groq_config import modelindex
from .groq_utils import (
    check_element_and_get_text,
    clear_chat,
    file_exists,
    get_content,
    get_cookie,
    get_query,
    save_cookie,
    select_model,
)

URL = 'https://groq.com/'
QUERY_INPUT_SELECTOR = "#chat"
QUERY_SUBMIT_SELECTOR = ".self-end"
END_TEXT_SELECTOR = "body > main > div > div.flex.flex-col-reverse.md\:flex-col.md\:relative.w-full.max-w-\[900px\].bg-background.z-10.gap-2.md\:gap-6 > div > a > div > div"

@contextmanager
def groq_context(cookie_file:str='groq_cookie.json', model:str='llama3-70b', headless:bool=False):

    url = URL

    if file_exists(cookie_file):
        cookie = get_cookie(cookie_file)
    else:
        headless=False
        cookie=None
    
    with sync_playwright() as p:
        browser = p.firefox.launch(headless=headless)
        context = browser.new_context()
        page = context.new_page()

        if cookie:
            context.add_cookies(cookie)
            print("Cookie loaded!!!")
        else:
            print("Cookie not loaded!!!", "You have 120 seconds to login to groq.com, Make it quick!!! HEADLESS is set to False")

        page.goto(url, timeout=60_000, wait_until='networkidle')
        
        if not cookie:
            page.wait_for_timeout(1000*120) #120 sec to login
            
        page.screenshot(path="screenshot1.png", full_page=True)

        # Set model
        select_model(page, model_choice=model)

        try:
            yield page
        finally:
            # Save cookie
            save_cookie(context.cookies(), cookie_file)
            browser.close()
            print("Browser closed!!!")

def get_groq_response(page:Page, query:str, save_output:bool=False, save_dir:str=str(os.getcwd())):
    # Add query
    page.locator(QUERY_INPUT_SELECTOR).fill(query)
    # Submit query
    page.locator(QUERY_SUBMIT_SELECTOR).click()

    # Check if generation finished if not wait till end and get tok/s
    is_present, end_text = check_element_and_get_text(page, END_TEXT_SELECTOR)

    if not is_present:
        print("Generation not finished!!!")
        page.screenshot(path="screenshot_generation_not_finished.png", full_page=True)
        return

    # Get query and text generated
    query_text, raw_query_html = get_query(page)
    response_content, raw_response_html = get_content(page)

    # Screenshot the content (for no reason)
    page.screenshot(path="screenshot2.png", full_page=True)
    clear_chat(page)

    output_dict = {
                "query": query_text,
                "response": response_content,
                "query_html":raw_query_html,
                "response_html":raw_response_html,
                "tok/s": end_text
            }
    if save_output:
        with open(os.path.join(save_dir,query_text + '.json'), 'w') as f:
            json.dump(output_dict, f)

    return output_dict
    

def groq(query_list:list[str], cookie_file:str='groq_cookie.json', model:str='llama3-70b', headless:bool=False, save_output:bool=True, save_dir=str(os.getcwd())):
    
    if isinstance(query_list, str):
        query_list = [query_list]
        
    with groq_context(cookie_file=cookie_file, model=model, headless=headless) as page:
        for query in query_list:
            response = get_groq_response(page, query, save_output=save_output, save_dir=save_dir)
            print('Query', ":", response.get('query', None))
            print('Response', ":", response.get('response', None))
            print('tok/s', ":", response.get('tok/s', None))
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('queries', nargs='+', help="one or alot of quoted string like 'what is the purpose of life?' 'why everyone hates meg griffin?'")
    parser.add_argument('--model', default='llama3-70b', help=f"Available models are {" ".join(modelindex)}")
    parser.add_argument('--cookie_file', type=str, default='groq_cookie.json')
    parser.add_argument('--headless', action='store_true', help= "set true to not see the browser")
    parser.add_argument('--save_output', action='store_true', help="set true to save the groq output with its query name.json")
    parser.add_argument('--output_dir', default=str(os.getcwd()), help="Path to save the output file. Defaults to current working directory.")

    args = parser.parse_args()
    
    groq(
        cookie_file=args.cookie_file, 
        model=args.model, 
        headless=args.headless,
        query_list=args.queries, 
        save_output=args.save_output, 
        save_dir=args.output_dir
    )


if __name__ == '__main__':
    main()
