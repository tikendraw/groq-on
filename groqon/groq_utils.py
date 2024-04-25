import json
import re
from html import unescape
from pathlib import Path

from playwright.sync_api import Page

from .groq_config import modelindex

CLEAR_CHAT_BUTTON = "body > main > div > div.flex.flex-col-reverse.md\:flex-col.md\:relative.w-full.max-w-\[900px\].bg-background.z-10.gap-2.md\:gap-6 > div > button.inline-flex.items-center.justify-center.whitespace-nowrap.rounded-md.text-sm.font-medium.ring-offset-background.transition-colors.focus-visible\:outline-none.disabled\:pointer-events-none.disabled\:opacity-50.underline-offset-4.h-10.p-0.text-muted-foreground.hover\:text-primaryaccent.hover\:no-underline"
DROPDOWN_BUTTON = "body > header > div.flex-1.flex.items-center.justify-center.md\:col-span-4.md\:col-start-2 > div > button.flex.h-10.w-full.items-center.justify-between.border.border-input.bg-background.py-2.text-sm.ring-offset-background.placeholder\:text-muted-foreground.focus\:outline-none.focus\:ring-ring.focus\:ring-offset-2.disabled\:cursor-not-allowed.disabled\:opacity-50.\[\&\>span\]\:line-clamp-1.ml-auto.max-w-\[160px\].sm\:max-w-\[190px\].rounded-none.border-none.overflow-hidden.px-2.sm\:px-3.focus\:ring-0"
QUERY_SELECTOR = "div.break-words"
RESPONSE_SELECTOR = "div.text-base"

def select_model(page:Page, model_choice):
    # dropdown_button
    page.locator(DROPDOWN_BUTTON).click()

    downpress = modelindex.index(model_choice)
    
    for i in range(downpress):
        page.keyboard.press("ArrowDown")
    page.keyboard.press("Enter")

    print(f'Model set to : {model_choice}')

def check_element_and_get_text(page:Page, selector, max_retries=100, retry_delay=1):
    retries = 0
    while retries < max_retries:
        element = page.locator(selector)
        if element:
            return True, element.inner_text()
        else:
            retries += 1
            print('Waiting 1 second...')
            page.wait_for_timeout(1000*retry_delay)
    return False, None

def clear_chat(page:Page):
    page.locator(CLEAR_CHAT_BUTTON).click()
    print('Chat cleared')

def get_text_by_selector(page:Page, selector):
    html = page.query_selector(selector).inner_html()
    return extract_to_markdown(html), html

# Get the text from the desired elements
def get_query(page:Page):
    return get_text_by_selector(page,QUERY_SELECTOR)

def get_content(page:Page):
    return get_text_by_selector(page,RESPONSE_SELECTOR)
    
    
#load the cookie
def get_cookie(file_name):
    """gets the cookie"""
    with open(file_name) as f:
        return json.load(f)
    return None

def file_exists(file):
  """Checks if a file exists using pathlib."""
  path = Path(file)
  return path.exists()

#save the cookie
def save_cookie(cookie,filename="cookies.json"):
    """saves the cookie"""
    with open(filename, 'w') as output:
        json.dump(cookie, output)
        

def extract_to_markdown(html_source):
    output = []
    in_table = False
    code_block = []
    for line in html_source.split('\n'):
        line = line.strip()
        if line.startswith('<pre>'):
            output.append('```\n')
            code_block = []
        elif line.startswith('</pre>'):
            code_block.append(line.replace('</pre>', ''))
            code = unescape('\n'.join(code_block))
            output.append(code + '\n')
            output.append('```\n\n')
            code_block = []
        elif line.startswith('<table'):
            table_lines = [line]
            in_table = True
        elif in_table and line.startswith('</table>'):
            table_lines.append(line)
            table = '\n'.join(table_lines)
            output.append(table + '\n\n')
            in_table = False
        elif in_table:
            table_lines.append(line)
        else:
            text = re.sub(r'<[^>]+>', '', line)
            if text:
                output.append(text + '\n')
    return ''.join(output)


def get_all_text(soup_obj):
    text = []
    for child in soup_obj.descendants:
        if child.string:
            text.append(child.string.strip())
        elif child.name in ['li', 'p', 'th', 'td', 'code']:
            text.append(child.get_text().strip())
    return text
