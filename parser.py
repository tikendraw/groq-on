from unstructured.documents.html import HTMLDocument


def check_tag(element, tag:str)-> bool:
    if element.tag==tag:
        return True
    return False

def check_ancestor(element, ancestor:str, n=-1) ->bool:
    if len(element.ancestortags)==0:
        return False
    if element.ancestortags[n]==ancestor:
        return True
    return False

def tag_replacement(element, at_start:str=None, at_end:str=None) -> str:
    if at_start:
        return at_start+element.text
    
    if at_end:
        return element.text+at_end

    if at_start and at_end:
        return at_start+element.text+at_end
    
def is_list_tag(element)->bool:
    if check_tag(element, tag='li') and not check_ancestor(element, ancestor='ul'):
        return True
    return False

def is_code_block(element)->bool:
    if check_ancestor(element, ancestor='code', n=-1):
        return True
    return False

def is_p_tag(element)->bool:
    if check_tag(element, tag='p'):
        return True
    return False


def get_text_from_html(html_code:str=None, html_file:str=None)->dict:
    if html_code:
        hh1 = HTMLDocument.from_string(html_code)
    
    if html_file:
        hh1 = HTMLDocument.from_file(html_file)

    output_dict = {
        'query':None,
        'content':None,
        'tokens/s':None
    }

    content = []
    code_block = []
    
    for num,element in enumerate(hh1.pages[0].elements):
        if num==0 and element.tag=='div':
            output_dict['tokens/s'] = element.text
        elif num==1 and element.tag=='p':
            output_dict['tokens/s'] = element.text

        if num==2 and element.tag=='div':
            output_dict['query'] = element.text
        
        if num>2:
            # print(element.tag, ':', element.text, ':', element.ancestortags if len(element.ancestortags)>0 else None)
            # print(
            #     "is code block: ", is_code_block(element)
            # )
            if is_code_block(element):
                print('code hai')
                code_block.append(element.text)
                print(code_block)
            
            elif is_list_tag(element) and not is_code_block(element):
                print('list had code nahi')
                content.append(tag_replacement(element, at_start='* ', at_end=' \n'))

            else:
                if code_block:
                    code_join = '\n```\n'+'\n'.join(code_block)+'\n```\n'
                    code_block = []
                    content.append(code_join)
                  
                content.append(element.text)

    output_dict['content'] = r'\n'.join(content)

    return output_dict

out = get_text_from_html(html_file='/home/t/got_json.html')
print(out)