import ast
import json
from html import unescape
import re


def parse_script(response: str) -> tuple[str, str]:
    """Split the response into a message and a script.

    Expected use is: run the script if there is one, otherwise print the message.
    """
    # Parse delimiter
    n_delimiters = response.count("```")
    if n_delimiters < 2:
        return response, ""
    segments = response.split("```")
    message = f"{segments[0]}\n{segments[-1]}"
    script = "```".join(segments[1:-1]).strip()  # Leave 'inner' delimiters alone

    # Check for common mistakes
    if script.split("\n")[0].startswith("python"):
        script = "\n".join(script.split("\n")[1:])
    try:  # Make sure it isn't json
        script = json.loads(script)
    except Exception:
        pass
    try:  # Make sure it's valid python
        ast.parse(script)
    except SyntaxError:
        return f"Script contains invalid Python:\n{response}", ""
    return message, script


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
