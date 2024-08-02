# GROQ without API: GroqOn

This project uses Playwright to access [GROQ](https://www.groq.com) using Python, without requiring an API key.

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [CLI Usage](#cli-usage)
    * [Starting the Server](#starting-the-server)
    * [Configuring GroqOn](#configuring-groqon)
    * [Querying the Server](#querying-the-server)
    * [Stopping the Server](#stopping-the-server)
4. [Python API Usage](#python-api-usage)
    * [Using Groq's Official Python API](#using-groqs-official-python-api)
    * [Using the GroqonClient Class](#using-the-groqonclient-class)
    * [Using requests Library for API Calls](#using-requests-library-for-api-calls)
5. [Key Features and Benefits](#key-features-and-benefits)
6. [Contribution](#contribution)

## What's New
* Support for new LLAMA3 and LLAMA3.1 models 
* Groq's Python API support (just add the base URL)

## Introduction
GroqOn is a powerful package that provides a convenient interface to interact with large language models (LLMs) hosted on **Groq.com for FREE**, with **no API key required**. It offers both a command-line interface (CLI) and a Python API for seamless integration into various workflows.

**How it Works**
* It emulates a user using Playwright to query the selected model and outputs the response as a JSON object.
* You need to log in for the first time with a Google account.
* It uses cookies to maintain login. (For the first time, you have to log in within 100 seconds (the browser will be open for 100 sec). Your cookies will be saved and reused for subsequent queries).
* This code doesn't share your cookies or any kind of data with anyone. Your data is saved locally in the `/home/username/.groqon` folder.

**Note**
Groq has set request limits per minute for each model. Once you hit the limit, you will get an error response. The limit is approximately 30 requests/minute (not confirmed), so either use different models (5 models * 30 req = 150 req) or make fewer queries.

## Installation

To install the GroqOn package, use pip:

```bash
pip install groqon
```

Make sure you have Playwright installed. If not, run this command:
```bash
playwright install firefox 
```

## CLI Usage
The GroqOn CLI provides several commands to interact with the GroqOn server.

### Starting the Server
To start the GroqOn server:
```bash
groqon serve [OPTIONS] -w 4
```
Options:
* `--cookie_file`, `-cf`: Path to the cookie file (default: ~/.groqon/groq_cookie.json)
* `--models`, `-m`: Comma-separated list of models to use
* `--headless`, `-hl`: Run in headless mode (default: True)
* `--n_workers`, `-w`: Number of worker windows (default: 2) (the more, the better, but it will consume more RAM)
* `--reset_login`, `-rl`: Reset login information (default: False)
* `--verbose`, `-v`: Enable verbose output (default: False)

### Configuring GroqOn
To configure GroqOn settings (optional, as it has default settings):
```bash
groqon config [OPTIONS]
```
Options:
* `--cookie_file`, `-cf`: Set the cookie file path
* `--models`, `-m`: Set default models
* `--headless`, `-hl`: Set headless mode
* `--n_workers`, `-w`: Set number of workers
* `--reset_login`, `-rl`: Set reset login option
* `--verbose`, `-v`: Set verbose output
* `--print_output`, `-p`: Set print output option

### Querying the Server
To send a query to the server:
```bash
groqon query 'Who is Megan Fox?' [OPTIONS] 
```
Options:
* `--save_dir`, `-o`: Directory to save the generated response
* `--models`, `-m`: Comma-separated model names to use (e.g., -m llama3-8b,gemma2,mixtral,gemma,llama3-70b)
* `--system_prompt`, `-sp`: System prompt for the query
* `--print_output`, `-p`: Print output to console (default: True)
* `--temperature`, `-t`: Temperature for text generation
* `--max_tokens`, `-mt`: Maximum number of tokens to generate
* `--stream`, `-s`: Enable streaming mode (default: True)
* `--top_p`, `-tp`: Top-p sampling parameter (not recommended)
* `--stop_server`, `-ss`: Stop the server after the query (default: False)

### Stopping the Server
To stop the GroqOn server:
```bash
groqon stop
```

## Python API Usage

### Using Groq's Official Python API
Make sure the GroqOn server is running or use the `groqon serve` command to start it.

1. Asynchronous Usage
```python
import asyncio
from groq import AsyncGroq

client = AsyncGroq(
    base_url="http://localhost:8888",  # Get port from web server
    api_key="not required",
)

async def query(prompt) -> None:
    chat_completion = await client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama3-8b-8192",
    )
    print("+" * 33)
    print("PROMPT: ", prompt)
    print("RESPONSE: ")
    print(chat_completion.choices[0].message.content)
    print("+" * 33)

async def main() -> None:
    prompts = [
        "What is the meaning of life?",
        "What is the meaning of death?",
        "What is the meaning of knowledge?",
    ]

    await asyncio.gather(
        query(prompts[0]),
        query(prompts[1]),
        query(prompts[2]),
    )

asyncio.run(main())
```

2. Synchronous Usage
```python
from groq import Groq

client = Groq(
    api_key='not required',
    base_url='http://localhost:8888',  # Only base URL of the server is required
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Explain the importance of fast language models",
        }
    ],
    model="llama3-8b-8192",
)

print(chat_completion.choices[0].message.content)
```

### Using the GroqonClient Class
To use the GroqOn package in your Python code:

Make sure the GroqOn server is running or use the `groqon serve` command to start it.

```python
from groqon import GroqonClient, GroqonClientConfig

# Define the configuration
config = GroqonClientConfig(
    headless=False,
    models=['llama3-8b'],  # Can be llama3-70b, gemma-7b, gemma2, mixtral
    system_prompt="You are a helpful assistant.",
    print_output=True,
    temperature=0.1,
    max_tokens=2048,
    stream=False
)

# Create a GroqonClient instance
client = GroqonClient(config=config)

# Asynchronous usage
async def single_query():
    response = await client.multi_query_async("What is the capital of France?")
    print(response)

async def multiple_queries():
    queries = [
        "What is the capital of France?",
        "Who wrote 'To Kill a Mockingbird'?",
        "What is the largest planet in our solar system?"
    ]
    responses = await client.multi_query_async(queries)
    for response in responses:
        print(response)

# Run the async functions
import asyncio
asyncio.run(single_query())
asyncio.run(multiple_queries())

# Synchronous usage
client.multi_query([
    'How old is the sun?',
    'Who is Lionel Messi?'
])
```

### Using requests Library for API Calls
You can also make API calls to the GroqOn server using the requests library:

```python
import requests

url = "http://localhost:8888/openai/v1/chat/completions"
data = {
    "model": "gemma-7b-it",
    "messages": [
        {"role": "system", "content": "Please try to provide useful, helpful, and actionable answers."},
        {"role": "user", "content": "What are the potential implications of AI on the job market?"}
    ],
    "temperature": 0.1,
    "max_tokens": 2048,
    "top_p": 1,
    "stream": True
}
headers = {"Content-Type": "application/json"}

response = requests.post(url, json=data, headers=headers)
print(response.json())
```

## Key Features and Benefits

* **Fast Inference**: GroqOn leverages the power of Groq's API to provide rapid inference for large language models, significantly reducing the time required for generating responses.
* **Multiple Model Support**: The package supports various LLMs, allowing users to choose the most suitable model for their specific tasks.
* **Asynchronous Processing**: With its asynchronous design, GroqOn can handle multiple queries simultaneously, improving overall throughput and efficiency.
* **Flexible Configuration**: Users can easily customize settings such as temperature, max tokens, and system prompts to fine-tune the model's behavior.
* **CLI and Python API**: GroqOn offers both a command-line interface for quick interactions and a Python API for seamless integration into existing codebases.
* **Error Handling**: Robust error handling ensures that issues are caught and reported, improving the reliability of the system.
* **Save and Print Options**: Users can save generated responses to files and control whether outputs are printed to the console.
* **HTTP Interface**: The server supports HTTP requests, making it easy to integrate with web applications and other services.
* **Worker Management**: The server uses multiple workers to handle requests efficiently, improving concurrency and responsiveness.

By leveraging these features, GroqOn provides a powerful and flexible solution for interacting with large language models, making it an excellent choice for developers and researchers working on natural language processing tasks.

## Contribution
Feel free to contribute your ideas and features to improve GroqOn.

## Support the Project
If you find GroqOn helpful, consider supporting the project:

[!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://buymeacoffee.com/tikendraw)

