# GROQ without API : GroqOn

This projects uses playwright to Access [GROQ](https://www.groq.com) using python

## Table of Contents
1. Introduction
2. Installation
3. CLI Usage
  * Starting the CLI
  * Configuring the CLI
  * Starting the Server
  * Querying the Server
  * Stopping the Server
4. Python API Usage
  * Using the AgroqClient Class
  * Using requests Library
5. Key Features and Benefits
6. Contribution

## Introduction
Groqon is a powerful package that provides a convenient interface to interact with large language models (LLMs) hosted on **Groq.com for FREE**, **No api is required**. It offers both a command-line interface (CLI) and a Python API for seamless integration into various workflows.


**Working**
* It emulates user using playwright and queries the selected model and ouputs the response as a json object.
* You have to login for the first time with google account.
* It uses cookie to login.(For the first time you have to login in under 100 seconds(browser will be open for 100 sec), your cookies will be saved and will be used again next time you query).
* This code doesn't share your cookie or any kind of data with anyone. Your data is saved locally in `/home/username/.groqon` folder. 


**Note**
Groq has set some requests/minute limit per model. Once you hit the limit, you will get error response. Limit is 30req/min (not sure), so either use all different models (5 models * 30 req) =150 req, or just make fewer queries.

## Installation

To install the Groqon package, use pip:

```bash
pip install groqon
```
make sure you have playwright installed, if not do this
```bash
# install firefox
playwright install firefox 
```

## CLI Usage
The Groqon CLI provides several commands to interact with the Agroq server.


  ### Starting the Server
  To start the Agroq server:
  ```bash
  groqon serve [OPTIONS] -w 4
  ```
  Options:

  * `--cookie_file`, `-cf` : Path to the cookie file (default: ~/.groqon/groq_cookie.json)
  * `--models`, `-m` : Comma-separated list of models to use
  * `--headless`, `-hl` : Run in headless mode (default: True)
  * `--n_workers`, `-w` : Number of worker windows (default: 2) (the more the better) (its like opening browser windows, so it will eat ram, do not go crazy with number, 2-8 is a good number)
  * `--reset_login`, `-rl` : Reset login information (default: False)
  * `--verbose`, `-v` : Enable verbose output (default: False)


  ### Querying the Server
  To send a query to the server:
  ```bash
  groqon query 'who is megan fox?' [OPTIONS] 
  ```
  Options:

  * `--save_dir` , `-o` : Directory to save the generated response
  * `--models` , `-m` : Comma-separated(no spaces) model names to use e.g. -m llama3-8b,gemma2,mixtral,gemma,llama3-70b
  * `--system_prompt` , `-sp` : System prompt for the query
  * `--print_output` , `-p` : Print output to console (default: True)
  * `--temperature` , `-t` : Temperature for text generation
  * `--max_tokens` , `-mt` : Maximum number of tokens to generate
  * `--stream` , `-s` : Enable streaming mode (default: True)
  * `--top_p` , `-tp` : Top-p sampling parameter (do not use this)
  * `--stop_server` , `-ss` : Stop the server after the query (default: False)

  ### Configuring the Package
  To configure Groqon settings: (You do not have to do it. It has default settings)
  ```bash
  groqon config [OPTIONS]
  ```
  Options:

  * `--cookie_file`, `-cf` : Set the cookie file path
  * `--models`, `-m` : Set default models
  * `--headless`, `-hl` : Set headless mode
  * `--n_workers`, `-w` : Set number of workers
  * `--reset_login`, `-rl` : Set reset login option
  * `--verbose`, `-v` : Set verbose output
  * `--print_output`, `-p` : Set print output option



  ### Stopping the Server
  To stop the Agroq server:
  ```bash
  groqon stop
  ```


## Python API Usage
  ### Making Requests to the Agroq Server
  To use the Groqon package in your Python code:
  
  Make sure you have start groqon server with `groqon serve` command in the background
  ```python
  from groqon import AgroqClient, AgroqClientConfig

  # Define the configuration
  config = AgroqClientConfig(
      headless=False,
      models=['llama3-8b'], # can be llama3-70b , gemma-7b , gemma2, mixtral
      system_prompt="You are a helpful assistant.",
      print_output=True,
      temperature=0.1,
      max_tokens=2048,
      stream=False
  )

  # Create an AgroqClient instance
  agroq = AgroqClient(config=config)

  # use Asynchronously =====================================

  # Make a single query
  async def single_query():
      response = await agroq.multi_query_async("What is the capital of France?")
      print(response)

  # Make multiple queries
  async def multiple_queries():
      queries = [
          "What is the capital of France?",
          "Who wrote 'To Kill a Mockingbird'?",
          "What is the largest planet in our solar system?"
      ]
      responses = await agroq.multi_query_async(queries)
      for response in responses:
          print(response)

  # Run the async functions
  import asyncio
  asyncio.run(single_query())
  asyncio.run(multiple_queries())


  # use synchronously =====================================
  agroq.multi_query([
    'how old is sun',
    'who is messi?'
  ])

  ```

  ### Using wget or requests for API Calls
  You can also make API calls to the Groqon server using wget or the requests library:
  Using wget:
  ```bash
  wget -q -O - --header="Content-Type: application/json" --post-data '{"model": "gemma-7b-it","messages": [{"role": "system", "content": "Please try to provide useful, helpful and actionable answer"},{"role": "user", "content": "who is megan fox?"}],"temperature": 0.1,"max_tokens": 2048,"top_p": 1,"stream": true}' http://localhost:8888/
  ```
  Using requests:
  ```python
import requests

url = "http://localhost:8888"
data = {
    "model": "gemma-7b-it",
    "messages": [
        {"role": "system", "content": "Please try to provide useful, helpful and actionable answers."},
        {"role": "user", "content": "why donald trump got shot?"}
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

* **Fast Inference**: Groqon leverages the power of Groq's API to provide rapid inference for large language models, significantly reducing the time required for generating responses.
* **Multiple Model** Support: The package supports various LLMs, allowing users to choose the most suitable model for their specific tasks.
* **Asynchronous Processing**: With its asynchronous design, Groqon can handle multiple queries simultaneously, improving overall throughput and efficiency.
* **Flexible Configuration**: Users can easily customize settings such as temperature, max tokens, and system prompts to fine-tune the model's behavior.
* **CLI and Python API** : Groqon offers both a command-line interface for quick interactions and a Python API for seamless integration into existing codebases.
* **Error Handling**: Robust error handling ensures that issues are caught and reported, improving the reliability of the system.
* **Save and  Print Options**: Users can save generated responses to files and control whether outputs are printed to the console.
* **HTTP Interface**: The server supports HTTP requests, making it easy to integrate with web applications and other services.
* **Worker Management**: The server uses multiple workers to handle requests efficiently, improving concurrency and responsiveness.

By leveraging these features, Groqon provides a powerful and flexible solution for interacting with large language models, making it an excellent choice for developers and researchers working on natural language processing tasks.


## Contribution
Feel free to contibute your ideas, features.

## Buy me a Coffee
[!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://buymeacoffee.com/tikendraw)

