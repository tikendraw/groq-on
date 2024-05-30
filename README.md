# GROQ without API : GroqOn

This projects uses playwright to Access [GROQ](https://www.groq.com) using python

## Inspiration(Story)
On a long summer evening of april 2024 I was sitting in front of cranky old fan working on my [News-app](https://www.github.com/tikendraw/news-app). I needed a llm api to summarize the news articles. As a poor man I do not have money to buy llm subscription, I chose to use [Gemini](https://gemini.google.com) (limit 1 req/sec), it took avg 8-9 sec per request to process, felt dissapointed.

At this point I have heard about Groq and its new LPU hardware that outputs insanely fast. 
There wasn't any free api for poor people like me. So I decided to emulate the user using playwright and can query and get the llm response. Finally as the thousands of seconds passed I could manage to make it work. Now you can use groq.com as llm too. I did it for poor people. 

## Working

* It emulates user using playwright and queries the selected model and ouputs the response as a json object.
* It uses cookie to login.(For the first time you have to login in under 120 seconds(browser will be open for 2 minutes), your cookies will be saved and will be used again next time you query)
* This code doesn't share your cookie or any kind of data with anyone.
* One way query.

## Installation

```
pip install groqon
```
make sure you have playwright installed, if not do this
```
# install firefox
playwright install firefox
```

## Usage
### CLI
```
groqon "how old is sun" "how to fuck around and find out"\
    --model llama3-70b\
    --output_dir ./output/\
```
### Code
```
from groqon.async_api import agroq

# pass single query
agroq('how old is earth', model='llama3-70b')

# pass list of query
agroq(["Is aunt may peter parker's actual mother?", "kya gangadhar hi shaktimaan hai?"], model='llama3-70b')

# pass other parameters
agroq(
    'Why am I awake at 2.30 AM?',
    model_list='llama3-70b', 
    cookie_file="./groq_cookie.json", 
    headless=False,
    save_dir='./newresponses/',
    system_prompt="you are jarvis/vision assistant from Ironman and marvel movie, and assistant of me, call me sir",
    print_output=True
    )
```
## help

```
agroq --help
usage: agroq [-h] [--model_list MODEL_LIST [MODEL_LIST ...]] [--cookie_file COOKIE_FILE] [--system_prompt SYSTEM_PROMPT] [--headless HEADLESS] [--output_dir OUTPUT_DIR] [--temperature TEMPERATURE]
             [--max_tokens MAX_TOKENS] [--n_workers N_WORKERS]
             queries [queries ...]

positional arguments:
  queries               one or more quoted string like 'what is the purpose of life?' 'why everyone hates meg griffin?'

options:
  -h, --help            show this help message and exit
  --model_list MODEL_LIST [MODEL_LIST ...]
                        Available models are gemma-7b-it llama3-70b-8192 llama3-8b-8192 mixtral-8x7b-32768, e.g enter as --model_list 'llama3-8b' 'gemma-7b'
  --cookie_file COOKIE_FILE
                        looks in current directory by default for groq_cookie.json. If not found, You will have to login when the browswer opens under 120 seconds. It's one time thing
  --system_prompt SYSTEM_PROMPT
                        System prompt to be given to the llm model. Its like 'you are samuel l jackson as my assistant'. Default is None.
  --headless HEADLESS   set true to not see the browser
  --output_dir OUTPUT_DIR
                        Path to save the output file. Defaults to current working directory.
  --temperature TEMPERATURE
                        Temperature value
  --max_tokens MAX_TOKENS
                        Maximum number of tokens (upper limit)
  --n_workers N_WORKERS
                        Number of browers instances to work simultaneously. Keep between 1-8 (eats up ram)


```

## TODO (Need Help)

* [ ] Set System Prompt
* [ ] Keep updated
* [ ] Add something
* [ ] Use Better parser
* [ ] Use color for better Visual / Rich text formatting
* [ ] Add logger
* [ ] Multiround chat

## Contribution

Feel free to add features and keep it maintained and do pull requests.

## Buy me a Coffee/Chai
[!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://buymeacoffee.com/tikendraw)

