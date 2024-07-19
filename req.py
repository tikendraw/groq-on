import asyncio
import random

import aiohttp
import requests

prompts = [ 'closest star to earth?',  "what comes after monday?",  "What is the capital of France?", "Who painted the Mona Lisa?", "What is the largest planet in our solar system?", 
           "Who wrote the novel \"To Kill a Mockingbird\"?", "What is the largest mammal on Earth?",  "Who is the lead singer of the band Queen?", "What is the chemical symbol for gold?", 
           "Who is the main character in the novel \"The Catcher in the Rye\"?", "What is the largest living species of lizard?", "Who is the founder of Google?", "What is the smallest country in the world?", 
           "Who is the lead singer of the band The Beatles?", "What is the chemical symbol for oxygen?", "Who wrote the famous novel \"1984\"?", "What is the largest species of shark?",
            "Who is the main character in the novel \"The Great Gatsby\"?", "What is the chemical symbol for carbon?", "Who is the founder of Facebook?", "What is the largest species of bear?", 
            "Who is the main character in the novel \"Pride and Prejudice\"?", "What is the tallest mountain in the world?", "Who discovered penicillin?", "What is the speed of light?",
            "What is the hardest natural substance on Earth?","Who developed the theory of relativity?","What is the smallest planet in our solar system?","What is the longest river in the world?",
            "Who invented the telephone?", "What is the chemical symbol for water?", "Who painted the Sistine Chapel ceiling?", "What is the largest ocean on Earth?", "Who wrote the play \"Romeo and Juliet\"?", 
            "What is the most abundant gas in the Earth's atmosphere?", "Who is the main character in the novel \"Moby-Dick\"?", "What is the largest desert in the world?", "Who is known as the father of modern physics?", 
            "What is the capital of Japan?", "Who wrote the novel \"The Great Gatsby\"?", "What is the largest species of bird?", "What is the capital of Australia?", "Who was the first president of the United States?",
            "What is the freezing point of water?", "Who wrote the novel \"Jane Eyre\"?", "What is the largest organ in the human body?", "Who is the main character in the novel \"To Kill a Mockingbird\"?",
            "What is the capital of Canada?", "a short love story between batman and catwoman", "Who painted the \"Starry Night\"?", "What is the boiling point of water?", "Who discovered the law of gravity?", "What is the capital of Italy?",
            "Who wrote the play \"Hamlet\"?", "What is the largest land animal?", "Who invented the light bulb?", "What is the chemical symbol for sodium?", "Who is the main character in the novel \"1984\"?",
            "What is the largest island in the world?", "Who wrote the novel \"Pride and Prejudice\"?", "What is the capital of Germany?", "What is the largest continent on Earth?", "Who developed the polio vaccine?",
            "What is the chemical symbol for iron?", "Who painted the \"Last Supper\"?", "What is the capital of Spain?", "Who wrote the novel \"Wuthering Heights\"?", "What is the largest fish in the world?",
            "Who is known as the father of modern chemistry?", "What is the capital of Russia?", "Who wrote the novel \"The Hobbit\"?", "What is the chemical symbol for potassium?", "Who painted the \"Birth of Venus\"?"

]



async def req(prompt):
    url = "http://localhost:8888/chat/completions"
    data = {
        "model": "gemma-7b-it",
        "messages": [
            {
                "role": "system",
                "content": "Please try to provide useful, helpful and actionable answers."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.1,
        "max_tokens": 2048,
        "top_p": 1,
        "stream": True
    }

    headers = {"Content-Type": "application/json"}
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=data, headers=headers) as response:
            return await response.json()

k = 10
sel_prompts = random.sample(prompts, k)

async def main():
    tasks = [req(prompt) for prompt in sel_prompts]
    results = await asyncio.gather(*tasks)
    for result in results:
        print(result)

# Run the main function
asyncio.run(main())