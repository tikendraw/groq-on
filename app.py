import groq
from groq import Groq

client = Groq(
    
)

try:
    client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {
                "role": "user",
                "content": "Explain the importance of low latency LLMs",
            },
        ],
        model="llama3-8b-8192",
    )
except groq.APIConnectionError as e:
    print("The server could not be reached")
    print(e.__cause__)  # an underlying Exception, likely raised within httpx.
except groq.RateLimitError as e:
    print("A 429 status code was received; we should back off a bit.")
except groq.APIStatusError as e:
    print("Another non-200-range status code was received")
    print(e.status_code)
    print(e.response)