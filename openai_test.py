from openai import OpenAI

# client = OpenAI()

# audio_file = open("speech.mp3", "rb")
# transcript = client.audio.transcriptions.create(
#   model="whisper-1",
#   file=audio_file
# )


client = OpenAI(
    base_url='http://localhost/8888',
    api_key='not_required'
)

completion = client.chat.completions.create(
  model="gpt-4o",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ]
)
print('hehe')
print(completion.choices[0].message)
