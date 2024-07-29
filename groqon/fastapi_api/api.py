from fastapi import FastAPI

from .v1 import chat
from .v1.experimental import transcription

app = FastAPI()

app.include_router(chat.chat_router)
# app.include_router(transcription.audio_router) # transcription


@app.get("/")
async def root():
    return {"message": "Hello Bigger Applications!"}
