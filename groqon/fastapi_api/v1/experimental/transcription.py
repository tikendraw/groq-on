import json
import os
from typing import AsyncGenerator

import aiofiles
import httpx
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from ...async_api.schema import GROQ_APIRequest

audio_router = APIRouter(
    prefix="/openai/v1/audio/transcriptions",
    tags=["transcription", "whisper", "audio"],
)


@audio_router.get("/test")
async def testt():
    return {"message": "transcription says - Hello, World!"}


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, "__iter__"):
            return list(obj)
        return json.JSONEncoder.default(self, obj)


@audio_router.post("/")
async def transcript(request: Request, file: UploadFile = File(...)):
    # Read the audio file

    # save that audio as audio 001
    upload_directory = "uploads"
    os.makedirs(upload_directory, exist_ok=True)

    # Define the file path where the audio file will be saved
    file_path = os.path.join(upload_directory, file.filename)

    # Save the audio file to the server
    try:
        async with aiofiles.open(file_path, "wb") as audio_file:
            content = await file.read()
            await audio_file.write(content)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to save the file: {str(e)}"
        )

    # Optionally, you can also access other parts of the request
    body = await request.form()  # Access the request form data
    headers = request.headers  # Access the request headers
    query_params = request.query_params  # Access the query parameters

    print("Request Form Data:", body)
    print("Request Headers:", headers)
    print("Query Parameters:", query_params)

    # Return a response
    return {
        "message": "Audio file received and saved",
        "filename": file.filename,
        "content_type": file.content_type,
        "file_path": file_path,
    }
