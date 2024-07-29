import json
from typing import AsyncGenerator

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from ...async_api.schema import GROQ_APIRequest

chat_router = APIRouter(
    prefix="/openai/v1/chat",
    tags=["chat"],
)


@chat_router.get("/")
async def testt():
    return {"message": "Hello, World!"}


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, "__iter__"):
            return list(obj)
        return json.JSONEncoder.default(self, obj)


@chat_router.post("/completions")
async def completions(request: GROQ_APIRequest):
    print(request.model_dump_json())

    # Convert the request to a dictionary and then to JSON using the custom encoder
    request_dict = request.model_dump(exclude_none=True)
    request_json = json.dumps(request_dict, cls=CustomJSONEncoder)

    # Forward the request to the actual server
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8888",
            content=request_json,
            headers={"Content-Type": "application/json"},
        )

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)

        print(" response in chat fapp router: ", response.content)
        # Check if the response is streamed
        if request.stream:
            print("respnes in fapp ++", response.content)
            return StreamingResponse(
                stream_response(response), media_type="text/event-stream"
            )
        else:
            # For non-streamed responses, return the JSON directly
            return JSONResponse(content=response.json())


async def stream_response(response: httpx.Response) -> AsyncGenerator[str, None]:
    buffer = ""
    async for chunk in response.aiter_bytes():
        buffer += chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk
        while "\n" in buffer:
            message, buffer = buffer.split("\n", 1)
            if message.startswith("data: "):
                if message.strip() == "data: [DONE]":
                    # If this is the [DONE] message, yield it with the previous chunk
                    if buffer.strip():
                        yield f"{buffer.strip()}\ndata: [DONE]\n\n"
                        buffer = ""
                    else:
                        yield f"{message}\n\n"
                else:
                    yield f"{message}\n\n"

    # If there's any remaining data in the buffer, yield it
    if buffer.strip():
        yield f"data: {buffer.strip()}\n\n"
