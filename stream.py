import asyncio

import aiohttp
from aiohttp import web


async def stream_text(request):
    response = web.StreamResponse()
    response.content_type = 'text/plain'
    await response.prepare(request)

    text_chunks = [
        "Hello, this is the first chunk of text.\n",
        "Here is the second chunk of text.\n",
        "And this is the third and final chunk of text.\n"
    ]

    for chunk in text_chunks:
        await response.write(chunk.encode('utf-8'))
        await asyncio.sleep(1)  # Simulate delay between chunks

    await response.write_eof()
    return response

app = web.Application()
app.router.add_get('/stream', stream_text)

if __name__ == '__main__':
    web.run_app(app, port=8888)
