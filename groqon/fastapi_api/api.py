# # from fastapi import FastAPI

# # from .v1 import chat
# # from .v1.experimental import transcription

# # app = FastAPI()

# # app.include_router(chat.chat_router)
# # # app.include_router(transcription.audio_router) # transcription


# # @app.get("/")
# # async def root():
# #     return {"message": "Hello Bigger Applications!"}

# from aiohttp import web

# from .v1.chat import routes

# app = web.Application()
# app.add_routes(routes)
# # web.run_app(app)

import asyncio

from aiohttp import web

from ..async_api.agroq_server import AgroqServer
from ..async_api.schema import AgroqServerConfig


class AgroqWebApp:
    def __init__(self):
        self.config = AgroqServerConfig(headless=False, n_workers=7)
        self.agroq = AgroqServer(config=self.config)
        self.app = web.Application()
        self.app.router.add_post('/openai/v1/chat/completions', self.handle_chat_completions)

    async def start(self):
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, 'localhost', 8080)
        await site.start()
        print("Web server started on http://localhost:8080")
        
        # Start the AgroqServer
        await self.agroq.astart()

    async def handle_chat_completions(self, request):
        try:
            data = await request.json()
            stream = data.get('stream', False)
            print('request server side: ', data)
            output = await self.agroq.process_request(data)
            
            print('got response: ', output)
            if not stream:
                return web.json_response(output, status=200)
            else:
                response = web.StreamResponse(
                    status=200,
                    reason='OK',
                    headers={'Content-Type': 'text/event-stream'}
                )
                await response.prepare(request)
                
                # output contains data: [DONE] but here we are concatinating
                # data: [DONE] to second last message as \n\ndata: [DONE]\n\n
                chunks, _end = output[:-1], output[-1]
                for i, chunk in enumerate(chunks):
                    if i == len(chunks) - 1:  
                        combined_chunk = f"{chunk.strip()}\n\n{_end}\n\n"
                        await response.write(f"{combined_chunk}".encode('utf-8'))

                    elif i < len(chunks) - 2:  # All chunks except the last two
                        await response.write(f"{chunk}\n\n".encode('utf-8'))
                    
                    await response.drain()

                return response
            
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)