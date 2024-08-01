import asyncio

import uvicorn
from aiohttp import web

from groqon.fastapi_api.api import AgroqWebApp

# if __name__ == "__main__":
    # uvicorn.run(app, host="0.0.0.0", port=8000)
    # web.run_app(app, host="0.0.0.0", port=8000)
    
if __name__ == '__main__':
    agroq_web_app = AgroqWebApp()
    asyncio.run(agroq_web_app.start())