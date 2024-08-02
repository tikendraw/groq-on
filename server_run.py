import asyncio

from groqon.server.groqon_server import GroqonWebApp

from .groqon.async_api.schema import GroqonConfig

if __name__ == "__main__":
    config = GroqonConfig(n_workers=2, headless=True)
    web_app = GroqonWebApp(config=config)
    asyncio.run(web_app.start())
