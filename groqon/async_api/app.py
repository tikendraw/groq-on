
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .agroqv3 import Agroq
from .schema import AgroqConfig, RequestModel, ResponseModel
from fastapi import FastAPI, BackgroundTasks, HTTPException

app = FastAPI()

config = AgroqConfig(
    query_list=[],
    headless=False,
    # save_dir=None,
    models=['llama3-8b'],
    system_prompt="Please try to provide useful, helpful and actionable answers.",
    print_output=True,
    temperature=0.1,
    max_tokens=2048,
    n_workers=1,
    reset_login=False,
)
agroq = Agroq(config=config)

@app.get('/')
async def home():
    return "Hello World!"

@app.post("/query")
async def query_agroq(query:str):
    try:
        responses = await agroq.multi_query_async(
            query_list=[query], 
            model_list=['llama3-8b']
        )
        return {"responses": [response.model_dump() for response in responses]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
