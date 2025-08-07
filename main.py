from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tools_calling import chatbot

class QueryInput(BaseModel):
    question:str
    # stream:bool = False

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.post("/ask")
def handle_question(input: QueryInput):
    try:
        response = chatbot(input.question).replace("*","")
        return JSONResponse(content={"response":response})
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))