from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from qsf.agents.workflow import pipeline

load_dotenv()

app = FastAPI()


class AnalyzeRequest(BaseModel):
    ticker: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
def analyze(request: AnalyzeRequest):
    state = pipeline.invoke({"ticker": request.ticker.upper().strip()})
    if state.get("error"):
        raise HTTPException(status_code=404, detail=state["error"])
    return state["result"]
