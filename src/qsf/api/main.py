import logging

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from qsf.agents.news_comparison import run_news_comparison
from qsf.agents.workflow import pipeline

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(name)s: %(message)s",
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    ticker: str


class NewsComparisonRequest(BaseModel):
    tickers: list[str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
def analyze(request: AnalyzeRequest):
    state = pipeline.invoke({"ticker": request.ticker.upper().strip()})
    if state.get("error"):
        raise HTTPException(status_code=404, detail=state["error"])
    return state["result"]


@app.post("/diagnostics/news-comparison")
def news_comparison(request: NewsComparisonRequest):
    tickers = [t.upper().strip() for t in request.tickers if t.strip()]
    if not tickers:
        raise HTTPException(status_code=422, detail="At least one ticker is required")
    return run_news_comparison(tickers)
