import logging

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from qsf.agents.news_comparison import run_news_comparison, run_news_comparison_stream
from qsf.agents.workflow import pipeline
from qsf.forecasting.pipeline import ForecastingPipeline

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


class ForecastRequest(BaseModel):
    ticker: str
    period: str = "2y"
    include_sentiment: bool = False


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


@app.post("/forecast")
def forecast(request: ForecastRequest):
    """Run the forecasting pipeline with optional sentiment integration.

    When include_sentiment is True, first runs the sentiment pipeline to get
    daily sentiment scores, then feeds them as features into the forecasting
    models alongside technical and macro indicators.
    """
    ticker = request.ticker.upper().strip()

    sentiment_daily = None
    if request.include_sentiment:
        state = pipeline.invoke({"ticker": ticker})
        if not state.get("error") and state.get("result"):
            import pandas as pd
            daily_data = state["result"].get("daily_data", [])
            if daily_data:
                sdf = pd.DataFrame(daily_data)
                sdf["date"] = pd.to_datetime(sdf["date"])
                sdf = sdf.set_index("date")
                sentiment_daily = sdf[["news_sentiment", "social_sentiment"]]

    fp = ForecastingPipeline(ticker=ticker, period=request.period)
    result = fp.run(sentiment_daily=sentiment_daily)

    if result.get("error"):
        raise HTTPException(status_code=404, detail=result["error"])

    return result


@app.post("/diagnostics/news-comparison")
def news_comparison(request: NewsComparisonRequest):
    tickers = [t.upper().strip() for t in request.tickers if t.strip()]
    if not tickers:
        raise HTTPException(status_code=422, detail="At least one ticker is required")
    return run_news_comparison(tickers)


@app.get("/diagnostics/news-comparison/stream")
def news_comparison_stream(tickers: list[str] = Query(...)):
    clean = [t.upper().strip() for t in tickers if t.strip()]
    if not clean:
        raise HTTPException(status_code=422, detail="At least one ticker is required")
    return StreamingResponse(
        run_news_comparison_stream(clean),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
