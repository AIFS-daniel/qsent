import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # Must run before auth is imported so env vars are populated

import httpx  # noqa: E402
from fastapi import Depends, FastAPI, HTTPException, Query  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse, Response, StreamingResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from qsf.agents.news_comparison import run_news_comparison, run_news_comparison_stream  # noqa: E402
from qsf.agents.workflow import pipeline  # noqa: E402
from qsf.api.auth import get_current_user  # noqa: E402
from qsf.api.auth import router as auth_router  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(name)s: %(message)s",
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)


class AnalyzeRequest(BaseModel):
    ticker: str


class NewsComparisonRequest(BaseModel):
    tickers: list[str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
def analyze(request: AnalyzeRequest, user: dict = Depends(get_current_user)):
    state = pipeline.invoke({"ticker": request.ticker.upper().strip()})
    if state.get("error"):
        raise HTTPException(status_code=404, detail=state["error"])
    return state["result"]


@app.post("/diagnostics/news-comparison")
def news_comparison(request: NewsComparisonRequest, user: dict = Depends(get_current_user)):
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


@app.get("/auth/avatar")
async def proxy_avatar(user: dict = Depends(get_current_user)):
    picture = user.get("picture", "")
    if not picture:
        raise HTTPException(status_code=404, detail="No avatar")
    async with httpx.AsyncClient() as client:
        r = await client.get(picture, follow_redirects=True, timeout=5)
    if r.status_code != 200:
        raise HTTPException(status_code=404, detail="Avatar unavailable")
    return Response(content=r.content, media_type=r.headers.get("content-type", "image/jpeg"))


@app.get("/login.html")
def serve_login():
    return FileResponse(PROJECT_ROOT / "login.html")


@app.get("/")
def serve_index():
    return FileResponse(PROJECT_ROOT / "index.html")
