"""
LangGraph pipeline for sentiment analysis.

Graph:
    fetch_market_data → fetch_news → fetch_reddit → score_sentiment → aggregate
"""
from typing import Any, Optional
from typing_extensions import TypedDict

from langgraph.graph import END, StateGraph

from qsf.agents.nodes import (
    aggregate,
    fetch_market_data,
    fetch_news,
    fetch_reddit,
    score_sentiment,
)


class PipelineState(TypedDict, total=False):
    ticker: str
    stock_df: Any
    news_items: list
    reddit_items: list
    scored_items: list
    result: Optional[dict]
    error: Optional[str]


def build_pipeline():
    graph = StateGraph(PipelineState)

    graph.add_node("fetch_market_data", fetch_market_data)
    graph.add_node("fetch_news", fetch_news)
    graph.add_node("fetch_reddit", fetch_reddit)
    graph.add_node("score_sentiment", score_sentiment)
    graph.add_node("aggregate", aggregate)

    graph.set_entry_point("fetch_market_data")
    graph.add_edge("fetch_market_data", "fetch_news")
    graph.add_edge("fetch_news", "fetch_reddit")
    graph.add_edge("fetch_reddit", "score_sentiment")
    graph.add_edge("score_sentiment", "aggregate")
    graph.add_edge("aggregate", END)

    return graph.compile()


pipeline = build_pipeline()
