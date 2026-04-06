"""
News API comparison runner.
Evaluates multiple news providers for ticker relevance using GPT classification.
"""
import logging

from qsf.common.utils import company_search_name
from qsf.ingestion.market import YFinanceMarketData
from qsf.ingestion.news import NewsAPIProvider
from qsf.ingestion.news_alphavantage import AlphaVantageNewsProvider
from qsf.ingestion.news_marketaux import MarketauxNewsProvider
from qsf.ingestion.news_massive import MassiveNewsProvider
from qsf.nlp.relevance import MAX_ARTICLES, RelevanceClassifier

logger = logging.getLogger(__name__)

_PROVIDER_REGISTRY = [
    ("NewsAPI",      NewsAPIProvider,          "1,000 req/day (30-day history)"),
    ("Massive",      MassiveNewsProvider,      "5 req/min, unlimited/day (2yr history)"),
    ("AlphaVantage", AlphaVantageNewsProvider, "25 req/day"),
    ("Marketaux",    MarketauxNewsProvider,    "100 req/day"),
]


def run_news_comparison(tickers: list[str], providers_config: list[str] | None = None) -> dict:
    """Compare news API providers across tickers using GPT-based relevance scoring."""
    classifier = RelevanceClassifier()
    results = []

    for ticker in tickers:
        full_name = YFinanceMarketData().get_company_name(ticker)
        company = company_search_name(full_name)

        provider_results = []
        for name, provider_cls, free_tier in _PROVIDER_REGISTRY:
            try:
                provider = provider_cls()
                articles = provider.get_articles(ticker, company_name=company)
                articles_to_classify = articles[:MAX_ARTICLES]
                relevance = classifier.classify(ticker, company, articles_to_classify)
                total = len(articles_to_classify)
                relevant = sum(relevance)
                precision = relevant / total if total > 0 else 0.0
                provider_results.append({
                    "name": name,
                    "total": total,
                    "relevant": relevant,
                    "precision": round(precision, 4),
                    "free_tier": free_tier,
                    "error": None,
                })
            except Exception as exc:
                logger.warning("Provider %s failed for %s: %s", name, ticker, exc)
                provider_results.append({
                    "name": name,
                    "total": 0,
                    "relevant": 0,
                    "precision": 0.0,
                    "free_tier": free_tier,
                    "error": f"{type(exc).__name__}: {exc}",
                })

        results.append({
            "ticker": ticker,
            "company": company,
            "providers": provider_results,
        })

    return {
        "tickers": tickers,
        "results": results,
    }
