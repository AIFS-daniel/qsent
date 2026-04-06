"""
GPT-based relevance classifier for news articles.
Determines whether an article is specifically about a given company/ticker.
"""
import logging
import os

from openai import OpenAI

logger = logging.getLogger(__name__)

MAX_ARTICLES = 20  # hard cap per provider to limit GPT spend


class RelevanceClassifier:
    def classify(self, ticker: str, company_name: str, articles: list[dict]) -> list[bool]:
        """Classify each article as relevant (True) or not (False).

        Only classifies up to MAX_ARTICLES; positions beyond the cap default to False.
        Returns a list of the same length as articles.
        """
        if not articles:
            return []

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        articles_to_classify = articles[:MAX_ARTICLES]
        results: list[bool] = []

        for article in articles_to_classify:
            text = article.get("text", "")
            prompt = (
                f"Is this article specifically about {company_name} (ticker: ${ticker}) "
                f"and relevant to its stock, financials, or business? "
                f"Reply only 'yes' or 'no'.\n\n{text}"
            )
            try:
                response = client.chat.completions.create(
                    model="gpt-4.1-nano",  # update to latest available model as needed
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=3,
                    temperature=0,
                )
                answer = response.choices[0].message.content.strip().lower()
                results.append(answer == "yes")
            except Exception as exc:
                logger.warning("Relevance classification failed: %s", exc)
                results.append(False)

        # Positions beyond cap default to False
        beyond_cap = len(articles) - len(articles_to_classify)
        results.extend([False] * beyond_cap)
        return results
