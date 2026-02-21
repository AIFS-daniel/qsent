"""
HuggingFace FinBERT sentiment model.
"""
import logging
import os

import requests

logger = logging.getLogger(__name__)

FINBERT_URL = "https://router.huggingface.co/hf-inference/models/ProsusAI/finbert"
SENTIMENT_MAP = {"positive": 1, "negative": -1, "neutral": 0}


class FinBERTModel:
    def score(self, texts: list[str]) -> list[float]:
        """Score sentiment for each text individually.

        The HuggingFace Inference API for FinBERT only processes the first item
        when given a batch, so we send one text per request.
        """
        headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}"}
        scores = []
        failed = 0

        for idx, text in enumerate(texts):
            try:
                response = requests.post(FINBERT_URL, headers=headers, json={"inputs": text})
                response.raise_for_status()
                result = response.json()
                if not isinstance(result, list) or not result:
                    logger.warning(
                        "FinBERTModel.score: item %d/%d returned unexpected response: %s",
                        idx + 1, len(texts), str(result)[:200],
                    )
                    failed += 1
                    continue
                # Single-text requests return [[{label_dicts}]] — unwrap the outer list
                label_scores = result[0] if isinstance(result[0], list) else result
                top = max(label_scores, key=lambda x: x["score"])
                scores.append(SENTIMENT_MAP.get(top["label"].lower(), 0) * top["score"])
            except requests.HTTPError as e:
                logger.warning(
                    "FinBERTModel.score: item %d/%d failed — HTTP %s: %s",
                    idx + 1, len(texts), e.response.status_code, e.response.text[:200],
                )
                failed += 1
            except Exception as e:
                logger.warning(
                    "FinBERTModel.score: item %d/%d failed — %s: %s",
                    idx + 1, len(texts), type(e).__name__, e,
                )
                failed += 1

        if failed:
            logger.warning(
                "FinBERTModel.score: %d/%d items failed to score",
                failed, len(texts),
            )

        return scores
