"""
HuggingFace FinBERT sentiment model.
"""
import logging
import os
import time

import requests

logger = logging.getLogger(__name__)

FINBERT_URL = "https://router.huggingface.co/hf-inference/models/ProsusAI/finbert"
SENTIMENT_MAP = {"positive": 1, "negative": -1, "neutral": 0}
HF_REQUEST_TIMEOUT = 10  # seconds — applies to health check and per-item scoring requests


class FinBERTModel:
    def score(self, texts: list[str]) -> list[float]:
        """Score sentiment for each text individually.

        The HuggingFace Inference API for FinBERT only processes the first item
        when given a batch, so we send one text per request.
        """
        headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}"}
        scores = []
        failed = 0

        logger.info("FinBERTModel.score: scoring %d items via HuggingFace API (1 request per item)", len(texts))
        if not texts:
            return scores
        try:
            t0 = time.monotonic()
            probe = requests.post(FINBERT_URL, headers=headers, json={"inputs": "test"}, timeout=HF_REQUEST_TIMEOUT)
            elapsed = time.monotonic() - t0
            logger.info("FinBERTModel.score: HuggingFace health check — HTTP %d in %.2fs", probe.status_code, elapsed)
        except Exception as e:
            logger.warning("FinBERTModel.score: HuggingFace health check failed — %s: %s", type(e).__name__, e)
        for idx, text in enumerate(texts):
            if idx % 10 == 0:
                logger.info("FinBERTModel.score: progress %d/%d", idx, len(texts))
            try:
                response = requests.post(FINBERT_URL, headers=headers, json={"inputs": text}, timeout=HF_REQUEST_TIMEOUT)
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
                score = SENTIMENT_MAP.get(top["label"].lower(), 0) * top["score"]
                logger.info(
                    "FinBERTModel.score: item %d/%d — %s (%.3f)",
                    idx + 1, len(texts), top["label"].lower(), score,
                )
                scores.append(score)
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
