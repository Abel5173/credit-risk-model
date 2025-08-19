import numpy as np
from functools import lru_cache
from typing import Tuple
from src.utils import timer_decorator


@lru_cache(maxsize=128)
def _calculate_credit_score_cached(probs_key: Tuple[float, ...], min_score: int = 300, max_score: int = 850):
    arr = np.array(probs_key, dtype=float)
    scores = np.clip(min_score + (max_score - min_score)
                     * (1.0 - arr), min_score, max_score)
    return scores


@timer_decorator
def calculate_credit_score(probs: np.ndarray, min_score: int = 300, max_score: int = 850) -> np.ndarray:
    """Convert risk probabilities into credit scores in the 300-850 range.

    Uses a cached backend keyed by a rounded tuple of probabilities for speed on repeated inputs.
    """
    key = tuple(round(float(p), 6) for p in np.asarray(probs).tolist())
    return _calculate_credit_score_cached(key, min_score=min_score, max_score=max_score)


@timer_decorator
@lru_cache(maxsize=128)
def predict_optimal_loan(prob: float, max_amount: float = 10000.0, max_duration_months: int = 24) -> Tuple[float, int]:
    """Simple rule-based optimal loan recommendation based on risk probability.

    Returns (amount, duration_months).
    """
    risk_factor = float(prob)
    amount = round(max_amount * (1.0 - risk_factor), 2)
    duration = int(round(max_duration_months * (1.0 - risk_factor)))
    return amount, duration
