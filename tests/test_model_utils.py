import numpy as np
from src.model_utils import calculate_credit_score, predict_optimal_loan


def test_credit_score_bounds_and_values():
    probs = np.array([0.0, 0.1, 0.5, 0.9, 1.0])
    scores = calculate_credit_score(probs)
    assert scores.min() >= 300 and scores.max() <= 850
    # Spot checks (approx)
    assert abs(scores[1] - 795) < 2  # prob=0.1 -> ~795
    assert abs(scores[3] - 355) < 2  # prob=0.9 -> ~355


def test_optimal_loan_rule_based():
    amount, duration = predict_optimal_loan(0.2)
    assert amount == 8000.0 and duration == 19
    amount2, duration2 = predict_optimal_loan(0.75)
    assert amount2 == 2500.0 and duration2 in (6, 7)  # rounding tolerance
