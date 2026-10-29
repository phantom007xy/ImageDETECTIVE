"""Metrics that keep real and AI images separate (a single "accuracy" can hide false alarms)."""

import numpy as np


def auc(scores, is_ai):
    """Area under the ROC curve: 0.5 = chance, 1.0 = every AI image scores above every real one."""
    scores, is_ai = np.asarray(scores, float), np.asarray(is_ai, bool)
    # Average ranks for ties, otherwise tied values would be ordered by position.
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(counts) - counts + 1 + (counts - 1) / 2)[inverse]
    n_ai, n_real = is_ai.sum(), (~is_ai).sum()
    return (ranks[is_ai].sum() - n_ai * (n_ai + 1) / 2) / (n_ai * n_real)


def rates_at(scores, is_ai, threshold):
    """At one threshold: share of real photos called AI (false alarms) and share of AI images caught."""
    scores, is_ai = np.asarray(scores, float), np.asarray(is_ai, bool)
    called_ai = scores >= threshold
    false_alarms = called_ai[~is_ai].mean()
    caught = called_ai[is_ai].mean()
    return {
        "false_alarms": false_alarms,
        "caught": caught,
        "balanced_accuracy": ((1 - false_alarms) + caught) / 2,
    }


def caught_at_false_alarm_rate(scores, is_ai, max_false_alarms=0.05):
    """Share of AI images caught when the threshold is set so at most this share of real photos is flagged.

    The threshold is chosen on the same data it is measured on, so this shows what the features
    can do, not a ready threshold for the app (that comes from a separate calibration set).
    """
    scores, is_ai = np.asarray(scores, float), np.asarray(is_ai, bool)
    threshold = np.quantile(scores[~is_ai], 1 - max_false_alarms)
    return (scores[is_ai] > threshold).mean()
