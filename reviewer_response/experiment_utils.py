"""Small dependency-free helpers shared by reviewer-response experiments."""

from __future__ import annotations

import math
import re
import statistics


SPATIAL_RELATION_PATTERNS = (
    r"\bleft of\b",
    r"\bright of\b",
    r"\bin front of\b",
    r"\bbehind\b",
    r"\babove\b",
    r"\bbelow\b",
    r"\bon top of\b",
    r"\bunder(?:neath)?\b",
    r"\bnext to\b",
    r"\bnear(?:est)?\b",
    r"\bfar(?:thest)?\b",
    r"\bbetween\b",
    r"\binside\b",
    r"\bwithin\b",
    r"\boutside\b",
    r"\bclosest to\b",
    r"\bfarthest from\b",
    r"\bopposite\b",
    r"\badjacent to\b",
)


def count_spatial_relations(text, patterns=SPATIAL_RELATION_PATTERNS):
    """Count non-overlapping explicit relation phrases in an instruction."""
    lowered = str(text).lower()
    return sum(len(re.findall(pattern, lowered)) for pattern in patterns)


def spatial_complexity_group(relation_count):
    """Map a relation count to the pre-registered 0, 1, or 2+ group."""
    if relation_count <= 0:
        return "0"
    if relation_count == 1:
        return "1"
    return "2+"


def summarize_values(values):
    """Return descriptive statistics and a normal-approximation 95% CI."""
    values = [float(value) for value in values]
    count = len(values)
    if not values:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "ci95_low": None,
            "ci95_high": None,
        }
    mean = statistics.mean(values)
    std = statistics.stdev(values) if count > 1 else 0.0
    half_width = 1.96 * std / math.sqrt(count) if count > 1 else 0.0
    return {
        "count": count,
        "mean": mean,
        "std": std,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
    }
