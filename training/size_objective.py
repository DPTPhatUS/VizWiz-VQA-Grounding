"""Training-count-based objective weights; never inspect validation or test labels."""

import math
from numbers import Real


def validate_group_weights(weights):
    if len(weights) != 3 or any(
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value < 0
        for value in weights
    ):
        raise ValueError("Use three finite, nonnegative group weights")
    values = tuple(float(value) for value in weights)
    if not math.isclose(sum(values), 1.0, rel_tol=0, abs_tol=1e-8):
        raise ValueError("Group weights must sum to one")
    return values


def resolve_group_weights(counts, focus_alpha):
    """Blend natural image frequencies with equal small/medium emphasis."""
    if (
        isinstance(focus_alpha, bool)
        or not isinstance(focus_alpha, Real)
        or not math.isfinite(focus_alpha)
        or not 0 <= focus_alpha <= 1
    ):
        raise ValueError("focus_alpha must be finite and in [0, 1]")
    if len(counts) != 3 or any(
        type(count) is not int or count <= 0 for count in counts
    ):
        raise ValueError("All three training groups need positive integer counts")
    natural = tuple(count / sum(counts) for count in counts)
    focus = (0.5, 0.5, 0.0)
    weights = tuple(
        (1 - focus_alpha) * p + focus_alpha * q for p, q in zip(natural, focus)
    )
    return validate_group_weights(natural), validate_group_weights(weights)
