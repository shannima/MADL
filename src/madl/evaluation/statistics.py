"""Small exact tests used by the paired decision-path analyses."""

from __future__ import annotations

import math


def exact_mcnemar_p(corrected: int, worsened: int) -> float:
    corrected = int(corrected)
    worsened = int(worsened)
    if corrected < 0 or worsened < 0:
        raise ValueError("McNemar counts cannot be negative")
    discordant = corrected + worsened
    if discordant == 0:
        return 1.0
    lower = min(corrected, worsened)
    tail = sum(math.comb(discordant, value) for value in range(lower + 1)) / (2**discordant)
    return min(1.0, 2.0 * tail)
