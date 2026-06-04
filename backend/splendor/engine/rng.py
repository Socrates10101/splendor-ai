"""Deterministic RNG so a game is fully reproducible from a single seed.

We deliberately use our own tiny PRNG (splitmix64 -> xorshift) instead of the
stdlib ``random`` module so the shuffle is stable across Python versions and
trivially portable to other languages if the engine is ever re-implemented.
"""
from __future__ import annotations

_MASK64 = (1 << 64) - 1


class Rng:
    def __init__(self, seed: int):
        self.state = seed & _MASK64

    def _next(self) -> int:
        # splitmix64
        self.state = (self.state + 0x9E3779B97F4A7C15) & _MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
        return (z ^ (z >> 31)) & _MASK64

    def randint(self, n: int) -> int:
        """Uniform integer in [0, n)."""
        if n <= 0:
            raise ValueError("n must be positive")
        return self._next() % n

    def shuffle(self, items: list) -> list:
        """Fisher-Yates in place; returns the same list for convenience."""
        for i in range(len(items) - 1, 0, -1):
            j = self._next() % (i + 1)
            items[i], items[j] = items[j], items[i]
        return items
