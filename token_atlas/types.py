"""Shared data contracts. A row always represents one token occurrence."""
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Article:
    id: str
    title: str
    text: str
    source: str = ""


@dataclass
class TokenResult:
    vectors: np.ndarray
    tokens: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.vectors.ndim != 2 or len(self.tokens) != len(self.vectors):
            raise ValueError("Token rows must match a two-dimensional vector matrix.")
        if not len(self.tokens) or not np.isfinite(self.vectors).all():
            raise ValueError("Token vectors must be nonempty and finite.")
