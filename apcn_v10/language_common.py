from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple
import re

from .semantic import EntityRef, SemanticNode

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+(?:'[a-z]+)?|[?]")

def tokenize(text: str) -> List[str]:
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]

def ngrams(tokens: Sequence[str], max_n: int = 5) -> Iterable[Tuple[str, int, float]]:
    n_tok = max(1, len(tokens))
    for n in range(1, min(max_n, len(tokens)) + 1):
        for i in range(len(tokens) - n + 1):
            phrase = " ".join(tokens[i:i+n])
            center = (i + (n - 1) / 2.0) / max(1, n_tok - 1)
            yield phrase, n, center

@dataclass
class LanguageEpisode:
    utterance: str
    program: SemanticNode
    skill: str
    discourse_focus: Optional[EntityRef] = None
    held_out_template: bool = False

@dataclass
class SkillState:
    attempts: int = 0
    correct: int = 0
    ema: float = 0.0

    def update(self, ok: bool) -> None:
        self.attempts += 1
        self.correct += int(ok)
        target = 1.0 if ok else 0.0
        alpha = 0.055 if self.attempts > 20 else 0.12
        self.ema = (1.0 - alpha) * self.ema + alpha * target

    @property
    def accuracy(self) -> float:
        return self.correct / self.attempts if self.attempts else 0.0
