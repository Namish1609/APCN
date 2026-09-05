from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Counter as CounterType, DefaultDict, Dict, Iterable, List, Tuple
import json
import re

_TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+(?:\.\d+)?", re.I)
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")


class EnglishExposureMemory:
    """Bounded non-neural surface-language exposure memory.

    Corpus exposure is deliberately separated from semantic grounding. Seeing a
    word or phrase makes it *familiar*, not understood. The memory keeps counts
    and bounded local context aggregates; raw source sentences are not archived.
    """

    VERSION = "APCN-V0.15-ENGLISH-EXPOSURE-MEMORY"

    def __init__(self, max_vocab: int = 50000, max_ngrams: int = 120000, max_context_per_word: int = 32):
        self.max_vocab = int(max_vocab)
        self.max_ngrams = int(max_ngrams)
        self.max_context_per_word = int(max_context_per_word)
        self.words: CounterType[str] = Counter()
        self.ngrams: CounterType[str] = Counter()
        self.contexts: DefaultDict[str, CounterType[str]] = defaultdict(Counter)
        self.documents_seen = 0
        self.sentences_seen = 0
        self.tokens_seen = 0

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return [m.group(0).lower() for m in _TOKEN.finditer(text)]

    def ingest(self, text: str) -> Dict[str, int]:
        before_tokens = self.tokens_seen
        before_sentences = self.sentences_seen
        self.documents_seen += 1
        for raw_sentence in _SENTENCE.split(str(text)):
            toks = self.tokenize(raw_sentence)
            if not toks:
                continue
            self.sentences_seen += 1
            self.tokens_seen += len(toks)
            self.words.update(toks)
            for n in (2, 3, 4):
                if len(toks) < n:
                    continue
                self.ngrams.update(" ".join(toks[i:i+n]) for i in range(len(toks)-n+1))
            for i, word in enumerate(toks):
                lo = max(0, i-2); hi = min(len(toks), i+3)
                for j in range(lo, hi):
                    if i != j:
                        self.contexts[word][toks[j]] += 1
        self._prune()
        return {
            "documents_added": 1,
            "sentences_added": self.sentences_seen - before_sentences,
            "tokens_added": self.tokens_seen - before_tokens,
            "vocabulary": len(self.words),
            "ngrams": len(self.ngrams),
        }

    def _prune_counter(self, counter: CounterType[str], limit: int) -> None:
        if len(counter) <= limit:
            return
        for key, _ in sorted(counter.items(), key=lambda kv: (kv[1], kv[0]))[:len(counter)-limit]:
            del counter[key]

    def _prune(self) -> None:
        self._prune_counter(self.words, self.max_vocab)
        self._prune_counter(self.ngrams, self.max_ngrams)
        # Drop contexts for words that are no longer in vocabulary, then cap each
        # retained context row independently.
        for word in list(self.contexts):
            if word not in self.words:
                del self.contexts[word]
                continue
            row = self.contexts[word]
            if len(row) > self.max_context_per_word:
                for key, _ in sorted(row.items(), key=lambda kv: (kv[1], kv[0]))[:len(row)-self.max_context_per_word]:
                    del row[key]

    def familiarity(self, word: str) -> int:
        return int(self.words.get(str(word).lower(), 0))

    def top_context(self, word: str, limit: int = 8) -> List[Tuple[str, int]]:
        return self.contexts.get(str(word).lower(), Counter()).most_common(limit)

    def coverage(self, text: str, semantic_terms: Iterable[str] = ()) -> Dict[str, object]:
        toks = self.tokenize(text)
        if not toks:
            return {"tokens": 0, "surface_familiar": 0.0, "semantic_anchor": 0.0, "unknown": []}
        semantic = {str(x).strip().lower() for x in semantic_terms if str(x).strip()}
        familiar = sum(1 for t in toks if t in self.words)
        anchored = sum(1 for t in toks if t in semantic)
        unknown = sorted({t for t in toks if t not in self.words})
        return {
            "tokens": len(toks),
            "surface_familiar": familiar/len(toks),
            "semantic_anchor": anchored/len(toks),
            "unknown": unknown[:64],
        }

    def summary(self, limit: int = 20) -> Dict[str, object]:
        return {
            "version": self.VERSION,
            "documents_seen": self.documents_seen,
            "sentences_seen": self.sentences_seen,
            "tokens_seen": self.tokens_seen,
            "vocabulary": len(self.words),
            "max_vocab": self.max_vocab,
            "ngrams": len(self.ngrams),
            "max_ngrams": self.max_ngrams,
            "top_words": self.words.most_common(limit),
            "top_phrases": self.ngrams.most_common(limit),
            "raw_documents_retained": 0,
            "raw_sentences_retained": 0,
        }

    def to_dict(self) -> Dict[str, object]:
        return {
            "version": self.VERSION,
            "max_vocab": self.max_vocab,
            "max_ngrams": self.max_ngrams,
            "max_context_per_word": self.max_context_per_word,
            "documents_seen": self.documents_seen,
            "sentences_seen": self.sentences_seen,
            "tokens_seen": self.tokens_seen,
            "words": dict(self.words),
            "ngrams": dict(self.ngrams),
            "contexts": {w: dict(row) for w, row in self.contexts.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "EnglishExposureMemory":
        obj = cls(
            int(data.get("max_vocab", 50000)),
            int(data.get("max_ngrams", 120000)),
            int(data.get("max_context_per_word", 32)),
        )
        obj.documents_seen = int(data.get("documents_seen", 0))
        obj.sentences_seen = int(data.get("sentences_seen", 0))
        obj.tokens_seen = int(data.get("tokens_seen", 0))
        obj.words.update({str(k): int(v) for k, v in dict(data.get("words", {})).items()})
        obj.ngrams.update({str(k): int(v) for k, v in dict(data.get("ngrams", {})).items()})
        for word, row in dict(data.get("contexts", {})).items():
            obj.contexts[str(word)].update({str(k): int(v) for k, v in dict(row).items()})
        obj._prune()
        return obj

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "EnglishExposureMemory":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
