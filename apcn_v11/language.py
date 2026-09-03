from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Counter as CounterType, DefaultDict, Dict, List, Optional, Sequence, Tuple
import json
import math

from apcn_v10.language_v101 import SemanticLanguageLearnerV101
from apcn_v10.language_session import AdaptiveLanguageSession
from apcn_v10.language_common import LanguageEpisode, tokenize


class ConstructionInducer:
    """Learns reusable sentence constructions from already-grounded cues.

    Content-bearing spans are abstracted to <COLOR>/<SHAPE>/<REL> using APCN's
    learned cue-feature associations. Remaining function-word structure is kept.
    Intent is then predicted from recurring abstract prefixes/skeletons. No
    English intent phrase is hardcoded here.
    """

    def __init__(self):
        self.pattern_intent: DefaultDict[str, CounterType[str]] = defaultdict(Counter)
        self.prefix_intent: DefaultDict[str, CounterType[str]] = defaultdict(Counter)
        self.observations = 0

    @staticmethod
    def _classify_span(learner, cue: str) -> Optional[str]:
        candidates = []
        for prefix, tag in (("relation:", "<REL>"), ("color:", "<COLOR>"), ("shape:", "<SHAPE>")):
            feat, score = learner.best_feature(cue, (prefix,))
            if feat is None:
                continue
            purity = learner.feature_purity(cue, feat)
            support = learner.cue_support(cue, feat)
            if support >= 4 and purity >= 0.62 and score > 0.06:
                candidates.append((score * purity, len(cue.split()), tag))
        if not candidates:
            return None
        candidates.sort(reverse=True)
        return candidates[0][2]

    def abstract_tokens(self, learner, tokens: Sequence[str]) -> List[str]:
        out: List[str] = []
        i = 0
        while i < len(tokens):
            found = None
            for n in range(min(5, len(tokens) - i), 0, -1):
                cue = " ".join(tokens[i:i+n])
                tag = self._classify_span(learner, cue)
                if tag is not None:
                    found = (n, tag)
                    break
            if found is None:
                out.append(tokens[i])
                i += 1
            else:
                n, tag = found
                out.append(tag)
                i += n
        return out

    @staticmethod
    def _collapse_entities(parts: Sequence[str]) -> List[str]:
        out: List[str] = []
        i = 0
        while i < len(parts):
            if i + 1 < len(parts) and parts[i] == "<COLOR>" and parts[i+1] == "<SHAPE>":
                out.append("<ENTITY>")
                i += 2
            else:
                out.append(parts[i])
                i += 1
        return out

    def abstract(self, learner, utterance: str) -> str:
        parts = self.abstract_tokens(learner, tokenize(utterance))
        parts = self._collapse_entities(parts)
        return " ".join(parts)

    def observe(self, learner, episode: LanguageEpisode) -> None:
        intent = episode.program.intent()
        if intent is None:
            return
        pattern = self.abstract(learner, episode.utterance)
        if not pattern:
            return
        self.pattern_intent[pattern][intent] += 1
        parts = pattern.split()
        # Learn reusable left-edge constructions, not just whole templates.
        for n in range(1, min(8, len(parts)) + 1):
            self.prefix_intent[" ".join(parts[:n])][intent] += 1
        self.observations += 1

    @staticmethod
    def _decision(counter: CounterType[str]) -> Tuple[Optional[str], float, int]:
        total = sum(counter.values())
        if total <= 0:
            return None, 0.0, 0
        ordered = counter.most_common(2)
        label, top = ordered[0]
        second = ordered[1][1] if len(ordered) > 1 else 0
        purity = top / total
        margin = (top - second) / total
        confidence = purity * (0.72 + 0.28 * min(1.0, math.log1p(total) / math.log(15.0))) * (0.7 + 0.3 * margin)
        return label, float(confidence), int(total)

    def predict(self, learner, tokens: Sequence[str]) -> Tuple[Optional[str], float, str]:
        pattern = " ".join(self._collapse_entities(self.abstract_tokens(learner, tokens)))
        if not pattern:
            return None, 0.0, ""
        candidates = []
        whole = self.pattern_intent.get(pattern)
        if whole:
            label, conf, support = self._decision(whole)
            if label:
                candidates.append((conf * 1.15, support, len(pattern.split()), label, pattern))
        parts = pattern.split()
        for n in range(1, min(8, len(parts)) + 1):
            prefix = " ".join(parts[:n])
            counter = self.prefix_intent.get(prefix)
            if not counter:
                continue
            label, conf, support = self._decision(counter)
            if label and support >= 4:
                length_bonus = 1.0 + 0.07 * (n - 1)
                candidates.append((conf * length_bonus, support, n, label, prefix))
        if not candidates:
            return None, 0.0, pattern
        candidates.sort(reverse=True)
        conf, support, _, label, evidence = candidates[0]
        return label, float(min(1.0, conf)), evidence

    def summary(self, limit: int = 12) -> Dict[str, object]:
        rows = []
        for pattern, counter in self.prefix_intent.items():
            label, conf, support = self._decision(counter)
            if label and support >= 4:
                rows.append((conf, support, pattern, label))
        rows.sort(reverse=True)
        return {
            "observations": self.observations,
            "patterns": len(self.pattern_intent),
            "prefix_constructions": len(self.prefix_intent),
            "strongest": [
                {"pattern": p, "intent": label, "confidence": conf, "support": support}
                for conf, support, p, label in rows[:limit]
            ],
        }

    def to_dict(self) -> Dict[str, object]:
        return {
            "observations": self.observations,
            "pattern_intent": {k: dict(v) for k, v in self.pattern_intent.items()},
            "prefix_intent": {k: dict(v) for k, v in self.prefix_intent.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "ConstructionInducer":
        obj = cls()
        obj.observations = int(data.get("observations", 0))
        obj.pattern_intent = defaultdict(Counter, {k: Counter(v) for k, v in data.get("pattern_intent", {}).items()})
        obj.prefix_intent = defaultdict(Counter, {k: Counter(v) for k, v in data.get("prefix_intent", {}).items()})
        return obj


class SemanticLanguageLearnerV11(SemanticLanguageLearnerV101):
    VERSION = "APCN-V0.11-SEMANTIC-LANGUAGE-MEMORY"

    def __init__(self):
        super().__init__()
        self.constructions = ConstructionInducer()

    def observe(self, episode: LanguageEpisode) -> None:
        super().observe(episode)
        # Observe after lexical/semantic evidence update so the abstractor can
        # use the newest grounded cue associations.
        self.constructions.observe(self, episode)

    def _best_intent(self, tokens: Sequence[str]) -> str:
        label, confidence, _ = self.constructions.predict(self, tokens)
        if label is not None and confidence >= 0.67:
            return label
        return super()._best_intent(tokens)

    def save(self, path: str | Path) -> None:
        super().save(path)
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        data["version"] = self.VERSION
        data["constructions"] = self.constructions.to_dict()
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SemanticLanguageLearnerV11":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        obj = cls()
        obj.episode_count = int(data.get("episode_count", 0))
        obj.feature_totals = Counter(data.get("feature_totals", {}))
        obj.cue_totals = Counter(data.get("cue_totals", {}))
        obj.cue_feature = defaultdict(Counter, {k: Counter(v) for k, v in data.get("cue_feature", {}).items()})
        obj.pos_feature = defaultdict(Counter, {k: Counter(v) for k, v in data.get("pos_feature", {}).items()})
        obj.pos_totals = Counter(data.get("pos_totals", {}))
        obj.constructions = ConstructionInducer.from_dict(data.get("constructions", {}))
        return obj


class AdaptiveLanguageSessionV11(AdaptiveLanguageSession):
    def __init__(self, seed: int = 11, learner: Optional[SemanticLanguageLearnerV11] = None):
        super().__init__(seed=seed, learner=learner or SemanticLanguageLearnerV11())
        self.last_step = None

    def step(self):
        row = super().step()
        self.last_step = row
        return row
