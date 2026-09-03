from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Optional, Sequence, Tuple
import json
import math

from apcn_v10.language_common import LanguageEpisode, tokenize
from apcn_v11.language import SemanticLanguageLearnerV11, AdaptiveLanguageSessionV11


class AdaptiveConstructionCalibrator:
    """Bounded recent-evidence overlay for stale language constructions.

    Lifetime APCN language counts remain the stable long-term baseline. This
    overlay stores exponentially decayed intent evidence for abstract sentence
    constructions so targeted corrections can matter even after many thousands
    of older episodes. It stores no raw sentences and performs no gradient step.
    """

    VERSION = "APCN-V0.12-CONSTRUCTION-CALIBRATOR"

    def __init__(self, decay: float = .985, max_patterns: int = 4096):
        self.decay = float(decay)
        self.max_patterns = int(max_patterns)
        self.rows: DefaultDict[str, Dict[str, float]] = defaultdict(dict)
        self.touches: DefaultDict[str, int] = defaultdict(int)
        self.observations = 0

    def _update(self, key: str, label: str) -> None:
        row = self.rows[key]
        for k in list(row):
            row[k] *= self.decay
            if row[k] < 1e-5:
                row.pop(k, None)
        row[label] = row.get(label, 0.0) + 1.0
        self.touches[key] += 1

    def observe(self, pattern: str, intent: str) -> None:
        if not pattern or not intent:
            return
        parts = pattern.split()
        self._update(pattern, intent)
        for n in range(1, min(8, len(parts)) + 1):
            self._update(" ".join(parts[:n]), intent)
        self.observations += 1
        if len(self.rows) > self.max_patterns:
            # Remove least-used, weakest rows. This is explicit bounded memory,
            # not an episode archive.
            ranked = sorted(self.rows, key=lambda k: (self.touches[k], sum(self.rows[k].values())))
            for key in ranked[:len(self.rows)-self.max_patterns]:
                self.rows.pop(key, None); self.touches.pop(key, None)

    @staticmethod
    def _decision(row: Dict[str, float]) -> Tuple[Optional[str], float, float]:
        total = float(sum(row.values()))
        if total <= 0:
            return None, 0.0, 0.0
        ranked = sorted(((v, k) for k, v in row.items()), reverse=True)
        top, label = ranked[0]
        second = ranked[1][0] if len(ranked) > 1 else 0.0
        purity = top / total
        margin = (top-second) / total
        support = 1.0 - math.exp(-total / 5.0)
        return label, float(purity * (.72 + .28*margin) * support), total

    def predict(self, pattern: str) -> Tuple[Optional[str], float, str]:
        if not pattern:
            return None, 0.0, ""
        parts = pattern.split(); candidates = []
        for n in range(1, min(8, len(parts)) + 1):
            key = " ".join(parts[:n])
            row = self.rows.get(key)
            if row:
                label, conf, support = self._decision(row)
                if label is not None:
                    candidates.append((conf * (1.0 + .055*(n-1)), support, n, label, key))
        row = self.rows.get(pattern)
        if row:
            label, conf, support = self._decision(row)
            if label is not None:
                candidates.append((conf*1.12, support, len(parts), label, pattern))
        if not candidates:
            return None, 0.0, pattern
        candidates.sort(reverse=True)
        conf, support, _, label, evidence = candidates[0]
        if support < 2.4:
            return None, float(conf), evidence
        return label, float(min(1.0, conf)), evidence

    def summary(self) -> Dict[str, object]:
        strong = []
        for key, row in self.rows.items():
            label, conf, support = self._decision(row)
            if label and support >= 2.4:
                strong.append((conf, support, key, label))
        strong.sort(reverse=True)
        return {
            "observations": self.observations,
            "patterns": len(self.rows),
            "max_patterns": self.max_patterns,
            "strongest": [
                {"pattern": p, "intent": label, "confidence": conf, "recent_support": support}
                for conf, support, p, label in strong[:16]
            ],
        }

    def to_dict(self) -> Dict[str, object]:
        return {
            "version": self.VERSION,
            "decay": self.decay,
            "max_patterns": self.max_patterns,
            "observations": self.observations,
            "rows": {k: dict(v) for k, v in self.rows.items()},
            "touches": dict(self.touches),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "AdaptiveConstructionCalibrator":
        obj = cls(float(data.get("decay", .985)), int(data.get("max_patterns", 4096)))
        obj.observations = int(data.get("observations", 0))
        obj.rows = defaultdict(dict, {k: {x: float(y) for x, y in v.items()}
                                      for k, v in data.get("rows", {}).items()})
        obj.touches = defaultdict(int, {k: int(v) for k, v in data.get("touches", {}).items()})
        return obj


class SemanticLanguageLearnerV12(SemanticLanguageLearnerV11):
    VERSION = "APCN-V0.12-SEMANTIC-LANGUAGE-MEMORY"

    def __init__(self):
        super().__init__()
        self.adaptive_constructions = AdaptiveConstructionCalibrator()

    def observe(self, episode: LanguageEpisode) -> None:
        super().observe(episode)
        intent = episode.program.intent()
        if intent is not None:
            pattern = self.constructions.abstract(self, episode.utterance)
            self.adaptive_constructions.observe(pattern, intent)

    def _best_intent(self, tokens: Sequence[str]) -> str:
        pattern = " ".join(self.constructions._collapse_entities(
            self.constructions.abstract_tokens(self, tokens)))
        label, confidence, _ = self.adaptive_constructions.predict(pattern)
        if label is not None and confidence >= .57:
            return label
        return super()._best_intent(tokens)

    def save(self, path: str | Path) -> None:
        super().save(path)
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        data["version"] = self.VERSION
        data["adaptive_constructions"] = self.adaptive_constructions.to_dict()
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SemanticLanguageLearnerV12":
        # Reuse V0.11's loader for all lifetime cue/construction statistics, then
        # transfer those aggregates into this class. Older checkpoints simply
        # start with an empty correction overlay.
        base = SemanticLanguageLearnerV11.load(path)
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        obj = cls()
        obj.episode_count = base.episode_count
        obj.feature_totals = base.feature_totals
        obj.cue_totals = base.cue_totals
        obj.cue_feature = base.cue_feature
        obj.pos_feature = base.pos_feature
        obj.pos_totals = base.pos_totals
        obj.constructions = base.constructions
        obj.adaptive_constructions = AdaptiveConstructionCalibrator.from_dict(
            data.get("adaptive_constructions", {}))
        return obj


class AdaptiveLanguageSessionV12(AdaptiveLanguageSessionV11):
    def __init__(self, seed: int = 12, learner: Optional[SemanticLanguageLearnerV12] = None):
        super().__init__(seed=seed, learner=learner or SemanticLanguageLearnerV12())
