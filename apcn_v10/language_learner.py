from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Counter as CounterType, DefaultDict, Dict, List, Optional, Sequence, Tuple
import json
import math

from .semantic import EntityRef, SemanticNode
from .language_common import LanguageEpisode, tokenize, ngrams

class SemanticLanguageLearnerV10:
    VERSION = "APCN-V0.10-SEMANTIC-LANGUAGE-MEMORY"

    def __init__(self):
        self.episode_count = 0
        self.feature_totals: CounterType[str] = Counter()
        self.cue_totals: CounterType[str] = Counter()
        self.cue_feature: DefaultDict[str, CounterType[str]] = defaultdict(Counter)
        self.pos_feature: DefaultDict[str, CounterType[str]] = defaultdict(Counter)
        self.pos_totals: CounterType[str] = Counter()
        self.discourse_focus: Optional[EntityRef] = None
        self.reference_feature = "reference:FOCUS"

    @staticmethod
    def _bucket(pos: float) -> str:
        if pos < 0.22:
            return "start"
        if pos > 0.78:
            return "end"
        return "mid"

    def observe(self, episode: LanguageEpisode) -> None:
        tokens = tokenize(episode.utterance)
        features = set(episode.program.features())
        if episode.discourse_focus is not None:
            features.add(self.reference_feature)
        self.episode_count += 1
        for feature in features:
            self.feature_totals[feature] += 1
        for cue, _, pos in ngrams(tokens, max_n=5):
            self.cue_totals[cue] += 1
            pk = f"{cue}@{self._bucket(pos)}"
            self.pos_totals[pk] += 1
            for feature in features:
                self.cue_feature[cue][feature] += 1
                self.pos_feature[pk][feature] += 1
        atom = episode.program.atom()
        if atom is not None and atom.subject is not None:
            self.discourse_focus = atom.subject
        else:
            atoms = [n for n in episode.program.walk() if n.op == "RELATION" and n.subject is not None]
            if atoms:
                self.discourse_focus = atoms[-1].subject

    def cue_score(self, cue: str, feature: str, position: Optional[str] = None) -> float:
        cue = cue.lower()
        if position is None:
            key, table, total = cue, self.cue_feature, float(self.cue_totals.get(cue, 0))
        else:
            key = f"{cue}@{position}"
            table, total = self.pos_feature, float(self.pos_totals.get(key, 0))
        joint = float(table.get(key, {}).get(feature, 0))
        feat_total = float(self.feature_totals.get(feature, 0))
        if joint <= 0 or total <= 0 or feat_total <= 0:
            return 0.0
        p_pos = joint / total
        negative_n = max(float(self.episode_count) - total, 1.0)
        p_neg = max(0.0, feat_total - joint) / negative_n
        discrimination = max(0.0, p_pos - p_neg)
        support = math.log1p(joint)
        phrase_bonus = 1.0 + 0.07 * (len(cue.split()) - 1)
        return float(discrimination * support * phrase_bonus)

    def feature_purity(self, cue: str, feature: str) -> float:
        cue = cue.lower()
        total = float(self.cue_totals.get(cue, 0))
        if total <= 0:
            return 0.0
        return float(self.cue_feature.get(cue, {}).get(feature, 0)) / total

    def cue_support(self, cue: str, feature: str) -> int:
        return int(self.cue_feature.get(cue.lower(), {}).get(feature, 0))

    def best_feature(self, cue: str, prefixes: Sequence[str], position: Optional[str] = None) -> Tuple[Optional[str], float]:
        candidates = [f for f in self.feature_totals if any(f.startswith(p) for p in prefixes)]
        scored = sorted(((self.cue_score(cue, f, position), f) for f in candidates), reverse=True)
        if not scored or scored[0][0] <= 0:
            return None, 0.0
        return scored[0][1], float(scored[0][0])

    def _entity_mentions(self, tokens: Sequence[str]) -> List[Tuple[int, int, EntityRef]]:
        colors: List[Tuple[int, str, float]] = []
        shapes: List[Tuple[int, str, float]] = []
        for i, token in enumerate(tokens):
            cf, cs = self.best_feature(token, ("color:",))
            sf, ss = self.best_feature(token, ("shape:",))
            if max(cs, ss) < 0.55:
                continue
            if cf is not None and cs >= ss * 1.12:
                colors.append((i, cf.split(":", 1)[1], cs))
            if sf is not None and ss >= cs * 1.12:
                shapes.append((i, sf.split(":", 1)[1], ss))
        mentions: List[Tuple[int, int, EntityRef]] = []
        used = set()
        for ci, color, _ in colors:
            choices = [(abs(si-ci), -score, si, shape) for si, shape, score in shapes if si not in used and abs(si-ci) <= 3]
            if not choices:
                continue
            _, _, si, shape = min(choices)
            used.add(si)
            mentions.append((min(ci, si), max(ci, si)+1, EntityRef(color, shape, len(mentions))))
        mentions.sort(key=lambda x: x[0])
        return mentions

    def _best_intent(self, tokens: Sequence[str]) -> str:
        totals: Dict[str, float] = defaultdict(float)
        n_tok = max(1, len(tokens))
        seen: Dict[Tuple[str, str], float] = {}
        for n in range(1, min(3, len(tokens)) + 1):
            for i in range(len(tokens)-n+1):
                cue = " ".join(tokens[i:i+n])
                center = (i + (n-1)/2.0) / max(1, n_tok-1)
                pos = self._bucket(center)
                for feat in self.feature_totals:
                    if not feat.startswith("intent:"):
                        continue
                    if self.cue_support(cue, feat) < 5:
                        continue
                    score = self.cue_score(cue, feat, pos)
                    if score > 0.04:
                        key = (cue, feat)
                        seen[key] = max(seen.get(key, 0.0), score)
        for (_, feat), score in seen.items():
            totals[feat] += score
        if not totals:
            return "ASSERT"
        ordered = sorted(((v, k) for k, v in totals.items()), reverse=True)
        if ordered[0][0] < 0.18:
            return "ASSERT"
        return ordered[0][1].split(":", 1)[1]

    def _best_relation(self, tokens: Sequence[str]) -> Optional[str]:
        rows = []
        for n in range(1, min(5, len(tokens)) + 1):
            for i in range(len(tokens)-n+1):
                cue = " ".join(tokens[i:i+n])
                feat, score = self.best_feature(cue, ("relation:",))
                if feat is not None:
                    rows.append((score, n, feat))
        if not rows:
            return None
        rows.sort(reverse=True)
        return rows[0][2].split(":", 1)[1]

    def _operator(self, tokens: Sequence[str], name: str) -> Optional[Tuple[int, int, str]]:
        target = f"operator:{name}"
        best = None
        for n in range(1, min(5, len(tokens)) + 1):
            for i in range(len(tokens)-n+1):
                cue = " ".join(tokens[i:i+n])
                feat, score = self.best_feature(cue, ("operator:",))
                if feat != target or score < 0.40:
                    continue
                if self.feature_purity(cue, target) < 0.58 or self.cue_support(cue, target) < 6:
                    continue
                candidate = (score, n, -i, i, i+n, cue)
                if best is None or candidate > best:
                    best = candidate
        return None if best is None else (best[3], best[4], best[5])

    def parse(self, utterance: str, discourse_focus: Optional[EntityRef] = None, allow_sequence: bool = True) -> Optional[SemanticNode]:
        tokens = tokenize(utterance)
        if not tokens:
            return None
        if allow_sequence:
            sep = self._operator(tokens, "SEQUENCE")
            if sep is not None:
                i, j, _ = sep
                a = self.parse(" ".join(tokens[:i]), discourse_focus, False)
                focus = discourse_focus
                if a is not None:
                    atoms = [n for n in a.walk() if n.op == "RELATION" and n.subject is not None]
                    if atoms:
                        focus = atoms[-1].subject
                b = self.parse(" ".join(tokens[j:]), focus, False)
                if a is not None and b is not None:
                    return SemanticNode("SEQUENCE", children=(a, b))

        relation = self._best_relation(tokens)
        if relation is None:
            return None
        intent = self._best_intent(tokens)
        mentions = self._entity_mentions(tokens)
        group = self._operator(tokens, "GROUP")
        neg = self._operator(tokens, "NEGATE")

        ref_present = False
        for n in (1, 2):
            for i in range(len(tokens)-n+1):
                cue = " ".join(tokens[i:i+n])
                feat, score = self.best_feature(cue, ("reference:",))
                if (feat == self.reference_feature and score >= 0.70
                        and self.feature_purity(cue, self.reference_feature) >= 0.62
                        and self.cue_support(cue, self.reference_feature) >= 6):
                    ref_present = True
                    break
            if ref_present:
                break
        focus = discourse_focus or self.discourse_focus

        if group is not None and len(mentions) >= 3:
            obj = mentions[-1][2]
            children = tuple(
                SemanticNode.relation_node(relation, m[2], obj, "GOAL")
                for m in mentions[:-1]
            )
            node = SemanticNode("GROUP", children=children)
        else:
            if ref_present and focus is not None:
                subject = focus
                obj = mentions[-1][2] if mentions else None
            elif len(mentions) >= 2:
                subject, obj = mentions[0][2], mentions[1][2]
            else:
                return None
            if obj is None:
                return None
            subject = EntityRef(subject.color, subject.shape, 0)
            obj = EntityRef(obj.color, obj.shape, 1 if (subject.color, subject.shape) != (obj.color, obj.shape) else 0)
            node = SemanticNode.relation_node(relation, subject, obj, intent)
        if neg is not None:
            node = SemanticNode("NEGATE", children=(node,))
        return node

    def save(self, path: str | Path) -> None:
        data = {
            "version": self.VERSION,
            "episode_count": self.episode_count,
            "feature_totals": dict(self.feature_totals),
            "cue_totals": dict(self.cue_totals),
            "cue_feature": {k: dict(v) for k, v in self.cue_feature.items()},
            "pos_feature": {k: dict(v) for k, v in self.pos_feature.items()},
            "pos_totals": dict(self.pos_totals),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SemanticLanguageLearnerV10":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        obj = cls()
        obj.episode_count = int(data.get("episode_count", 0))
        obj.feature_totals = Counter(data.get("feature_totals", {}))
        obj.cue_totals = Counter(data.get("cue_totals", {}))
        obj.cue_feature = defaultdict(Counter, {k: Counter(v) for k, v in data.get("cue_feature", {}).items()})
        obj.pos_feature = defaultdict(Counter, {k: Counter(v) for k, v in data.get("pos_feature", {}).items()})
        obj.pos_totals = Counter(data.get("pos_totals", {}))
        return obj
