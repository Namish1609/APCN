from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Counter as CounterType, DefaultDict, Dict, List, Optional, Sequence, Tuple
import json
import math
import re

from .semantic import EntityRef, SemanticNode
from .teacher import LanguageEpisode

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+(?:'[a-z]+)?")


def tokenize(text: str) -> List[str]:
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]


def ngrams(tokens: Sequence[str], max_n: int = 3) -> List[Tuple[str, int, float]]:
    out: List[Tuple[str, int, float]] = []
    n_tok = max(1, len(tokens))
    for n in range(1, min(max_n, len(tokens)) + 1):
        for i in range(len(tokens) - n + 1):
            phrase = " ".join(tokens[i:i+n])
            center = (i + (n - 1) / 2.0) / max(1, n_tok - 1)
            out.append((phrase, n, center))
    return out


@dataclass
class CueScore:
    cue: str
    feature: str
    score: float
    support: int


class SemanticLanguageLearner:
    VERSION = "APCN-V0.9-SEMANTIC-LANGUAGE-MEMORY"

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
        if pos < 0.24:
            return "start"
        if pos > 0.76:
            return "end"
        return "mid"

    def observe(self, episode: LanguageEpisode) -> None:
        tokens = tokenize(episode.utterance)
        feats = set(episode.program.features())
        if episode.discourse_focus is not None:
            feats.add(self.reference_feature)
        self.episode_count += 1
        for feat in feats:
            self.feature_totals[feat] += 1
        for cue, n, pos in ngrams(tokens, max_n=3):
            self.cue_totals[cue] += 1
            pos_key = f"{cue}@{self._bucket(pos)}"
            self.pos_totals[pos_key] += 1
            for feat in feats:
                self.cue_feature[cue][feat] += 1
                self.pos_feature[pos_key][feat] += 1
        atom = episode.program.atom()
        if atom is not None and atom.subject is not None:
            self.discourse_focus = atom.subject
        elif episode.program.op in {"GROUP", "SEQUENCE"} and episode.program.children:
            last_atom = episode.program.children[-1].atom()
            if last_atom is not None and last_atom.subject is not None:
                self.discourse_focus = last_atom.subject

    def cue_score(self, cue: str, feature: str, position: Optional[str] = None) -> float:
        cue = cue.lower()
        if position is None:
            key, table, total = cue, self.cue_feature, float(self.cue_totals.get(cue, 0))
        else:
            key, table, total = f"{cue}@{position}", self.pos_feature, float(self.pos_totals.get(f"{cue}@{position}", 0))
        joint = float(table.get(key, {}).get(feature, 0))
        feat_total = float(self.feature_totals.get(feature, 0))
        if joint <= 0 or total <= 0 or feat_total <= 0:
            return 0.0
        # Contrastive association: the feature must be more probable with this
        # cue than without it. This suppresses generic words that occur with
        # nearly every color, shape, relation or intent.
        p_pos = joint / total
        neg_n = max(float(self.episode_count) - total, 1.0)
        p_neg = max(0.0, feat_total - joint) / neg_n
        discrimination = max(0.0, p_pos - p_neg)
        support = math.log1p(joint)
        phrase_bonus = 1.0 + 0.10 * (len(cue.split()) - 1)
        return float(discrimination * support * phrase_bonus)

    def best_feature(self, cue: str, prefixes: Sequence[str], position: Optional[str] = None) -> Tuple[Optional[str], float]:
        candidates = [f for f in self.feature_totals if any(f.startswith(p) for p in prefixes)]
        scored = [(self.cue_score(cue, f, position), f) for f in candidates]
        scored.sort(reverse=True)
        if not scored or scored[0][0] <= 0:
            return None, 0.0
        return scored[0][1], float(scored[0][0])

    def top_cues(self, feature: str, limit: int = 8) -> List[CueScore]:
        rows = []
        for cue, counts in self.cue_feature.items():
            if feature in counts:
                rows.append(CueScore(cue, feature, self.cue_score(cue, feature), counts[feature]))
        rows.sort(key=lambda x: (x.score, x.support), reverse=True)
        return rows[:limit]

    def _entity_mentions(self, tokens: Sequence[str]) -> List[Tuple[int, int, EntityRef]]:
        color_hits: List[Tuple[int, str, float]] = []
        shape_hits: List[Tuple[int, str, float]] = []
        for i, tok in enumerate(tokens):
            cf, cs = self.best_feature(tok, ("color:",))
            sf, ss = self.best_feature(tok, ("shape:",))
            best = max(cs, ss)
            if best < 1.00:
                continue
            if cf is not None and cs >= ss * 1.25:
                color_hits.append((i, cf.split(":", 1)[1], cs))
            elif sf is not None and ss >= cs * 1.25:
                shape_hits.append((i, sf.split(":", 1)[1], ss))
        mentions: List[Tuple[int, int, EntityRef]] = []
        used_shapes = set()
        for ci, color, _ in color_hits:
            choices = [(abs(si-ci), -ss, si, shape) for si, shape, ss in shape_hits if si not in used_shapes and 0 < abs(si-ci) <= 2]
            if not choices:
                continue
            _, _, si, shape = min(choices)
            used_shapes.add(si)
            mentions.append((min(ci, si), max(ci, si)+1, EntityRef(color, shape, len(mentions))))
        mentions.sort(key=lambda x: x[0])
        return mentions

    def _best_intent(self, tokens: Sequence[str]) -> str:
        # Intent is learned as a construction/position phenomenon. For example,
        # the same surface cue can mean something different at clause start.
        best: Dict[str, float] = {}
        n_tok = max(1, len(tokens))
        for n in range(1, min(3, len(tokens)) + 1):
            for i in range(len(tokens) - n + 1):
                cue = " ".join(tokens[i:i+n])
                center = (i + (n - 1) / 2) / max(1, n_tok - 1)
                pos = self._bucket(center)
                vals = []
                for feat in self.feature_totals:
                    if feat.startswith("intent:"):
                        vals.append((self.cue_score(cue, feat, pos), feat))
                vals.sort(reverse=True)
                if not vals or vals[0][0] < 0.20:
                    continue
                second = vals[1][0] if len(vals) > 1 else 0.0
                if vals[0][0] < second * 1.12:
                    continue
                feat = vals[0][1]
                best[feat] = max(best.get(feat, 0.0), vals[0][0])
        return max(best.items(), key=lambda kv: kv[1])[0].split(":", 1)[1] if best else "ASSERT"

    def _best_relation(self, tokens: Sequence[str]) -> Optional[str]:
        rows = []
        for n in range(1, min(3, len(tokens)) + 1):
            for i in range(len(tokens) - n + 1):
                cue = " ".join(tokens[i:i+n])
                feat, score = self.best_feature(cue, ("relation:",))
                if feat is not None:
                    rows.append((score, n, feat))
        rows.sort(reverse=True)
        return None if not rows else rows[0][2].split(":", 1)[1]

    def _operator(self, tokens: Sequence[str], name: str) -> Optional[Tuple[int, int, str]]:
        target = f"operator:{name}"
        best = None
        for n in range(1, min(3, len(tokens)) + 1):
            for i in range(len(tokens) - n + 1):
                cue = " ".join(tokens[i:i+n])
                feat, score = self.best_feature(cue, ("operator:",))
                if feat != target or score < 0.65:
                    continue
                cand = (score, i, i+n, cue)
                if best is None or cand > best:
                    best = cand
        return None if best is None else (best[1], best[2], best[3])

    def parse(self, utterance: str, discourse_focus: Optional[EntityRef] = None, _allow_sequence: bool = True) -> Optional[SemanticNode]:
        tokens = tokenize(utterance)
        if not tokens:
            return None

        if _allow_sequence:
            sep = self._operator(tokens, "SEQUENCE")
            if sep is not None:
                i, j, _ = sep
                left, right = " ".join(tokens[:i]), " ".join(tokens[j:])
                a = self.parse(left, discourse_focus, _allow_sequence=False)
                focus = discourse_focus
                if a is not None:
                    atoms = [n for n in a.walk() if n.op == "RELATION" and n.subject is not None]
                    if atoms:
                        focus = atoms[-1].subject
                b = self.parse(right, focus, _allow_sequence=False)
                if a and b:
                    return SemanticNode("SEQUENCE", children=(a, b))

        neg = self._operator(tokens, "NEGATE")
        group = self._operator(tokens, "GROUP")
        relation = self._best_relation(tokens)
        intent = self._best_intent(tokens)
        mentions = self._entity_mentions(tokens)
        focus = discourse_focus or self.discourse_focus
        ref_present = False
        for tok in tokens:
            feat, score = self.best_feature(tok, ("reference:",))
            if feat == self.reference_feature and score >= 2.00:
                ref_present = True
                break
        if relation is None:
            return None

        if group is not None and len(mentions) >= 3:
            obj = mentions[-1][2]
            children = tuple(SemanticNode.relation_node(relation, m[2], obj, intent="GOAL") for m in mentions[:-1])
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
            node = SemanticNode.relation_node(relation, subject, obj, intent=intent)
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
    def load(cls, path: str | Path) -> "SemanticLanguageLearner":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        obj = cls()
        obj.episode_count = int(data.get("episode_count", 0))
        obj.feature_totals = Counter(data.get("feature_totals", {}))
        obj.cue_totals = Counter(data.get("cue_totals", {}))
        obj.cue_feature = defaultdict(Counter, {k: Counter(v) for k, v in data.get("cue_feature", {}).items()})
        obj.pos_feature = defaultdict(Counter, {k: Counter(v) for k, v in data.get("pos_feature", {}).items()})
        obj.pos_totals = Counter(data.get("pos_totals", {}))
        return obj
