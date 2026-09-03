from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple
import json
import math


@dataclass
class ConceptNode:
    id: str
    label: str
    kind: str
    support: float = 0.0
    confidence: float = 0.0
    source: str = ""


@dataclass
class ConceptEdge:
    src: str
    dst: str
    relation: str
    weight: float = 0.0
    support: float = 0.0


class UnifiedConceptGraph:
    """Sparse explicit memory shared across perception, language and definitions.

    V0.11 deliberately keeps subsystem evidence separate at first and adds
    evidence-backed bridges between it. A lexical word, a visual concept and a
    semantic feature are not silently merged merely because their labels match.
    """

    VERSION = "APCN-V0.11-UNIFIED-CONCEPT-GRAPH"

    def __init__(self):
        self.nodes: Dict[str, ConceptNode] = {}
        self.edges: Dict[Tuple[str, str, str], ConceptEdge] = {}
        self.sync_count = 0

    def upsert_node(self, node_id: str, label: str, kind: str, *, support: float = 0.0,
                    confidence: float = 0.0, source: str = "") -> ConceptNode:
        old = self.nodes.get(node_id)
        if old is None:
            old = ConceptNode(node_id, label, kind, float(support), float(confidence), source)
            self.nodes[node_id] = old
        else:
            old.label = label or old.label
            old.kind = kind or old.kind
            old.support = max(old.support, float(support))
            old.confidence = max(old.confidence, float(confidence))
            old.source = source or old.source
        return old

    def strengthen(self, src: str, dst: str, relation: str, weight: float,
                   support: float = 1.0) -> ConceptEdge:
        key = (src, dst, relation)
        e = self.edges.get(key)
        w = max(0.0, min(1.0, float(weight)))
        if e is None:
            e = ConceptEdge(src, dst, relation, w, float(support))
            self.edges[key] = e
        else:
            # Evidence-weighted online update; no gradient/backpropagation.
            total = e.support + float(support)
            if total > 0:
                e.weight = (e.weight * e.support + w * float(support)) / total
            e.support = total
        return e

    def neighbors(self, node_id: str, relation: Optional[str] = None) -> List[ConceptEdge]:
        out = [e for e in self.edges.values() if e.src == node_id and (relation is None or e.relation == relation)]
        out.sort(key=lambda e: (e.weight, e.support), reverse=True)
        return out

    def sync_visual(self, learner) -> None:
        for token, stats in learner.token_stats.items():
            q = float(learner.concept_quality(token))
            wid = f"word:{token}"
            vid = f"visual:{token}"
            self.upsert_node(wid, token, "lexical", support=stats.count, confidence=q, source="visual")
            self.upsert_node(vid, token, "perceptual_concept", support=stats.count, confidence=q, source="visual")
            if q > 0:
                self.strengthen(wid, vid, "GROUNDS_IN", q, max(1.0, stats.count))
        for fam in learner.discover_families():
            fid = f"visual_family:{fam['id']}"
            self.upsert_node(fid, str(fam['id']), "discovered_family",
                             support=len(fam["members"]), confidence=float(fam["mean_quality"]), source="visual")
            for member in fam["members"]:
                self.strengthen(fid, f"visual:{member}", "HAS_MEMBER", float(fam["mean_similarity"]), 1.0)

    def _best_cues_for_feature(self, learner, feature: str, limit: int = 4):
        rows = []
        for cue in learner.cue_totals:
            support = learner.cue_support(cue, feature)
            if support < 3:
                continue
            purity = learner.feature_purity(cue, feature)
            score = learner.cue_score(cue, feature)
            if score > 0:
                rows.append((score * (0.5 + 0.5 * purity), purity, support, cue))
        rows.sort(reverse=True)
        return rows[:limit]

    def sync_language(self, learner) -> None:
        for feature, support in learner.feature_totals.items():
            fid = f"semantic:{feature}"
            kind = feature.split(":", 1)[0] if ":" in feature else "semantic"
            self.upsert_node(fid, feature, f"semantic_{kind}", support=support,
                             confidence=1.0 - math.exp(-float(support) / 30.0), source="language")
            for score, purity, cue_support, cue in self._best_cues_for_feature(learner, feature):
                wid = f"word:{cue}"
                self.upsert_node(wid, cue, "lexical", support=learner.cue_totals.get(cue, 0),
                                 confidence=purity, source="language")
                self.strengthen(wid, fid, "DENOTES", min(1.0, score), cue_support)

        # If the same learned lexical cue independently grounds in vision and
        # denotes a semantic feature, create an explicit equivalence hypothesis.
        for node_id in list(self.nodes):
            if not node_id.startswith("word:"):
                continue
            visual_edges = self.neighbors(node_id, "GROUNDS_IN")
            semantic_edges = self.neighbors(node_id, "DENOTES")
            for ve in visual_edges[:2]:
                for se in semantic_edges[:2]:
                    conf = math.sqrt(max(0.0, ve.weight * se.weight))
                    self.strengthen(ve.dst, se.dst, "SAME_CONCEPT_HYPOTHESIS", conf,
                                    min(ve.support, se.support))

    def sync_definitions(self, store) -> None:
        for name, rec in store.records.items():
            nid = f"concept:{name}"
            info = store.understanding(name)
            confidence = 1.0 if info.get("complete") else 0.45
            self.upsert_node(nid, name, f"defined_{rec.kind}", support=rec.support,
                             confidence=confidence, source="definitions")
            self.upsert_node(f"word:{name}", name, "lexical", support=rec.support,
                             confidence=confidence, source="definitions")
            self.strengthen(f"word:{name}", nid, "DENOTES", confidence, max(1, rec.support))
            for dep in rec.dependencies():
                did = f"concept:{dep}"
                self.upsert_node(did, dep, "concept", source="definitions")
                self.strengthen(nid, did, "DEPENDS_ON", 1.0, max(1, rec.support))

    def sync(self, *, visual=None, language=None, definitions=None) -> None:
        if visual is not None:
            self.sync_visual(visual)
        if language is not None:
            self.sync_language(language)
        if definitions is not None:
            self.sync_definitions(definitions)
        self.sync_count += 1

    def summary(self) -> Dict[str, object]:
        kinds: Dict[str, int] = {}
        rels: Dict[str, int] = {}
        for n in self.nodes.values():
            kinds[n.kind] = kinds.get(n.kind, 0) + 1
        for e in self.edges.values():
            rels[e.relation] = rels.get(e.relation, 0) + 1
        return {
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            "node_kinds": dict(sorted(kinds.items())),
            "edge_relations": dict(sorted(rels.items())),
            "sync_count": self.sync_count,
        }

    def save(self, path: str | Path) -> None:
        data = {
            "version": self.VERSION,
            "sync_count": self.sync_count,
            "nodes": [asdict(n) for n in self.nodes.values()],
            "edges": [asdict(e) for e in self.edges.values()],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "UnifiedConceptGraph":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        obj = cls()
        obj.sync_count = int(data.get("sync_count", 0))
        for row in data.get("nodes", []):
            n = ConceptNode(**row)
            obj.nodes[n.id] = n
        for row in data.get("edges", []):
            e = ConceptEdge(**row)
            obj.edges[(e.src, e.dst, e.relation)] = e
        return obj
