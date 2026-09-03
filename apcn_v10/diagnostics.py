from __future__ import annotations

from typing import Dict, List, Optional, Set
import math

from .language_common import tokenize


def _node(nodes: List[Dict[str, object]], node_id: str, label: str, kind: str, firing: float) -> None:
    if any(str(n.get("id")) == node_id for n in nodes):
        return
    nodes.append({"id": node_id, "label": label, "kind": kind, "firing": max(0.0, min(1.0, float(firing)))})


def language_trace(learner, utterance: str, program=None) -> Dict[str, object]:
    nodes: List[Dict[str, object]] = []
    edges: List[Dict[str, object]] = []
    tokens = tokenize(utterance)
    features = list(getattr(learner, "feature_totals", {}).keys())
    for i, token in enumerate(tokens[:12]):
        wid = f"w:{i}:{token}"
        _node(nodes, wid, token, "word", 0.75)
        scored = []
        for feat in features:
            try:
                score = float(learner.cue_score(token, feat))
            except Exception:
                score = 0.0
            if score > 0:
                scored.append((score, feat))
        scored.sort(reverse=True)
        for score, feat in scored[:2]:
            sid = f"s:{feat}"
            fire = 1.0 - math.exp(-max(0.0, score))
            _node(nodes, sid, feat.split(":", 1)[-1], "semantic", fire)
            edges.append({"src": wid, "dst": sid, "weight": min(1.0, fire)})
    if program is not None:
        for feat in program.features():
            sid = f"s:{feat}"
            _node(nodes, sid, feat.split(":", 1)[-1], "semantic", 1.0)
    return {"nodes": nodes, "edges": edges}


def definition_trace(store, concept: Optional[str]) -> Dict[str, object]:
    nodes: List[Dict[str, object]] = []
    edges: List[Dict[str, object]] = []
    if not concept:
        return {"nodes": nodes, "edges": edges}
    root = concept.strip().lower()
    seen: Set[str] = set()

    def walk(name: str, depth: int = 0) -> None:
        if name in seen or depth > 5:
            return
        seen.add(name)
        rec = store.records.get(name)
        complete = bool(store.understanding(name).get("complete")) if rec is not None else False
        _node(nodes, f"c:{name}", name, "concept", 1.0 if depth == 0 else (0.85 if complete else 0.55))
        if rec is None or rec.definition is None:
            return
        op_id = f"op:{name}:{rec.definition.op}"
        _node(nodes, op_id, rec.definition.op, "operator", 0.9)
        edges.append({"src": f"c:{name}", "dst": op_id, "weight": 0.9})
        for dep in sorted(rec.dependencies()):
            walk(dep, depth + 1)
            edges.append({"src": op_id, "dst": f"c:{dep}", "weight": 0.75})
    walk(root)
    return {"nodes": nodes, "edges": edges}
