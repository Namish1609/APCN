from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional
import numpy as np

from apcn_v07.learner import GroundedConceptLearner


@dataclass
class ConceptActivation:
    token: str
    score: float
    quality: float
    support: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "token": self.token,
            "score": self.score,
            "quality": self.quality,
            "support": self.support,
        }


class EnhancedConceptLearner(GroundedConceptLearner):
    """
    V0.8 adds introspection and structure discovery on top of V0.7's compact
    cross-situational learner. No backpropagation and no retained episode archive.
    """

    VERSION = "APCN-V0.8-CONCEPT-MEMORY"

    def role_guess(self, token: str) -> str:
        token = token.lower()
        stats = self.token_stats.get(token)
        if stats is None:
            return "unknown"
        q = self.concept_quality(token)
        if stats.count < 5:
            return "uncertain"
        if q >= 0.16:
            return "visually_grounded"
        if stats.count >= 20:
            return "structural_or_abstract"
        return "uncertain"

    def semantic_activations(
        self,
        x: np.ndarray,
        tokens: Optional[Iterable[str]] = None,
        min_quality: float = 0.03,
        top_k: int = 12,
    ) -> List[ConceptActivation]:
        candidates = list(tokens) if tokens is not None else list(self.token_stats)
        out: List[ConceptActivation] = []
        for token in sorted(set(t.lower() for t in candidates)):
            stats = self.token_stats.get(token)
            if stats is None:
                continue
            q = self.concept_quality(token)
            if q < min_quality:
                continue
            out.append(ConceptActivation(token, self.similarity(token, x), q, stats.count))
        out.sort(key=lambda a: (a.score, a.quality, a.support), reverse=True)
        return out[:top_k]

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na <= 1e-12 or nb <= 1e-12:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def relevance_similarity(self, a: str, b: str) -> float:
        return self._cosine(self.relevance(a), self.relevance(b))

    def discover_families(
        self,
        min_quality: float = 0.12,
        min_support: int = 18,
        similarity_threshold: float = 0.72,
        min_family_size: int = 2,
    ) -> List[Dict[str, object]]:
        """Discover token families from similarity of learned relevance profiles.

        Labels such as "color" or "shape" are never supplied here. A family is
        simply a connected component in relevance-profile similarity space.
        """
        eligible = [
            token for token, stats in self.token_stats.items()
            if stats.count >= min_support and self.concept_quality(token) >= min_quality
        ]
        if not eligible:
            return []
        graph: Dict[str, set[str]] = {t: set() for t in eligible}
        for i, a in enumerate(eligible):
            for b in eligible[i + 1:]:
                sim = self.relevance_similarity(a, b)
                if sim >= similarity_threshold:
                    graph[a].add(b)
                    graph[b].add(a)

        seen = set()
        families: List[Dict[str, object]] = []
        for root in eligible:
            if root in seen:
                continue
            stack = [root]
            component = []
            seen.add(root)
            while stack:
                cur = stack.pop()
                component.append(cur)
                for nxt in graph[cur]:
                    if nxt not in seen:
                        seen.add(nxt)
                        stack.append(nxt)
            if len(component) < min_family_size:
                continue
            component.sort(key=lambda t: self.concept_quality(t), reverse=True)
            pair_sims = []
            for i, a in enumerate(component):
                for b in component[i + 1:]:
                    pair_sims.append(self.relevance_similarity(a, b))
            families.append({
                "id": f"family_{len(families)+1:02d}",
                "members": component,
                "mean_similarity": float(np.mean(pair_sims)) if pair_sims else 1.0,
                "mean_quality": float(np.mean([self.concept_quality(t) for t in component])),
            })
        families.sort(key=lambda f: (len(f["members"]), f["mean_quality"]), reverse=True)
        return families

    def activation_trace(
        self,
        x: np.ndarray,
        utterance: str = "",
        top_tokens: int = 10,
        top_features_per_token: int = 5,
    ) -> Dict[str, object]:
        """Return a neuron-like sparse firing graph for UI visualization."""
        utterance_tokens = set(self.tokenize(utterance))
        acts = self.semantic_activations(x, top_k=top_tokens)
        act_map = {a.token: a for a in acts}
        all_tokens = list(dict.fromkeys(list(utterance_tokens) + [a.token for a in acts]))
        nodes: List[Dict[str, object]] = []
        edges: List[Dict[str, object]] = []
        used_features: Dict[int, float] = {}

        for token in all_tokens:
            stats = self.token_stats.get(token)
            q = self.concept_quality(token) if stats is not None else 0.0
            score = act_map[token].score if token in act_map else 0.0
            firing = float(np.clip(max(score, 0.25 * q if token in utterance_tokens else 0.0), 0.0, 1.0))
            nodes.append({
                "id": f"word:{token}",
                "label": token,
                "kind": "word" if token in utterance_tokens else "concept",
                "firing": firing,
                "quality": q,
                "role": self.role_guess(token),
            })
            if stats is None:
                continue
            rel = self.relevance(token)
            if not np.any(rel > 0):
                continue
            indices = np.argsort(rel)[::-1][:top_features_per_token]
            max_rel = float(rel[indices[0]]) if len(indices) and rel[indices[0]] > 0 else 1.0
            for idx in indices:
                if rel[idx] <= 0:
                    continue
                weight = float(np.clip(rel[idx] / max_rel, 0.0, 1.0))
                used_features[int(idx)] = max(used_features.get(int(idx), 0.0), weight * max(firing, q))
                edges.append({
                    "src": f"word:{token}",
                    "dst": f"feature:{int(idx)}",
                    "weight": weight,
                    "kind": "grounds_in",
                })

        for idx, firing in used_features.items():
            nodes.append({
                "id": f"feature:{idx}",
                "label": f"f{idx:03d}",
                "kind": "feature",
                "firing": float(np.clip(firing, 0.0, 1.0)),
            })

        families = self.discover_families()
        for fam in families[:4]:
            fid = str(fam["id"])
            nodes.append({
                "id": fid,
                "label": fid,
                "kind": "family",
                "firing": float(np.clip(fam["mean_quality"], 0.0, 1.0)),
            })
            for member in fam["members"]:
                if any(n["id"] == f"word:{member}" for n in nodes):
                    edges.append({"src": fid, "dst": f"word:{member}", "weight": float(fam["mean_similarity"]), "kind": "family_member"})

        return {"nodes": nodes, "edges": edges, "families": families}

    def memory_summary(self) -> Dict[str, object]:
        grounded = []
        structural = []
        uncertain = []
        for token, stats in sorted(self.token_stats.items()):
            row = {
                "token": token,
                "support": stats.count,
                "quality": self.concept_quality(token),
                "role": self.role_guess(token),
            }
            if row["role"] == "visually_grounded":
                grounded.append(row)
            elif row["role"] == "structural_or_abstract":
                structural.append(row)
            else:
                uncertain.append(row)
        grounded.sort(key=lambda r: r["quality"], reverse=True)
        structural.sort(key=lambda r: r["support"], reverse=True)
        return {
            "episode_count": self.episode_count,
            "vocabulary_size": len(self.token_stats),
            "grounded": grounded,
            "structural_or_abstract": structural,
            "uncertain": uncertain,
            "families": self.discover_families(),
        }
