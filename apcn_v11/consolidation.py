from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence
import math
import numpy as np


@dataclass
class LearningPrescription:
    domain: str
    target: str
    contrast: str
    priority: float
    reason: str
    strategy: str


class ConsolidationEngine:
    """Non-gradient error reduction and memory consolidation.

    The objective is diagnostic, not differentiable training loss. APCN measures
    recurring confusion, concept ambiguity and memory complexity, then chooses
    targeted contrasts/constructions to gather next.
    """

    def __init__(self, error_memory=None):
        self.error_memory = error_memory

    @staticmethod
    def _pair_separation(learner, a: str, b: str) -> float:
        sa = learner.token_stats.get(a)
        sb = learner.token_stats.get(b)
        if sa is None or sb is None or sa.count < 3 or sb.count < 3:
            return 0.0
        pooled = 0.5 * (sa.var + sb.var) + 0.05 * learner.global_stats.var + 1e-8
        d2 = (sa.mean - sb.mean) ** 2 / pooled
        rel = learner.candidate_relevance([a, b])
        if np.any(rel > 0):
            w = rel / max(float(rel.sum()), 1e-12)
            return float(np.sqrt(max(0.0, np.sum(w * d2))))
        return float(np.sqrt(max(0.0, np.mean(d2))))

    def visual_ambiguities(self, learner, candidates: Sequence[str], limit: int = 8) -> List[Dict[str, object]]:
        rows = []
        names = [str(x).lower() for x in candidates if x in learner.token_stats]
        for i, a in enumerate(names):
            for b in names[i+1:]:
                sep = self._pair_separation(learner, a, b)
                # Smaller separation = more likely confusion.
                ambiguity = 1.0 / (1.0 + sep)
                rows.append({"a": a, "b": b, "separation": sep, "ambiguity": ambiguity})
        rows.sort(key=lambda r: r["ambiguity"], reverse=True)
        return rows[:limit]

    def prescriptions(self, *, visual_learner=None, colors: Sequence[str] = (), shapes: Sequence[str] = (),
                      language_learner=None, limit: int = 12) -> List[LearningPrescription]:
        out: List[LearningPrescription] = []

        if self.error_memory is not None:
            for e in self.error_memory.top(limit=limit):
                p = min(1.0, 0.20 + e.recent_weight / max(4.0, self.error_memory.total_weight(e.domain)))
                if e.domain.startswith("visual"):
                    strategy = "generate minimal-pair scenes with nuisance factors matched, then gradually randomize"
                else:
                    strategy = "generate contrastive semantic episodes differing only in the failed construction/operator"
                out.append(LearningPrescription(
                    e.domain, e.truth, e.predicted, p,
                    f"repeated confusion {e.truth} -> {e.predicted} (count={e.count}, recent={e.recent_weight:.2f})",
                    strategy,
                ))

        if visual_learner is not None:
            for group, names in (("visual_color", colors), ("visual_shape", shapes)):
                for row in self.visual_ambiguities(visual_learner, names, limit=6):
                    out.append(LearningPrescription(
                        group, row["a"], row["b"], float(row["ambiguity"]),
                        f"compact statistics give low class separation ({row['separation']:.3f})",
                        "teach balanced A/B minimal contrasts before adding rotation, clutter and lighting variation",
                    ))

        if language_learner is not None and hasattr(language_learner, "constructions"):
            summary = language_learner.constructions.summary(limit=30)
            # Weak learned constructions receive targeted paraphrase evidence.
            for row in summary.get("strongest", []):
                conf = float(row["confidence"])
                if conf < 0.85:
                    out.append(LearningPrescription(
                        "language_construction", str(row["intent"]), str(row["pattern"]),
                        1.0 - conf,
                        f"construction confidence only {conf:.3f} with support {row['support']}",
                        "generate paraphrases preserving semantic intent while varying content words and relation aliases",
                    ))

        # De-duplicate and rank.
        best: Dict[tuple, LearningPrescription] = {}
        for p in out:
            key = (p.domain, p.target, p.contrast)
            if key not in best or p.priority > best[key].priority:
                best[key] = p
        rows = list(best.values())
        rows.sort(key=lambda x: x.priority, reverse=True)
        return rows[:limit]

    def objective(self, *, error_rate: float, active_edges: int, ambiguous_pairs: int,
                  edge_scale: float = 10000.0) -> Dict[str, float]:
        """MDL-like diagnostic objective; lower is better.

        This is used to compare memory/consolidation policies. It is never
        differentiated and does not update parameters by gradient descent.
        """
        prediction_term = float(max(0.0, min(1.0, error_rate)))
        complexity_term = math.log1p(max(0, active_edges)) / math.log1p(edge_scale)
        ambiguity_term = 1.0 - math.exp(-max(0, ambiguous_pairs) / 12.0)
        total = prediction_term + 0.08 * complexity_term + 0.18 * ambiguity_term
        return {
            "prediction_error": prediction_term,
            "memory_complexity": complexity_term,
            "concept_ambiguity": ambiguity_term,
            "total": float(total),
        }

    @staticmethod
    def as_rows(items: Sequence[LearningPrescription]) -> List[Dict[str, object]]:
        return [asdict(x) for x in items]
