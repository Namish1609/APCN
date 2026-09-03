from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Sequence
import json

from .language_learner import SemanticLanguageLearnerV10
from .language_session import AdaptiveLanguageSession, GeneratedLanguageTestReport, run_generated_language_test
from .definitions import ConceptStore, DefinitionCurriculum


class SemanticLanguageLearnerV101(SemanticLanguageLearnerV10):
    """V0.10.1 intent-construction fix.

    V0.10 accumulated many weak 1-3 token intent associations. On held-out
    templates, generic cues could collectively overpower a highly diagnostic
    sentence prefix. V0.10.1 explicitly evaluates learned prefix constructions
    up to five tokens, using purity/support/contrast and a length bonus. This is
    still learned from cue statistics; no English phrase is hardcoded to an
    intent label.
    """

    VERSION = "APCN-V0.10.1-SEMANTIC-LANGUAGE-MEMORY"

    def _best_intent(self, tokens: Sequence[str]) -> str:
        if not tokens:
            return "ASSERT"
        intent_features = [f for f in self.feature_totals if f.startswith("intent:")]
        if not intent_features:
            return "ASSERT"

        prefix_candidates = []
        max_n = min(5, len(tokens))
        for n in range(1, max_n + 1):
            cue = " ".join(tokens[:n])
            scores = []
            for feat in intent_features:
                support = self.cue_support(cue, feat)
                if support < 4:
                    continue
                purity = self.feature_purity(cue, feat)
                score = self.cue_score(cue, feat, "start")
                if score <= 0:
                    continue
                value = score * (0.55 + 0.65 * purity) * (1.0 + 0.16 * (n - 1))
                scores.append((value, purity, support, feat))
            scores.sort(reverse=True)
            if scores:
                top = scores[0]
                second = scores[1][0] if len(scores) > 1 else 0.0
                margin = top[0] / max(second, 1e-9)
                prefix_candidates.append((top[0], top[1], margin, n, top[2], top[3]))
        prefix_candidates.sort(reverse=True)
        for value, purity, margin, n, support, feat in prefix_candidates:
            if purity >= 0.64 and support >= 5 and margin >= 1.12 and value >= 0.10:
                return feat.split(":", 1)[1]

        totals: Dict[str, float] = defaultdict(float)
        best_per_feature: Dict[str, list] = defaultdict(list)
        n_tok = max(1, len(tokens))
        for n in range(1, min(5, len(tokens)) + 1):
            for i in range(len(tokens) - n + 1):
                cue = " ".join(tokens[i:i+n])
                center = (i + (n - 1) / 2.0) / max(1, n_tok - 1)
                pos = self._bucket(center)
                for feat in intent_features:
                    support = self.cue_support(cue, feat)
                    if support < 5:
                        continue
                    purity = self.feature_purity(cue, feat)
                    score = self.cue_score(cue, feat, pos)
                    if score <= 0.035 or purity < 0.38:
                        continue
                    weight = score * (0.45 + 0.55 * purity) * (1.0 + 0.08 * (n - 1))
                    if i == 0:
                        weight *= 1.25
                    best_per_feature[feat].append(weight)
        for feat, vals in best_per_feature.items():
            vals.sort(reverse=True)
            totals[feat] = sum(vals[:4])
        if not totals:
            return "ASSERT"
        ordered = sorted(((v, k) for k, v in totals.items()), reverse=True)
        if ordered[0][0] < 0.10:
            return "ASSERT"
        if len(ordered) > 1 and ordered[0][0] < ordered[1][0] * 1.06:
            return "ASSERT"
        return ordered[0][1].split(":", 1)[1]


class AdaptiveLanguageSessionV101(AdaptiveLanguageSession):
    def __init__(self, seed: int = 10, learner: Optional[SemanticLanguageLearnerV101] = None):
        super().__init__(seed=seed, learner=learner or SemanticLanguageLearnerV101())

    def save(self, output_dir: str | Path = "outputs/v0_10_1") -> Path:
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        memory = out / "language_memory_v0_10_1.json"
        self.learner.save(memory)
        return memory


class CognitiveSessionV101:
    def __init__(self, seed: int = 10):
        self.seed = seed
        self.language = AdaptiveLanguageSessionV101(seed)
        self.concepts = ConceptStore()
        self.definitions = DefinitionCurriculum(self.concepts)
        self.test_history = []

    def test_language(self, samples: int = 600) -> GeneratedLanguageTestReport:
        rep = run_generated_language_test(
            self.language.learner,
            samples=samples,
            seed=self.seed + 10101 + len(self.test_history) * 31,
        )
        self.test_history.append({
            "episodes": self.language.learner.episode_count,
            "exact": rep.exact_accuracy,
            "intent": rep.intent_accuracy,
            "relation": rep.relation_accuracy,
            "operator": rep.operator_accuracy,
        })
        return rep

    def save(self, output_dir: str | Path = "outputs/v0_10_1") -> Dict[str, str]:
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        lang = self.language.save(out)
        concepts = out / "concept_store_v0_10_1.json"; self.concepts.save(concepts)
        state = out / "session_v0_10_1.json"
        state.write_text(json.dumps({
            "version": "0.10.1",
            "seed": self.seed,
            "language_episodes": self.language.learner.episode_count,
            "definition_count": self.concepts.definition_count,
            "test_history": self.test_history,
        }, indent=2), encoding="utf-8")
        return {"language": str(lang), "concepts": str(concepts), "session": str(state)}
