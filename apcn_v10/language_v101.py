from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence
import json

from .language_learner import SemanticLanguageLearnerV10
from .language_session import AdaptiveLanguageSession, GeneratedLanguageTestReport, run_generated_language_test
from .definitions import ConceptStore, DefinitionCurriculum


class SemanticLanguageLearnerV101(SemanticLanguageLearnerV10):
    """V0.10.1 semantic-language patch.

    Intent is inferred from learned construction evidence rather than literal
    English keys. Punctuation can become a learned structural cue, and cues
    that already function strongly as references/operators cannot independently
    hijack clause intent.
    """

    VERSION = "APCN-V0.10.1-SEMANTIC-LANGUAGE-MEMORY"
    QUESTION_CUE = "__surface_question_mark__"

    def __init__(self):
        super().__init__()
        self._surface_question = False

    def observe(self, episode) -> None:
        super().observe(episode)
        if "?" in episode.utterance:
            cue = self.QUESTION_CUE
            self.cue_totals[cue] += 1
            pkey = f"{cue}@start"
            self.pos_totals[pkey] += 1
            features = set(episode.program.features())
            if episode.discourse_focus is not None:
                features.add(self.reference_feature)
            for feature in features:
                self.cue_feature[cue][feature] += 1
                self.pos_feature[pkey][feature] += 1

    def _learned_question_intent(self) -> Optional[str]:
        cue = self.QUESTION_CUE
        candidates = []
        for feat in self.feature_totals:
            if not feat.startswith("intent:"):
                continue
            support = self.cue_support(cue, feat)
            if support < 5:
                continue
            purity = self.feature_purity(cue, feat)
            score = self.cue_score(cue, feat, "start")
            if score > 0:
                candidates.append((score * (0.5 + 0.5 * purity), purity, support, feat))
        candidates.sort(reverse=True)
        if not candidates:
            return None
        top = candidates[0]
        second = candidates[1][0] if len(candidates) > 1 else 0.0
        if top[1] >= 0.72 and top[2] >= 5 and top[0] >= second * 1.10:
            return top[3].split(":", 1)[1]
        return None

    def _functional_conflict(self, cue: str) -> float:
        scores = []
        for feat in self.feature_totals:
            if feat.startswith("reference:") or feat.startswith("operator:"):
                scores.append(self.feature_purity(cue, feat))
        return max(scores, default=0.0)

    def _intent_candidate(self, cue: str, position: str, length: int, at_start: bool):
        if self._functional_conflict(cue) >= 0.58:
            return None
        rows = []
        for feat in self.feature_totals:
            if not feat.startswith("intent:"):
                continue
            support = self.cue_support(cue, feat)
            if support < 5:
                continue
            purity = self.feature_purity(cue, feat)
            score = self.cue_score(cue, feat, position)
            if score <= 0:
                continue
            value = score * (0.45 + 0.75 * purity) * (1.0 + 0.18 * (length - 1))
            if at_start:
                value *= 1.18
            rows.append((value, purity, support, feat))
        rows.sort(reverse=True)
        if not rows:
            return None
        top = rows[0]
        second = rows[1][0] if len(rows) > 1 else 0.0
        margin = top[0] / max(second, 1e-9)
        return top[0], top[1], margin, top[2], top[3]

    def _best_intent(self, tokens: Sequence[str]) -> str:
        if self._surface_question:
            learned = self._learned_question_intent()
            if learned is not None:
                return learned
        if not tokens:
            return "ASSERT"

        candidates = []
        for n in range(1, min(5, len(tokens)) + 1):
            cue = " ".join(tokens[:n])
            row = self._intent_candidate(cue, "start", n, True)
            if row is not None:
                candidates.append((*row, n))
        candidates.sort(reverse=True)
        for value, purity, margin, support, feat, n in candidates:
            if purity >= 0.68 and support >= 6 and margin >= 1.10 and value >= 0.10:
                return feat.split(":", 1)[1]

        # A polyfunctional discourse/reference cue at sentence start may hide a
        # command cue one or two positions later. Internal fallback is therefore
        # permitted only to establish GOAL. It cannot infer QUERY from an
        # interior copula or other generic token; questions use learned surface
        # and clause-level evidence above.
        goal_feature = "intent:GOAL"
        early = []
        limit = min(4, len(tokens))
        for i in range(limit):
            for n in range(1, min(3, len(tokens) - i) + 1):
                cue = " ".join(tokens[i:i+n])
                if self._functional_conflict(cue) >= 0.58:
                    continue
                support = self.cue_support(cue, goal_feature)
                if support < 5:
                    continue
                purity = self.feature_purity(cue, goal_feature)
                center = (i + (n - 1) / 2.0) / max(1, len(tokens) - 1)
                score = self.cue_score(cue, goal_feature, self._bucket(center))
                if score <= 0:
                    continue
                value = score * (0.45 + 0.75 * purity) * (1.0 + 0.12 * (n - 1))
                early.append((value, purity, support, n, -i))
        early.sort(reverse=True)
        if early:
            value, purity, support, n, neg_i = early[0]
            if purity >= 0.65 and support >= 5 and value >= 0.08:
                return "GOAL"

        return "ASSERT"

    def parse(self, utterance: str, discourse_focus=None, allow_sequence: bool = True):
        previous = self._surface_question
        self._surface_question = "?" in utterance
        try:
            return super().parse(utterance, discourse_focus, allow_sequence)
        finally:
            self._surface_question = previous


class AdaptiveLanguageSessionV101(AdaptiveLanguageSession):
    def __init__(self, seed: int = 10, learner: Optional[SemanticLanguageLearnerV101] = None):
        super().__init__(seed=seed, learner=learner or SemanticLanguageLearnerV101())
        self.last_step = None

    def step(self):
        result = super().step()
        self.last_step = result
        return result

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
