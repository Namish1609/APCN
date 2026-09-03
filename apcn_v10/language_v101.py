from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Sequence
import json

from .language_learner import SemanticLanguageLearnerV10
from .language_session import AdaptiveLanguageSession, GeneratedLanguageTestReport, run_generated_language_test
from .definitions import ConceptStore, DefinitionCurriculum


class SemanticLanguageLearnerV101(SemanticLanguageLearnerV10):
    """V0.10.1 semantic-language patch.

    Intent is inferred from learned construction evidence rather than literal
    English keys. V0.10.1 also lets punctuation become a learned structural
    cue: if question-mark-bearing demonstrations consistently have one intent,
    the learner can acquire that association statistically.
    """

    VERSION = "APCN-V0.10.1-SEMANTIC-LANGUAGE-MEMORY"
    QUESTION_CUE = "__surface_question_mark__"

    def __init__(self):
        super().__init__()
        self._surface_question = False

    def observe(self, episode) -> None:
        super().observe(episode)
        # The base tokenizer intentionally strips punctuation. In V0.10.1 we
        # retain a generic surface-event cue for '?'. It is NOT mapped to QUERY
        # by code: its semantic association is accumulated from demonstrations.
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

    def _best_intent(self, tokens: Sequence[str]) -> str:
        if self._surface_question:
            learned = self._learned_question_intent()
            if learned is not None:
                return learned

        intent_features = [f for f in self.feature_totals if f.startswith("intent:")]
        if not tokens or not intent_features:
            return "ASSERT"

        # Prefer a compact learned construction beginning at clause start.
        # Longer, purer constructions dominate generic words such as "the".
        candidates = []
        for n in range(1, min(5, len(tokens)) + 1):
            cue = " ".join(tokens[:n])
            scored = []
            for feat in intent_features:
                support = self.cue_support(cue, feat)
                if support < 5:
                    continue
                purity = self.feature_purity(cue, feat)
                score = self.cue_score(cue, feat, "start")
                if score <= 0:
                    continue
                value = score * (0.45 + 0.75 * purity) * (1.0 + 0.18 * (n - 1))
                scored.append((value, purity, support, feat))
            scored.sort(reverse=True)
            if scored:
                top = scored[0]
                second = scored[1][0] if len(scored) > 1 else 0.0
                margin = top[0] / max(second, 1e-9)
                candidates.append((top[0], top[1], margin, n, top[2], top[3]))
        candidates.sort(reverse=True)
        for value, purity, margin, n, support, feat in candidates:
            # A command construction normally has a strong, pure clause-start
            # cue. Assertions often lack one; in that case ASSERT is safer than
            # summing many weak correlations and hallucinating GOAL.
            if purity >= 0.68 and support >= 6 and margin >= 1.10 and value >= 0.10:
                return feat.split(":", 1)[1]

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
