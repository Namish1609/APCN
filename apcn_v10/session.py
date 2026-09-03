from __future__ import annotations

from pathlib import Path
from typing import Dict
import json

from .definitions import ConceptStore, DefinitionCurriculum
from .language import AdaptiveLanguageSession, GeneratedLanguageTestReport, run_generated_language_test


class CognitiveSessionV10:
    """V0.10 language + definition state.

    Perception remains in APCN V0.8.2 and is wired into the V0.10 desktop UI so
    existing compact visual memory remains usable instead of being reset.
    """

    def __init__(self, seed: int = 10):
        self.seed = seed
        self.language = AdaptiveLanguageSession(seed)
        self.concepts = ConceptStore()
        self.definitions = DefinitionCurriculum(self.concepts)
        self.test_history = []

    def test_language(self, samples: int = 600) -> GeneratedLanguageTestReport:
        report = run_generated_language_test(
            self.language.learner,
            samples=samples,
            seed=self.seed + 10000 + len(self.test_history) * 17,
        )
        self.test_history.append({
            "episodes": self.language.learner.episode_count,
            "exact": report.exact_accuracy,
            "intent": report.intent_accuracy,
            "relation": report.relation_accuracy,
            "operator": report.operator_accuracy,
        })
        return report

    def save(self, output_dir: str | Path = "outputs/v0_10") -> Dict[str, str]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        language_path = self.language.save(out)
        concept_path = out / "concept_store_v0_10.json"
        self.concepts.save(concept_path)
        state_path = out / "session_v0_10.json"
        state_path.write_text(json.dumps({
            "version": "0.10.0",
            "seed": self.seed,
            "test_history": self.test_history,
            "language_episodes": self.language.learner.episode_count,
            "definition_count": self.concepts.definition_count,
        }, indent=2), encoding="utf-8")
        return {
            "language": str(language_path),
            "concepts": str(concept_path),
            "session": str(state_path),
        }
