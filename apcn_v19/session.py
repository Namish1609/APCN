from __future__ import annotations

from pathlib import Path
from typing import Dict
import json
import re

from apcn_v15.conversation import ConversationReply
from apcn_v18.session import CognitiveSessionV18

from .language_math import OnlineConstructionArithmeticV19


class CognitiveSessionV19(CognitiveSessionV18):
    VERSION = "0.19.0"

    DEMO = re.compile(
        r"^(?:teach|learn|remember)(?:\s+that)?\s+(.+?)\s+(?:equals|=>|=)\s*(-?\d+(?:\.\d+)?)\s*[.!]*$",
        re.I,
    )
    OPERATOR_ALIAS = re.compile(
        r"^(?:teach|learn|remember)(?:\s+that)?\s+['\"]?(.+?)['\"]?\s+means\s+(addition|add|sum|subtraction|subtract|minus)\s*[.!]*$",
        re.I,
    )

    def __init__(self, seed: int = 19, *, bootstrap_english: bool = True):
        super().__init__(seed)
        self.seed = seed
        self.language_math_v19 = OnlineConstructionArithmeticV19(bootstrap=bootstrap_english)
        self.v19_history = []

    def _record_v19(self, reply: ConversationReply, **meta) -> ConversationReply:
        row = {
            "act": reply.act,
            "confidence": reply.confidence,
            "learned": reply.learned,
            **meta,
        }
        self.v19_history.append(row)
        if len(self.v19_history) > 4096:
            del self.v19_history[: len(self.v19_history) - 4096]
        return reply

    def talk(self, text: str) -> ConversationReply:
        q = str(text).strip()
        m = self.DEMO.match(q)
        if m:
            surface, raw_result = m.group(1), m.group(2)
            row = self.language_math_v19.observe(surface, float(raw_result), source="user", weight=5)
            if row.get("learned"):
                op = str(row.get("operation", "operation")).lower()
                reply = ConversationReply(
                    f"Learned that construction from the example. Its current arithmetic meaning is {op}.",
                    "LEARN_ARITHMETIC_CONSTRUCTION",
                    .99,
                    learned=True,
                    trace=["v019:online_demonstration", f"v019:operation:{op}"],
                )
            else:
                reply = ConversationReply(
                    f"I could not learn a unique arithmetic meaning from that example: {row.get('reason', 'ambiguous demonstration')}.",
                    "CLARIFY",
                    .40,
                    trace=["v019:demonstration_ambiguous"],
                )
            return self._record_v19(reply, route="demonstration")

        m = self.OPERATOR_ALIAS.match(q)
        if m:
            phrase, meaning = m.group(1), m.group(2)
            row = self.language_math_v19.teach_operator_phrase(phrase, meaning, weight=10)
            if row.get("learned"):
                reply = ConversationReply(
                    f"Learned the arithmetic meaning of '{row['phrase']}' as {str(row['operation']).lower()}.",
                    "LEARN_ARITHMETIC_LANGUAGE",
                    .99,
                    learned=True,
                    trace=["v019:operator_language_teaching"],
                )
            else:
                reply = ConversationReply(
                    f"I could not learn that arithmetic wording: {row.get('reason', 'unsupported meaning')}.",
                    "CLARIFY",
                    .35,
                    trace=["v019:operator_language_teaching_failed"],
                )
            return self._record_v19(reply, route="operator_alias")

        numbers, _ = self.language_math_v19.extract_numbers(q)
        if len(numbers) == 2:
            interpretation = self.language_math_v19.infer(q)
            if interpretation.understood:
                reply = ConversationReply(
                    interpretation.text,
                    "ANSWER_ARITHMETIC",
                    interpretation.confidence,
                    trace=[
                        "v019:online_construction_inference",
                        f"v019:operation:{interpretation.operation}",
                        *[f"v019:evidence:{x}" for x in interpretation.evidence[-6:]],
                    ],
                )
                return self._record_v19(reply, route="arithmetic", operation=interpretation.operation)
            # Two numbers alone are not enough evidence to steal arbitrary V0.18 turns.
            mathish = any(
                token in self.language_math_v19.feature_counts
                for token in self.language_math_v19._features(self.language_math_v19.extract_numbers(q)[1])
            )
            if mathish:
                reply = ConversationReply(
                    interpretation.text,
                    "CLARIFY",
                    max(.30, interpretation.confidence),
                    trace=["v019:arithmetic_ambiguous"],
                )
                return self._record_v19(reply, route="arithmetic_ambiguous")

        reply = super().talk(text)
        return self._record_v19(reply, route="v018_fallback")

    def v19_memory_audit(self) -> Dict[str, object]:
        return {
            "version": self.VERSION,
            "language_math": self.language_math_v19.summary(),
            "history_size": len(self.v19_history),
            "architecture_contract": {
                "backpropagation": False,
                "gradient_descent": False,
                "external_llm": False,
                "pretrained_language_model": False,
                "online_demonstration_learning": True,
                "addition_and_subtraction_execution": True,
                "world_model_required_for_arithmetic": False,
                "v018_semantic_fallback_available": True,
            },
        }

    def memory_audit(self) -> Dict[str, object]:
        base = super().memory_audit()
        base["v019_language_math"] = self.v19_memory_audit()
        return base

    def save(self, output_dir: str | Path = "outputs/v0_19") -> Dict[str, str]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        base = out / "base_v18"
        super().save(base)
        language = out / "language_math_v0_19.json"
        state = out / "session_v0_19.json"
        self.language_math_v19.save(language)
        state.write_text(
            json.dumps(
                {
                    "version": self.VERSION,
                    "seed": self.seed,
                    "v19_history": self.v19_history,
                    "memory_audit": self.v19_memory_audit(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return {"base_v18": str(base), "language_math": str(language), "session": str(state)}

    @classmethod
    def load_checkpoint(cls, output_dir: str | Path = "outputs/v0_19", *, seed: int = 19) -> "CognitiveSessionV19":
        out = Path(output_dir)
        base = out / "base_v18"
        if not base.exists():
            raise FileNotFoundError(f"missing V0.19 base checkpoint: {base}")
        old = CognitiveSessionV18.load_checkpoint(base, seed=seed)
        obj = cls(seed, bootstrap_english=False)
        for name, value in old.__dict__.items():
            setattr(obj, name, value)
        language = out / "language_math_v0_19.json"
        obj.language_math_v19 = (
            OnlineConstructionArithmeticV19.load(language)
            if language.exists()
            else OnlineConstructionArithmeticV19(bootstrap=True)
        )
        obj.v19_history = []
        state = out / "session_v0_19.json"
        if state.exists():
            data = json.loads(state.read_text(encoding="utf-8"))
            obj.v19_history = list(data.get("v19_history", []))[-4096:]
        obj.seed = seed
        return obj
