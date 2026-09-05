from __future__ import annotations

from .ui import APCNV14Window


class APCNV14LaunchWindow(APCNV14Window):
    """Initialization-safe V0.14 desktop window.

    The UI inheritance chain constructs temporary older cognitive sessions before
    V0.13/V0.14 replace them. Older constructors call ``self._refresh_header()``,
    which dynamically dispatches to the most-derived implementation. This wrapper
    therefore keeps header refresh tolerant of partially initialized V0.10-V0.14
    state and prevents startup crashes when fields such as ``errors`` or ``world``
    do not exist yet.
    """

    def _refresh_header(self):
        if not hasattr(self, "memory_label"):
            return

        cognitive = getattr(self, "cognitive", None)
        perception = getattr(self, "perception", None)
        learner = getattr(perception, "learner", None)

        visual_episodes = int(getattr(learner, "episode_count", 0) or 0)

        language = getattr(cognitive, "language", None)
        language_learner = getattr(language, "learner", None)
        language_episodes = int(getattr(language_learner, "episode_count", 0) or 0)

        concepts = getattr(cognitive, "concepts", None)
        definition_count = int(getattr(concepts, "definition_count", 0) or 0)

        errors = getattr(cognitive, "errors", None)
        signatures = getattr(errors, "signatures", {}) if errors is not None else {}
        error_count = len(signatures) if signatures is not None else 0

        world = getattr(cognitive, "world", None)
        instance_memory = getattr(world, "instances", None) if world is not None else None
        instance_map = getattr(instance_memory, "instances", {}) if instance_memory is not None else {}
        instance_count = len(instance_map) if instance_map is not None else 0

        self.memory_label.setText(
            f"visual {visual_episodes} • language {language_episodes} • "
            f"definitions {definition_count} • persistent instances {instance_count} • "
            f"errors {error_count}"
        )
