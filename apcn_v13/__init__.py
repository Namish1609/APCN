"""APCN V0.13 — Persistent Real-World Object Memory.

V0.13 extends V0.12 with bounded multi-view instance memory, temporal identity,
explicit occlusion/out-of-view/lost beliefs, trajectory events and human identity
correction. The implementation remains non-neural.
"""

from .identity import InstanceMemory, PersistentInstance, IdentityMatch
from .world import PersistentWorldModel, Detection, TrackBelief, WorldEvent
from .session import CognitiveSessionV13

__all__ = [
    "InstanceMemory", "PersistentInstance", "IdentityMatch",
    "PersistentWorldModel", "Detection", "TrackBelief", "WorldEvent",
    "CognitiveSessionV13",
]

__version__ = "0.13.0"
