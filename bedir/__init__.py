"""BedirAI demo client: run evidence-bound debates and verify every quote yourself."""

from .client import API, DebateError, debate
from .verify import Match, check, locate, sha256

__all__ = ["API", "DebateError", "Match", "check", "debate", "locate", "sha256"]
__version__ = "0.1.0"
