# Stub for the Omniverse Kit `omni` namespace.
#
# Kit injects its extension packages onto sys.path when the app launches, so
# nothing under `omni.*` exists on disk for a type checker to find.  Editor-only.
from typing import Any

def __getattr__(name: str) -> Any: ...
