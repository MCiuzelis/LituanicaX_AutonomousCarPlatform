# Stub for OpenUSD's Python bindings.
#
# pxr's submodules fill their namespaces at import time from C++ via
# Tf.PreparePythonModule(), so a type checker sees an empty module and flags
# every single `Usd.PrimRange` / `UsdGeom.Mesh` / `Gf.Vec3d` as unknown.  This
# stub makes any pxr symbol resolve to Any, which is the honest description:
# the real thing is only knowable at runtime.  Editor-only; never imported.
from typing import Any

def __getattr__(name: str) -> Any: ...
