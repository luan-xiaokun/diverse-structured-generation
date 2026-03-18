import warnings

warnings.warn(
    "diverse_guide_rust is deprecated; use diverse_guide instead",
    DeprecationWarning,
    stacklevel=2,
)
from diverse_guide import *  # noqa: F401, F403, E402
