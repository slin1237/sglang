"""
sglang_router.mini_lb - DEPRECATED

This module has been moved to sgl_model_gateway.mini_lb.
Please update your imports.
"""

import warnings

warnings.warn(
    "sglang_router.mini_lb has been moved to sgl_model_gateway.mini_lb. "
    "Please update your imports.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from sgl_model_gateway.mini_lb
from sgl_model_gateway.mini_lb import *  # noqa: F401, F403
