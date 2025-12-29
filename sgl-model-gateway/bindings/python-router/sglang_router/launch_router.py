"""
sglang_router.launch_router - DEPRECATED

This module has been moved to sgl_model_gateway.launch_router.
Please update your imports.
"""

import warnings

warnings.warn(
    "sglang_router.launch_router has been moved to sgl_model_gateway.launch_router. "
    "Please update your imports.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from sgl_model_gateway.launch_router
from sgl_model_gateway.launch_router import (
    launch_router,
    parse_router_args,
    RouterArgs,
)

__all__ = ["launch_router", "parse_router_args", "RouterArgs"]
