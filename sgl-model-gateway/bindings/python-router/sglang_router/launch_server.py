"""
sglang_router.launch_server - DEPRECATED

This module has been moved to sgl_model_gateway.launch_server.
Please update your imports.
"""

import warnings

warnings.warn(
    "sglang_router.launch_server has been moved to sgl_model_gateway.launch_server. "
    "Please update your imports.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from sgl_model_gateway.launch_server
from sgl_model_gateway.launch_server import main

__all__ = ["main"]
