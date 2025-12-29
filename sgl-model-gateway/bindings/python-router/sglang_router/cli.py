"""
sglang_router.cli - DEPRECATED

This module has been moved to sgl_model_gateway.cli.
Please update your imports.
"""

import warnings

warnings.warn(
    "sglang_router.cli has been moved to sgl_model_gateway.cli. "
    "Please update your imports.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from sgl_model_gateway.cli
from sgl_model_gateway.cli import main

__all__ = ["main"]
