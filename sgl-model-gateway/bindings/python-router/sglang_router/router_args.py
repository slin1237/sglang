"""
sglang_router.router_args - DEPRECATED

This module has been moved to sgl_model_gateway.router_args.
Please update your imports.
"""

import warnings

warnings.warn(
    "sglang_router.router_args has been moved to sgl_model_gateway.router_args. "
    "Please update your imports.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from sgl_model_gateway.router_args
from sgl_model_gateway.router_args import RouterArgs

__all__ = ["RouterArgs"]
