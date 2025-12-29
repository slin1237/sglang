"""
Allow running the CLI via: python -m sglang_router

DEPRECATED: Please use python -m sgl_model_gateway instead.
"""

import warnings

warnings.warn(
    "Running 'python -m sglang_router' is deprecated. "
    "Please use 'python -m sgl_model_gateway' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from sgl_model_gateway.cli import main

if __name__ == "__main__":
    main()
