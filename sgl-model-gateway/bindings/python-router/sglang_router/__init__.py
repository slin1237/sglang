"""
sglang_router - DEPRECATED

This package has been renamed to sgl-model-gateway.
Please update your dependencies and imports:

    pip uninstall sglang-router
    pip install sgl-model-gateway

    # Update imports from:
    from sglang_router import Router, RouterArgs

    # To:
    from sgl_model_gateway import Router, RouterArgs

This compatibility package will be removed in a future release.
"""

import warnings

warnings.warn(
    "The 'sglang-router' package has been renamed to 'sgl-model-gateway'. "
    "Please update your dependencies: pip install sgl-model-gateway. "
    "This compatibility package will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from sgl_model_gateway
from sgl_model_gateway import __version__
from sgl_model_gateway.router import (
    Router,
    PolicyType,
    BackendType,
    HistoryBackendType,
    PyApiKeyEntry,
    PyControlPlaneAuthConfig,
    PyJwtConfig,
    PyOracleConfig,
    PyPostgresConfig,
    PyRole,
    policy_from_str,
    backend_from_str,
    history_backend_from_str,
    role_from_str,
    build_control_plane_auth_config,
)
from sgl_model_gateway.router_args import RouterArgs

# Re-export the Rust bindings module for compatibility
from sgl_model_gateway import sgl_model_gateway_rs as sglang_router_rs

__all__ = [
    "__version__",
    # Router classes and enums
    "Router",
    "RouterArgs",
    "PolicyType",
    "BackendType",
    "HistoryBackendType",
    "PyApiKeyEntry",
    "PyControlPlaneAuthConfig",
    "PyJwtConfig",
    "PyOracleConfig",
    "PyPostgresConfig",
    "PyRole",
    # Utility functions
    "policy_from_str",
    "backend_from_str",
    "history_backend_from_str",
    "role_from_str",
    "build_control_plane_auth_config",
    # Rust bindings (aliased for backward compatibility)
    "sglang_router_rs",
]
