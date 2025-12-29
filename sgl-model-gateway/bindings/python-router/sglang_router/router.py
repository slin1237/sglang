"""
sglang_router.router - DEPRECATED

This module has been moved to sgl_model_gateway.router.
Please update your imports.
"""

import warnings

warnings.warn(
    "sglang_router.router has been moved to sgl_model_gateway.router. "
    "Please update your imports.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from sgl_model_gateway.router
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

__all__ = [
    "Router",
    "PolicyType",
    "BackendType",
    "HistoryBackendType",
    "PyApiKeyEntry",
    "PyControlPlaneAuthConfig",
    "PyJwtConfig",
    "PyOracleConfig",
    "PyPostgresConfig",
    "PyRole",
    "policy_from_str",
    "backend_from_str",
    "history_backend_from_str",
    "role_from_str",
    "build_control_plane_auth_config",
]
