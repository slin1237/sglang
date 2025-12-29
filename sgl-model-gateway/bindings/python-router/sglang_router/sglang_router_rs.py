"""
sglang_router.sglang_router_rs - DEPRECATED

This module has been moved to sgl_model_gateway.sgl_model_gateway_rs.
Please update your imports.
"""

import warnings

warnings.warn(
    "sglang_router.sglang_router_rs has been moved to sgl_model_gateway.sgl_model_gateway_rs. "
    "Please update your imports.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from sgl_model_gateway.sgl_model_gateway_rs
from sgl_model_gateway.sgl_model_gateway_rs import *  # noqa: F401, F403
from sgl_model_gateway.sgl_model_gateway_rs import (
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
    get_version_string,
    get_verbose_version_string,
    get_available_tool_call_parsers,
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
    "get_version_string",
    "get_verbose_version_string",
    "get_available_tool_call_parsers",
]
