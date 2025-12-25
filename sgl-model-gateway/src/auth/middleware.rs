//! Control plane authentication middleware.
//!
//! Provides middleware for authenticating and authorizing access to control plane APIs.
//! Supports both JWT/OIDC tokens and API keys.

use axum::{
    body::Body,
    extract::{Request, State},
    http::{header, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
};
use std::sync::Arc;
use tracing::{debug, warn};

use super::audit::AuditLogger;
use super::config::{ControlPlaneAuthConfig, Role};
use super::jwt::JwtValidator;
use crate::middleware::RequestId;

/// Authenticated principal information.
#[derive(Debug, Clone)]
pub struct Principal {
    /// Subject identifier (user ID, email, or API key ID)
    pub id: String,

    /// Display name if available
    pub name: Option<String>,

    /// Authentication method used
    pub auth_method: AuthMethod,

    /// Assigned role
    pub role: Role,
}

/// Authentication method used to authenticate the principal.
#[derive(Debug, Clone)]
pub enum AuthMethod {
    /// JWT/OIDC token from external IDP
    Jwt { issuer: String },
    /// API key for service accounts
    ApiKey { key_id: String },
}

impl std::fmt::Display for AuthMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AuthMethod::Jwt { issuer } => write!(f, "jwt:{}", issuer),
            AuthMethod::ApiKey { key_id } => write!(f, "api_key:{}", key_id),
        }
    }
}

/// Extension trait for extracting Principal from request extensions.
pub trait PrincipalExt {
    fn principal(&self) -> Option<&Principal>;
}

impl<B> PrincipalExt for http::Request<B> {
    fn principal(&self) -> Option<&Principal> {
        self.extensions().get::<Principal>()
    }
}

/// State for the control plane authentication middleware.
#[derive(Clone)]
pub struct ControlPlaneAuthState {
    /// Authentication configuration
    pub config: ControlPlaneAuthConfig,

    /// JWT validator (if JWT auth is configured)
    pub jwt_validator: Option<Arc<JwtValidator>>,

    /// Audit logger
    pub audit_logger: AuditLogger,
}

impl ControlPlaneAuthState {
    /// Create a new control plane auth state.
    pub fn new(config: ControlPlaneAuthConfig, jwt_validator: Option<Arc<JwtValidator>>) -> Self {
        let audit_logger = AuditLogger::new(config.audit_enabled);
        Self {
            config,
            jwt_validator,
            audit_logger,
        }
    }

    /// Create from config, initializing JWT validator if needed.
    pub async fn from_config(
        config: ControlPlaneAuthConfig,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let jwt_validator = if let Some(jwt_config) = &config.jwt {
            Some(Arc::new(JwtValidator::from_config(jwt_config.clone()).await?))
        } else {
            None
        };

        Ok(Self::new(config, jwt_validator))
    }

    /// Check if authentication is required.
    pub fn is_auth_required(&self) -> bool {
        self.config.is_enabled()
    }
}

/// Control plane authentication middleware.
///
/// This middleware:
/// 1. Extracts the Bearer token from the Authorization header
/// 2. Attempts JWT validation first (if configured)
/// 3. Falls back to API key validation (if configured)
/// 4. Checks if the authenticated principal has admin role
/// 5. Logs audit events for control plane access
///
/// Returns 401 Unauthorized if authentication fails.
/// Returns 403 Forbidden if the user doesn't have admin role.
pub async fn control_plane_auth_middleware(
    State(auth_state): State<ControlPlaneAuthState>,
    mut request: Request<Body>,
    next: Next,
) -> Response {
    // If no authentication is configured, allow through (backward compatibility)
    if !auth_state.is_auth_required() {
        return next.run(request).await;
    }

    let method = request.method().to_string();
    let path = request.uri().path().to_string();
    let request_id = request
        .extensions()
        .get::<RequestId>()
        .map(|r| r.0.clone());

    // Extract Authorization header
    let auth_header = request
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|h| h.to_str().ok());

    let token = match auth_header {
        Some(header_value) if header_value.starts_with("Bearer ") => {
            &header_value[7..] // Skip "Bearer "
        }
        _ => {
            debug!("Missing or invalid Authorization header for control plane API");
            auth_state.audit_logger.log_auth_failure(
                &method,
                &path,
                "Missing or invalid Authorization header",
                request_id.as_deref(),
            );
            return (
                StatusCode::UNAUTHORIZED,
                [("WWW-Authenticate", "Bearer realm=\"control-plane\"")],
                "Missing or invalid Authorization header",
            )
                .into_response();
        }
    };

    // Try JWT validation first
    if let Some(jwt_validator) = &auth_state.jwt_validator {
        match jwt_validator.validate(token).await {
            Ok(validated_token) => {
                // Check admin role
                if !validated_token.role.is_admin() {
                    warn!(
                        "User {} has role {:?} but admin is required for control plane access",
                        validated_token.subject, validated_token.role
                    );
                    auth_state.audit_logger.log_denied(
                        &validated_token.subject,
                        "jwt",
                        validated_token.role,
                        &method,
                        &path,
                        "Admin role required for control plane access",
                        request_id.as_deref(),
                    );
                    return (
                        StatusCode::FORBIDDEN,
                        "Admin role required for control plane access",
                    )
                        .into_response();
                }

                // Create principal
                let principal = Principal {
                    id: validated_token.subject.clone(),
                    name: validated_token.name.clone(),
                    auth_method: AuthMethod::Jwt {
                        issuer: validated_token.issuer.clone(),
                    },
                    role: validated_token.role,
                };

                debug!(
                    "JWT authentication successful for {} with role {:?}",
                    principal.id, principal.role
                );

                // Add principal to request extensions
                request.extensions_mut().insert(principal.clone());

                // Log successful authentication
                auth_state.audit_logger.log_success(
                    &principal.id,
                    "jwt",
                    principal.role,
                    &method,
                    &path,
                    None,
                    request_id.as_deref(),
                );

                return next.run(request).await;
            }
            Err(e) => {
                debug!("JWT validation failed: {}, trying API key", e);
                // Continue to API key validation
            }
        }
    }

    // Try API key validation
    if let Some(api_key_entry) = auth_state.config.find_api_key(token) {
        // Check admin role
        if !api_key_entry.role.is_admin() {
            warn!(
                "API key {} has role {:?} but admin is required for control plane access",
                api_key_entry.id, api_key_entry.role
            );
            auth_state.audit_logger.log_denied(
                &api_key_entry.id,
                "api_key",
                api_key_entry.role,
                &method,
                &path,
                "Admin role required for control plane access",
                request_id.as_deref(),
            );
            return (
                StatusCode::FORBIDDEN,
                "Admin role required for control plane access",
            )
                .into_response();
        }

        // Create principal
        let principal = Principal {
            id: api_key_entry.id.clone(),
            name: Some(api_key_entry.name.clone()),
            auth_method: AuthMethod::ApiKey {
                key_id: api_key_entry.id.clone(),
            },
            role: api_key_entry.role,
        };

        debug!(
            "API key authentication successful for {} with role {:?}",
            principal.id, principal.role
        );

        // Add principal to request extensions
        request.extensions_mut().insert(principal.clone());

        // Log successful authentication
        auth_state.audit_logger.log_success(
            &principal.id,
            "api_key",
            principal.role,
            &method,
            &path,
            None,
            request_id.as_deref(),
        );

        return next.run(request).await;
    }

    // Authentication failed
    debug!("Control plane authentication failed: invalid token");
    auth_state.audit_logger.log_auth_failure(
        &method,
        &path,
        "Invalid token",
        request_id.as_deref(),
    );

    (
        StatusCode::UNAUTHORIZED,
        [("WWW-Authenticate", "Bearer realm=\"control-plane\"")],
        "Invalid authentication token",
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auth_method_display() {
        let jwt = AuthMethod::Jwt {
            issuer: "https://example.com".to_string(),
        };
        assert_eq!(jwt.to_string(), "jwt:https://example.com");

        let api_key = AuthMethod::ApiKey {
            key_id: "key-123".to_string(),
        };
        assert_eq!(api_key.to_string(), "api_key:key-123");
    }

    #[test]
    fn test_control_plane_auth_state_no_config() {
        let config = ControlPlaneAuthConfig::default();
        let state = ControlPlaneAuthState::new(config, None);
        assert!(!state.is_auth_required());
    }

    #[test]
    fn test_control_plane_auth_state_with_api_keys() {
        use super::super::config::ApiKeyEntry;

        let config = ControlPlaneAuthConfig {
            jwt: None,
            api_keys: vec![ApiKeyEntry::new("test", "Test Key", "secret", Role::Admin)],
            audit_enabled: true,
        };
        let state = ControlPlaneAuthState::new(config, None);
        assert!(state.is_auth_required());
    }
}
