//! Audit logging for control plane operations.
//!
//! Provides structured audit events for security monitoring and compliance.

use chrono::{DateTime, Utc};
use serde::Serialize;
use std::net::IpAddr;
use tracing::{info, span, Level};

use super::config::Role;

/// Outcome of an audited operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum AuditOutcome {
    /// Operation completed successfully
    Success,
    /// Operation failed (application error)
    Failure,
    /// Operation denied (authorization failure)
    Denied,
}

impl std::fmt::Display for AuditOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AuditOutcome::Success => write!(f, "success"),
            AuditOutcome::Failure => write!(f, "failure"),
            AuditOutcome::Denied => write!(f, "denied"),
        }
    }
}

/// Audit event for control plane operations.
#[derive(Debug, Clone, Serialize)]
pub struct AuditEvent {
    /// Timestamp of the event
    pub timestamp: DateTime<Utc>,

    /// Principal who performed the action (subject ID or API key ID)
    pub principal: String,

    /// Authentication method used (jwt, api_key)
    pub auth_method: String,

    /// Role of the principal
    pub role: Role,

    /// HTTP method
    pub method: String,

    /// Request path
    pub path: String,

    /// Resource being accessed (e.g., worker ID, wasm module ID)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resource: Option<String>,

    /// Operation outcome
    pub outcome: AuditOutcome,

    /// Client IP address
    #[serde(skip_serializing_if = "Option::is_none")]
    pub client_ip: Option<IpAddr>,

    /// Request ID for correlation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,

    /// Additional details or error message
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<String>,
}

impl AuditEvent {
    /// Create a new audit event builder.
    pub fn builder() -> AuditEventBuilder {
        AuditEventBuilder::default()
    }
}

/// Builder for audit events.
#[derive(Default)]
pub struct AuditEventBuilder {
    principal: Option<String>,
    auth_method: Option<String>,
    role: Option<Role>,
    method: Option<String>,
    path: Option<String>,
    resource: Option<String>,
    outcome: Option<AuditOutcome>,
    client_ip: Option<IpAddr>,
    request_id: Option<String>,
    details: Option<String>,
}

impl AuditEventBuilder {
    pub fn principal(mut self, principal: impl Into<String>) -> Self {
        self.principal = Some(principal.into());
        self
    }

    pub fn auth_method(mut self, method: impl Into<String>) -> Self {
        self.auth_method = Some(method.into());
        self
    }

    pub fn role(mut self, role: Role) -> Self {
        self.role = Some(role);
        self
    }

    pub fn method(mut self, method: impl Into<String>) -> Self {
        self.method = Some(method.into());
        self
    }

    pub fn path(mut self, path: impl Into<String>) -> Self {
        self.path = Some(path.into());
        self
    }

    pub fn resource(mut self, resource: impl Into<String>) -> Self {
        self.resource = Some(resource.into());
        self
    }

    pub fn outcome(mut self, outcome: AuditOutcome) -> Self {
        self.outcome = Some(outcome);
        self
    }

    pub fn client_ip(mut self, ip: IpAddr) -> Self {
        self.client_ip = Some(ip);
        self
    }

    pub fn request_id(mut self, id: impl Into<String>) -> Self {
        self.request_id = Some(id.into());
        self
    }

    pub fn details(mut self, details: impl Into<String>) -> Self {
        self.details = Some(details.into());
        self
    }

    pub fn build(self) -> AuditEvent {
        AuditEvent {
            timestamp: Utc::now(),
            principal: self.principal.unwrap_or_else(|| "unknown".to_string()),
            auth_method: self.auth_method.unwrap_or_else(|| "unknown".to_string()),
            role: self.role.unwrap_or_default(),
            method: self.method.unwrap_or_else(|| "UNKNOWN".to_string()),
            path: self.path.unwrap_or_else(|| "/".to_string()),
            resource: self.resource,
            outcome: self.outcome.unwrap_or(AuditOutcome::Success),
            client_ip: self.client_ip,
            request_id: self.request_id,
            details: self.details,
        }
    }
}

/// Audit logger that emits structured audit events.
#[derive(Clone, Default)]
pub struct AuditLogger {
    enabled: bool,
}

impl AuditLogger {
    /// Create a new audit logger.
    pub fn new(enabled: bool) -> Self {
        Self { enabled }
    }

    /// Check if audit logging is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Log an audit event.
    pub fn log(&self, event: &AuditEvent) {
        if !self.enabled {
            return;
        }

        // Create a span for structured logging
        let _span = span!(
            Level::INFO,
            "audit",
            principal = %event.principal,
            auth_method = %event.auth_method,
            role = %event.role,
            method = %event.method,
            path = %event.path,
            outcome = %event.outcome,
        )
        .entered();

        // Log the event
        info!(
            target: "sgl_model_gateway::audit",
            timestamp = %event.timestamp.to_rfc3339(),
            principal = %event.principal,
            auth_method = %event.auth_method,
            role = %event.role,
            method = %event.method,
            path = %event.path,
            resource = ?event.resource,
            outcome = %event.outcome,
            client_ip = ?event.client_ip,
            request_id = ?event.request_id,
            details = ?event.details,
            "control_plane_audit"
        );
    }

    /// Log a successful operation.
    pub fn log_success(
        &self,
        principal: &str,
        auth_method: &str,
        role: Role,
        method: &str,
        path: &str,
        resource: Option<&str>,
        request_id: Option<&str>,
    ) {
        let mut builder = AuditEvent::builder()
            .principal(principal)
            .auth_method(auth_method)
            .role(role)
            .method(method)
            .path(path)
            .outcome(AuditOutcome::Success);

        if let Some(r) = resource {
            builder = builder.resource(r);
        }
        if let Some(id) = request_id {
            builder = builder.request_id(id);
        }

        self.log(&builder.build());
    }

    /// Log a denied operation (authorization failure).
    pub fn log_denied(
        &self,
        principal: &str,
        auth_method: &str,
        role: Role,
        method: &str,
        path: &str,
        reason: &str,
        request_id: Option<&str>,
    ) {
        let mut builder = AuditEvent::builder()
            .principal(principal)
            .auth_method(auth_method)
            .role(role)
            .method(method)
            .path(path)
            .outcome(AuditOutcome::Denied)
            .details(reason);

        if let Some(id) = request_id {
            builder = builder.request_id(id);
        }

        self.log(&builder.build());
    }

    /// Log an authentication failure (before principal is known).
    pub fn log_auth_failure(&self, method: &str, path: &str, reason: &str, request_id: Option<&str>) {
        let mut builder = AuditEvent::builder()
            .principal("unauthenticated")
            .auth_method("none")
            .role(Role::User)
            .method(method)
            .path(path)
            .outcome(AuditOutcome::Denied)
            .details(reason);

        if let Some(id) = request_id {
            builder = builder.request_id(id);
        }

        self.log(&builder.build());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audit_event_builder() {
        let event = AuditEvent::builder()
            .principal("user@example.com")
            .auth_method("jwt")
            .role(Role::Admin)
            .method("POST")
            .path("/workers")
            .resource("worker-123")
            .outcome(AuditOutcome::Success)
            .request_id("req-abc")
            .build();

        assert_eq!(event.principal, "user@example.com");
        assert_eq!(event.auth_method, "jwt");
        assert_eq!(event.role, Role::Admin);
        assert_eq!(event.method, "POST");
        assert_eq!(event.path, "/workers");
        assert_eq!(event.resource, Some("worker-123".to_string()));
        assert_eq!(event.outcome, AuditOutcome::Success);
    }

    #[test]
    fn test_audit_outcome_display() {
        assert_eq!(AuditOutcome::Success.to_string(), "success");
        assert_eq!(AuditOutcome::Failure.to_string(), "failure");
        assert_eq!(AuditOutcome::Denied.to_string(), "denied");
    }
}
