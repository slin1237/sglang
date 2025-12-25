//! JWT validation for control plane authentication.
//!
//! Supports:
//! - RS256, RS384, RS512 (RSA)
//! - ES256, ES384 (ECDSA)
//! - Audience and issuer validation
//! - Role extraction from claims

use jsonwebtoken::{
    decode, decode_header,
    jwk::{AlgorithmParameters, Jwk},
    Algorithm, DecodingKey, TokenData, Validation,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tracing::{debug, error, warn};

use super::config::{JwtConfig, Role};
use super::jwks::{JwksError, JwksProvider};

/// Error types for JWT validation.
#[derive(Debug, thiserror::Error)]
pub enum JwtValidatorError {
    #[error("Invalid token format")]
    InvalidFormat,

    #[error("Token header missing 'kid' claim")]
    MissingKid,

    #[error("Failed to get signing key: {0}")]
    KeyError(#[from] JwksError),

    #[error("Unsupported algorithm: {0:?}")]
    UnsupportedAlgorithm(Algorithm),

    #[error("Token validation failed: {0}")]
    ValidationFailed(String),

    #[error("Failed to decode token: {0}")]
    DecodeFailed(#[from] jsonwebtoken::errors::Error),

    #[error("Failed to extract role from claims")]
    RoleExtractionFailed,

    #[error("No role mapping found for: {0}")]
    NoRoleMapping(String),
}

/// Standard JWT claims we extract.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StandardClaims {
    /// Subject (user ID)
    pub sub: Option<String>,

    /// Issuer
    pub iss: Option<String>,

    /// Audience (can be string or array)
    #[serde(default)]
    pub aud: Audience,

    /// Expiration time
    pub exp: Option<u64>,

    /// Issued at
    pub iat: Option<u64>,

    /// Not before
    pub nbf: Option<u64>,

    /// JWT ID
    pub jti: Option<String>,

    /// Email (common claim)
    pub email: Option<String>,

    /// Name (common claim)
    pub name: Option<String>,

    /// Preferred username (OIDC claim)
    pub preferred_username: Option<String>,

    /// All other claims
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

/// Audience claim can be a single string or an array.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Audience {
    Single(String),
    Multiple(Vec<String>),
    #[default]
    None,
}

impl Audience {
    pub fn contains(&self, aud: &str) -> bool {
        match self {
            Audience::Single(s) => s == aud,
            Audience::Multiple(v) => v.iter().any(|s| s == aud),
            Audience::None => false,
        }
    }
}

/// Validated token with extracted claims and role.
#[derive(Debug, Clone)]
pub struct ValidatedToken {
    /// Subject (user ID)
    pub subject: String,

    /// Issuer
    pub issuer: String,

    /// Assigned role
    pub role: Role,

    /// Email if present
    pub email: Option<String>,

    /// Display name if present
    pub name: Option<String>,

    /// Full claims for additional processing
    pub claims: StandardClaims,
}

/// JWT validator with JWKS integration.
pub struct JwtValidator {
    /// JWKS provider for key fetching
    jwks_provider: Arc<JwksProvider>,

    /// JWT configuration
    config: JwtConfig,

    /// Pre-configured validation settings
    validation: Validation,
}

impl JwtValidator {
    /// Create a new JWT validator with explicit JWKS URI.
    pub fn new(config: JwtConfig, jwks_provider: Arc<JwksProvider>) -> Self {
        let mut validation = Validation::default();

        // Set audience
        validation.set_audience(&[&config.audience]);

        // Set issuer
        validation.set_issuer(&[&config.issuer]);

        // Set leeway for clock skew
        validation.leeway = config.leeway_secs;

        // We'll set the algorithm per-token based on the key
        validation.algorithms = vec![
            Algorithm::RS256,
            Algorithm::RS384,
            Algorithm::RS512,
            Algorithm::ES256,
            Algorithm::ES384,
        ];

        Self {
            jwks_provider,
            config,
            validation,
        }
    }

    /// Create a new JWT validator using OIDC discovery.
    pub async fn from_config(config: JwtConfig) -> Result<Self, JwtValidatorError> {
        let ttl = Duration::from_secs(config.jwks_cache_ttl_secs);

        let jwks_provider = if let Some(jwks_uri) = &config.jwks_uri {
            Arc::new(JwksProvider::new(jwks_uri.clone(), ttl))
        } else {
            Arc::new(JwksProvider::from_issuer(&config.issuer, ttl).await?)
        };

        Ok(Self::new(config, jwks_provider))
    }

    /// Validate a JWT token and extract claims.
    pub async fn validate(&self, token: &str) -> Result<ValidatedToken, JwtValidatorError> {
        // Decode header to get kid
        let header = decode_header(token)?;
        let kid = header.kid.ok_or(JwtValidatorError::MissingKid)?;

        debug!("Validating JWT with kid: {}", kid);

        // Get the signing key
        let jwk = self.jwks_provider.get_key(&kid).await?;

        // Create decoding key from JWK
        let decoding_key = Self::jwk_to_decoding_key(&jwk)?;

        // Determine algorithm from JWK
        let algorithm = Self::jwk_to_algorithm(&jwk)?;

        // Create validation with specific algorithm
        let mut validation = self.validation.clone();
        validation.algorithms = vec![algorithm];

        // Decode and validate token
        let token_data: TokenData<StandardClaims> =
            decode(token, &decoding_key, &validation)?;

        let claims = token_data.claims;

        // Extract subject
        let subject = claims
            .sub
            .clone()
            .or_else(|| claims.email.clone())
            .or_else(|| claims.preferred_username.clone())
            .unwrap_or_else(|| "unknown".to_string());

        // Extract issuer
        let issuer = claims
            .iss
            .clone()
            .unwrap_or_else(|| self.config.issuer.clone());

        // Extract role
        let role = self.extract_role(&claims)?;

        debug!(
            "JWT validated: subject={}, issuer={}, role={:?}",
            subject, issuer, role
        );

        Ok(ValidatedToken {
            subject,
            issuer,
            role,
            email: claims.email.clone(),
            name: claims.name.clone(),
            claims,
        })
    }

    /// Extract role from claims using configured role claim and mapping.
    fn extract_role(&self, claims: &StandardClaims) -> Result<Role, JwtValidatorError> {
        // Try to get the role claim value
        let role_value = claims.extra.get(&self.config.role_claim);

        let role_strings: Vec<String> = match role_value {
            Some(serde_json::Value::String(s)) => vec![s.clone()],
            Some(serde_json::Value::Array(arr)) => arr
                .iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect(),
            None => {
                // Try alternate claim names
                let alternates = ["role", "roles", "groups", "group"];
                let mut found = Vec::new();
                for alt in alternates {
                    if let Some(v) = claims.extra.get(alt) {
                        match v {
                            serde_json::Value::String(s) => found.push(s.clone()),
                            serde_json::Value::Array(arr) => {
                                found.extend(arr.iter().filter_map(|v| v.as_str().map(String::from)))
                            }
                            _ => {}
                        }
                    }
                }
                found
            }
            _ => Vec::new(),
        };

        // If no role mapping configured, check for direct "admin" or "user" values
        if self.config.role_mapping.is_empty() {
            for role_str in &role_strings {
                if let Ok(role) = role_str.parse::<Role>() {
                    return Ok(role);
                }
            }
            // Default to User if no explicit role found
            warn!("No role found in JWT claims, defaulting to User");
            return Ok(Role::User);
        }

        // Use role mapping
        for role_str in &role_strings {
            if let Some(role) = self.config.role_mapping.get(role_str) {
                return Ok(*role);
            }
        }

        // Check if any mapped role is admin - if so, we need explicit mapping
        // Otherwise default to User for safety
        warn!(
            "No matching role mapping found for {:?}, defaulting to User",
            role_strings
        );
        Ok(Role::User)
    }

    /// Convert a JWK to a DecodingKey.
    fn jwk_to_decoding_key(jwk: &Jwk) -> Result<DecodingKey, JwtValidatorError> {
        match &jwk.algorithm {
            AlgorithmParameters::RSA(rsa) => {
                Ok(DecodingKey::from_rsa_components(&rsa.n, &rsa.e)?)
            }
            AlgorithmParameters::EllipticCurve(ec) => {
                Ok(DecodingKey::from_ec_components(&ec.x, &ec.y)?)
            }
            AlgorithmParameters::OctetKey(_) => {
                Err(JwtValidatorError::UnsupportedAlgorithm(Algorithm::HS256))
            }
            AlgorithmParameters::OctetKeyPair(_) => {
                Err(JwtValidatorError::UnsupportedAlgorithm(Algorithm::EdDSA))
            }
        }
    }

    /// Determine the algorithm from a JWK.
    fn jwk_to_algorithm(jwk: &Jwk) -> Result<Algorithm, JwtValidatorError> {
        // First check if algorithm is explicitly specified
        if let Some(alg) = &jwk.common.key_algorithm {
            return Ok(match alg {
                jsonwebtoken::jwk::KeyAlgorithm::RS256 => Algorithm::RS256,
                jsonwebtoken::jwk::KeyAlgorithm::RS384 => Algorithm::RS384,
                jsonwebtoken::jwk::KeyAlgorithm::RS512 => Algorithm::RS512,
                jsonwebtoken::jwk::KeyAlgorithm::ES256 => Algorithm::ES256,
                jsonwebtoken::jwk::KeyAlgorithm::ES384 => Algorithm::ES384,
                other => {
                    return Err(JwtValidatorError::ValidationFailed(format!(
                        "Unsupported key algorithm: {:?}",
                        other
                    )))
                }
            });
        }

        // Infer from key type
        match &jwk.algorithm {
            AlgorithmParameters::RSA(_) => Ok(Algorithm::RS256), // Default RSA to RS256
            AlgorithmParameters::EllipticCurve(ec) => {
                use jsonwebtoken::jwk::EllipticCurve;
                match ec.curve {
                    EllipticCurve::P256 => Ok(Algorithm::ES256),
                    EllipticCurve::P384 => Ok(Algorithm::ES384),
                    // Other curves not supported for ECDSA
                    _ => Err(JwtValidatorError::ValidationFailed(format!(
                        "Unsupported EC curve: {:?}",
                        ec.curve
                    ))),
                }
            }
            _ => Err(JwtValidatorError::ValidationFailed(
                "Cannot determine algorithm from key".to_string(),
            )),
        }
    }

    /// Get a reference to the JWKS provider.
    pub fn jwks_provider(&self) -> &Arc<JwksProvider> {
        &self.jwks_provider
    }

    /// Get a reference to the JWT config.
    pub fn config(&self) -> &JwtConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audience_contains() {
        let single = Audience::Single("api://test".to_string());
        assert!(single.contains("api://test"));
        assert!(!single.contains("other"));

        let multiple = Audience::Multiple(vec!["api://test".to_string(), "api://other".to_string()]);
        assert!(multiple.contains("api://test"));
        assert!(multiple.contains("api://other"));
        assert!(!multiple.contains("unknown"));

        let none = Audience::None;
        assert!(!none.contains("anything"));
    }
}
