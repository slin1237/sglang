//! JWKS (JSON Web Key Set) fetching and caching.
//!
//! Handles:
//! - OIDC discovery to find JWKS endpoint
//! - Fetching and parsing JWKS
//! - Caching with TTL and automatic refresh

use jsonwebtoken::jwk::{Jwk, JwkSet};
use parking_lot::RwLock;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, error, info, warn};

/// Error types for JWKS operations.
#[derive(Debug, thiserror::Error)]
pub enum JwksError {
    #[error("Failed to fetch OIDC discovery document: {0}")]
    DiscoveryFetch(String),

    #[error("Failed to parse OIDC discovery document: {0}")]
    DiscoveryParse(String),

    #[error("JWKS URI not found in discovery document")]
    JwksUriNotFound,

    #[error("Failed to fetch JWKS: {0}")]
    JwksFetch(String),

    #[error("Failed to parse JWKS: {0}")]
    JwksParse(String),

    #[error("Key not found for kid: {0}")]
    KeyNotFound(String),
}

/// OIDC discovery document (subset of fields we need).
#[derive(Debug, serde::Deserialize)]
struct OidcDiscovery {
    jwks_uri: String,
    #[allow(dead_code)]
    issuer: String,
}

/// Cached JWKS with expiration tracking.
struct CachedJwks {
    jwks: JwkSet,
    fetched_at: Instant,
    ttl: Duration,
}

impl CachedJwks {
    fn is_expired(&self) -> bool {
        self.fetched_at.elapsed() > self.ttl
    }
}

/// JWKS provider with caching and automatic refresh.
pub struct JwksProvider {
    /// HTTP client for fetching JWKS
    client: reqwest::Client,
    /// JWKS endpoint URL
    jwks_uri: String,
    /// Cached JWKS
    cache: RwLock<Option<CachedJwks>>,
    /// Cache TTL
    ttl: Duration,
}

impl JwksProvider {
    /// Create a new JWKS provider with explicit JWKS URI.
    pub fn new(jwks_uri: impl Into<String>, ttl: Duration) -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .redirect(reqwest::redirect::Policy::none()) // Prevent SSRF via redirects
            .build()
            .expect("Failed to create HTTP client");

        Self {
            client,
            jwks_uri: jwks_uri.into(),
            cache: RwLock::new(None),
            ttl,
        }
    }

    /// Create a new JWKS provider using OIDC discovery.
    pub async fn from_issuer(issuer: &str, ttl: Duration) -> Result<Self, JwksError> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .expect("Failed to create HTTP client");

        // Normalize issuer URL
        let issuer = issuer.trim_end_matches('/');
        let discovery_url = format!("{}/.well-known/openid-configuration", issuer);

        info!("Fetching OIDC discovery from: {}", discovery_url);

        let response = client
            .get(&discovery_url)
            .send()
            .await
            .map_err(|e| JwksError::DiscoveryFetch(e.to_string()))?;

        if !response.status().is_success() {
            return Err(JwksError::DiscoveryFetch(format!(
                "HTTP {}",
                response.status()
            )));
        }

        let discovery: OidcDiscovery = response
            .json()
            .await
            .map_err(|e| JwksError::DiscoveryParse(e.to_string()))?;

        info!("Discovered JWKS URI: {}", discovery.jwks_uri);

        Ok(Self {
            client,
            jwks_uri: discovery.jwks_uri,
            cache: RwLock::new(None),
            ttl,
        })
    }

    /// Get the JWKS URI.
    pub fn jwks_uri(&self) -> &str {
        &self.jwks_uri
    }

    /// Fetch JWKS from the endpoint.
    async fn fetch_jwks(&self) -> Result<JwkSet, JwksError> {
        debug!("Fetching JWKS from: {}", self.jwks_uri);

        let response = self
            .client
            .get(&self.jwks_uri)
            .send()
            .await
            .map_err(|e| JwksError::JwksFetch(e.to_string()))?;

        if !response.status().is_success() {
            return Err(JwksError::JwksFetch(format!(
                "HTTP {}",
                response.status()
            )));
        }

        let jwks: JwkSet = response
            .json()
            .await
            .map_err(|e| JwksError::JwksParse(e.to_string()))?;

        debug!("Fetched JWKS with {} keys", jwks.keys.len());
        Ok(jwks)
    }

    /// Get the cached JWKS, refreshing if expired or not present.
    pub async fn get_jwks(&self) -> Result<JwkSet, JwksError> {
        // Check cache first
        {
            let cache = self.cache.read();
            if let Some(cached) = cache.as_ref() {
                if !cached.is_expired() {
                    return Ok(cached.jwks.clone());
                }
            }
        }

        // Cache miss or expired, fetch new JWKS
        let jwks = self.fetch_jwks().await?;

        // Update cache
        {
            let mut cache = self.cache.write();
            *cache = Some(CachedJwks {
                jwks: jwks.clone(),
                fetched_at: Instant::now(),
                ttl: self.ttl,
            });
        }

        Ok(jwks)
    }

    /// Get a specific key by kid (key ID).
    pub async fn get_key(&self, kid: &str) -> Result<Jwk, JwksError> {
        let jwks = self.get_jwks().await?;

        // First try to find by kid
        if let Some(key) = jwks.find(kid) {
            return Ok(key.clone());
        }

        // Key not found - try refreshing the cache in case keys were rotated
        warn!("Key {} not found in cached JWKS, refreshing...", kid);
        let jwks = self.fetch_jwks().await?;

        // Update cache
        {
            let mut cache = self.cache.write();
            *cache = Some(CachedJwks {
                jwks: jwks.clone(),
                fetched_at: Instant::now(),
                ttl: self.ttl,
            });
        }

        jwks.find(kid)
            .cloned()
            .ok_or_else(|| JwksError::KeyNotFound(kid.to_string()))
    }

    /// Force refresh the JWKS cache.
    pub async fn refresh(&self) -> Result<(), JwksError> {
        let jwks = self.fetch_jwks().await?;
        let mut cache = self.cache.write();
        *cache = Some(CachedJwks {
            jwks,
            fetched_at: Instant::now(),
            ttl: self.ttl,
        });
        Ok(())
    }
}

/// Thread-safe JWKS provider handle.
pub type JwksProviderHandle = Arc<JwksProvider>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cached_jwks_expiration() {
        let jwks = JwkSet { keys: vec![] };
        let cached = CachedJwks {
            jwks,
            fetched_at: Instant::now() - Duration::from_secs(100),
            ttl: Duration::from_secs(60),
        };
        assert!(cached.is_expired());

        let jwks = JwkSet { keys: vec![] };
        let cached = CachedJwks {
            jwks,
            fetched_at: Instant::now(),
            ttl: Duration::from_secs(60),
        };
        assert!(!cached.is_expired());
    }
}
