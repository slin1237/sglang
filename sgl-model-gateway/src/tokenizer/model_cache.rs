//! Model-aware tokenizer cache.
//!
//! This module provides a cache that stores tokenizers by their path and
//! can look up the appropriate tokenizer for a given model ID using the
//! worker registry.

use std::sync::Arc;

use anyhow::{anyhow, Result};
use dashmap::DashMap;
use tracing::debug;

use super::{
    cache::{CacheConfig, CachedTokenizer},
    factory::create_tokenizer_async_with_chat_template,
    traits::Tokenizer,
};
use crate::core::WorkerRegistry;

/// A cache for tokenizers that can look up tokenizers by model ID.
///
/// This cache stores tokenizers by their path to avoid repeated loading,
/// and uses the worker registry to find the tokenizer path for a given model.
pub struct TokenizerCache {
    /// Cache of tokenizers keyed by tokenizer path
    cache: DashMap<String, Arc<dyn Tokenizer>>,
    /// Worker registry for looking up tokenizer paths
    worker_registry: Arc<WorkerRegistry>,
    /// Cache configuration for new tokenizers
    cache_config: CacheConfig,
}

impl TokenizerCache {
    /// Create a new tokenizer cache.
    pub fn new(worker_registry: Arc<WorkerRegistry>) -> Self {
        Self {
            cache: DashMap::new(),
            worker_registry,
            cache_config: CacheConfig::default(),
        }
    }

    /// Create a new tokenizer cache with custom cache configuration.
    pub fn with_cache_config(
        worker_registry: Arc<WorkerRegistry>,
        cache_config: CacheConfig,
    ) -> Self {
        Self {
            cache: DashMap::new(),
            worker_registry,
            cache_config,
        }
    }

    /// Get the tokenizer path for a model by looking up workers.
    fn get_tokenizer_path_for_model(&self, model_id: &str) -> Option<String> {
        // Search through all workers to find one that supports this model
        // and has a tokenizer path configured
        for worker in self.worker_registry.get_all() {
            if worker.supports_model(model_id) {
                if let Some(path) = worker.tokenizer_path(model_id) {
                    return Some(path.to_string());
                }
            }
        }

        // If no specific tokenizer path found, try using the model ID itself
        // This works for HuggingFace model IDs that can be downloaded
        debug!(
            "No tokenizer path found for model '{}', trying model ID as path",
            model_id
        );
        Some(model_id.to_string())
    }

    /// Get or create a tokenizer for the given path.
    async fn get_or_create_tokenizer(&self, tokenizer_path: &str) -> Result<Arc<dyn Tokenizer>> {
        // Check cache first
        if let Some(tokenizer) = self.cache.get(tokenizer_path) {
            return Ok(Arc::clone(tokenizer.value()));
        }

        // Create new tokenizer
        debug!("Creating tokenizer for path: {}", tokenizer_path);
        let base_tokenizer = create_tokenizer_async_with_chat_template(tokenizer_path, None)
            .await
            .map_err(|e| anyhow!("Failed to create tokenizer for '{}': {}", tokenizer_path, e))?;

        // Optionally wrap with caching layer
        let tokenizer: Arc<dyn Tokenizer> =
            if self.cache_config.enable_l0 || self.cache_config.enable_l1 {
                Arc::new(CachedTokenizer::new(
                    base_tokenizer,
                    self.cache_config.clone(),
                ))
            } else {
                base_tokenizer
            };

        // Cache and return
        self.cache
            .insert(tokenizer_path.to_string(), Arc::clone(&tokenizer));
        Ok(tokenizer)
    }

    /// Get a tokenizer for the specified model.
    pub async fn get_tokenizer_for_model(&self, model_id: &str) -> Result<Arc<dyn Tokenizer>> {
        let tokenizer_path = self
            .get_tokenizer_path_for_model(model_id)
            .ok_or_else(|| anyhow!("No tokenizer found for model '{}'", model_id))?;

        self.get_or_create_tokenizer(&tokenizer_path).await
    }

    /// Encode text to token IDs for a given model.
    pub async fn encode(&self, model_id: &str, text: &str) -> Result<Vec<u32>> {
        let tokenizer = self.get_tokenizer_for_model(model_id).await?;
        let encoding = tokenizer.encode(text)?;
        Ok(encoding.token_ids().to_vec())
    }

    /// Encode multiple texts to token IDs for a given model.
    pub async fn encode_batch(&self, model_id: &str, texts: &[&str]) -> Result<Vec<Vec<u32>>> {
        let tokenizer = self.get_tokenizer_for_model(model_id).await?;
        let encodings = tokenizer.encode_batch(texts)?;
        Ok(encodings.iter().map(|e| e.token_ids().to_vec()).collect())
    }

    /// Decode token IDs to text for a given model.
    pub async fn decode(
        &self,
        model_id: &str,
        token_ids: &[u32],
        skip_special_tokens: bool,
    ) -> Result<String> {
        let tokenizer = self.get_tokenizer_for_model(model_id).await?;
        tokenizer.decode(token_ids, skip_special_tokens)
    }

    /// Clear the tokenizer cache.
    pub fn clear(&self) {
        self.cache.clear();
        debug!("Tokenizer cache cleared");
    }

    /// Get the number of cached tokenizers.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Check if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }
}

impl std::fmt::Debug for TokenizerCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TokenizerCache")
            .field("cache_size", &self.cache.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_cache_creation() {
        let registry = Arc::new(WorkerRegistry::new());
        let cache = TokenizerCache::new(registry);
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }
}
