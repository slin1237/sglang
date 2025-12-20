//! Tokenizer Service for the /tokenize and /detokenize endpoints.
//!
//! This module provides a service that caches tokenizers and handles
//! tokenization/detokenization requests. It looks up tokenizer paths
//! from the worker registry based on model IDs.

use std::sync::Arc;

use anyhow::{anyhow, Result};
use dashmap::DashMap;
use tracing::debug;

use super::{
    cache::{CacheConfig, CachedTokenizer},
    factory::create_tokenizer_async_with_chat_template,
    traits::Tokenizer,
};
use crate::{
    core::WorkerRegistry,
    protocols::tokenize::{
        CountOutput, DetokenizeRequest, DetokenizeResponse, StringOrArray, TextOutput,
        TokenizeRequest, TokenizeResponse, TokensInput, TokensOutput,
    },
};

/// Default max model length when not available from tokenizer
const DEFAULT_MAX_MODEL_LEN: i64 = -1;

/// Service for handling tokenize and detokenize requests.
///
/// Caches tokenizers by their path to avoid repeated loading.
pub struct TokenizerService {
    /// Cache of tokenizers keyed by tokenizer path
    tokenizer_cache: DashMap<String, Arc<dyn Tokenizer>>,
    /// Worker registry for looking up tokenizer paths
    worker_registry: Arc<WorkerRegistry>,
    /// Cache configuration for new tokenizers
    cache_config: CacheConfig,
}

impl TokenizerService {
    /// Create a new tokenizer service.
    pub fn new(worker_registry: Arc<WorkerRegistry>) -> Self {
        Self {
            tokenizer_cache: DashMap::new(),
            worker_registry,
            cache_config: CacheConfig::default(),
        }
    }

    /// Create a new tokenizer service with custom cache configuration.
    pub fn with_cache_config(worker_registry: Arc<WorkerRegistry>, cache_config: CacheConfig) -> Self {
        Self {
            tokenizer_cache: DashMap::new(),
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
        if let Some(tokenizer) = self.tokenizer_cache.get(tokenizer_path) {
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
                Arc::new(CachedTokenizer::new(base_tokenizer, self.cache_config.clone()))
            } else {
                base_tokenizer
            };

        // Cache and return
        self.tokenizer_cache
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

    /// Handle a tokenize request.
    pub async fn tokenize(&self, request: &TokenizeRequest) -> Result<TokenizeResponse> {
        let tokenizer = self.get_tokenizer_for_model(&request.model).await?;

        // Get max model length if available (not directly accessible from trait, use default)
        let max_model_len = DEFAULT_MAX_MODEL_LEN;

        match &request.prompt {
            StringOrArray::String(text) => {
                let encoding = tokenizer.encode(text)?;
                let token_ids: Vec<u32> = encoding.token_ids().to_vec();
                let count = token_ids.len();

                Ok(TokenizeResponse {
                    tokens: TokensOutput::Single(token_ids),
                    count: CountOutput::Single(count),
                    max_model_len,
                })
            }
            StringOrArray::Array(texts) => {
                let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
                let encodings = tokenizer.encode_batch(&refs)?;

                let token_ids_batch: Vec<Vec<u32>> = encodings
                    .iter()
                    .map(|e| e.token_ids().to_vec())
                    .collect();
                let counts: Vec<usize> = token_ids_batch.iter().map(|ids| ids.len()).collect();

                Ok(TokenizeResponse {
                    tokens: TokensOutput::Batch(token_ids_batch),
                    count: CountOutput::Batch(counts),
                    max_model_len,
                })
            }
        }
    }

    /// Handle a detokenize request.
    pub async fn detokenize(&self, request: &DetokenizeRequest) -> Result<DetokenizeResponse> {
        let tokenizer = self.get_tokenizer_for_model(&request.model).await?;

        match &request.tokens {
            TokensInput::Single(token_ids) => {
                let text = tokenizer.decode(token_ids, request.skip_special_tokens)?;
                Ok(DetokenizeResponse {
                    text: TextOutput::Single(text),
                })
            }
            TokensInput::Batch(token_ids_batch) => {
                let mut texts = Vec::with_capacity(token_ids_batch.len());
                for token_ids in token_ids_batch {
                    let text = tokenizer.decode(token_ids, request.skip_special_tokens)?;
                    texts.push(text);
                }
                Ok(DetokenizeResponse {
                    text: TextOutput::Batch(texts),
                })
            }
        }
    }

    /// Clear the tokenizer cache.
    pub fn clear_cache(&self) {
        self.tokenizer_cache.clear();
        debug!("Tokenizer cache cleared");
    }

    /// Get the number of cached tokenizers.
    pub fn cache_size(&self) -> usize {
        self.tokenizer_cache.len()
    }
}

impl std::fmt::Debug for TokenizerService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TokenizerService")
            .field("cache_size", &self.tokenizer_cache.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_service_creation() {
        let registry = Arc::new(WorkerRegistry::new());
        let service = TokenizerService::new(registry);
        assert_eq!(service.cache_size(), 0);
    }
}
