//! Tokenizer cache for /v1/tokenize and /v1/detokenize endpoints.
//!
//! Provides a cache of tokenizers indexed by model ID, using the WorkerRegistry
//! to look up tokenizer paths and the tokenizer factory to create tokenizers.

use std::sync::Arc;

use anyhow::{anyhow, Result};
use dashmap::DashMap;
use tracing::debug;

use crate::{
    core::WorkerRegistry,
    tokenizer::{
        cache::{CacheConfig, CachedTokenizer},
        factory::create_tokenizer_with_chat_template_blocking,
        traits::Tokenizer,
    },
};

/// Cache of tokenizers by model ID.
///
/// This cache is used by the tokenize and detokenize endpoints to look up
/// tokenizers by model ID. It uses the WorkerRegistry to find the tokenizer
/// path for a given model, then creates and caches the tokenizer.
#[derive(Clone)]
pub struct TokenizerCache {
    /// Worker registry for looking up model -> tokenizer path
    worker_registry: Arc<WorkerRegistry>,

    /// Cached tokenizers by model ID
    cache: Arc<DashMap<String, Arc<dyn Tokenizer>>>,

    /// Configuration for the tokenizer cache layer
    cache_config: CacheConfig,
}

impl TokenizerCache {
    /// Create a new tokenizer cache with default configuration.
    pub fn new(worker_registry: Arc<WorkerRegistry>) -> Self {
        Self {
            worker_registry,
            cache: Arc::new(DashMap::new()),
            cache_config: CacheConfig::default(),
        }
    }

    /// Create a new tokenizer cache with custom configuration.
    pub fn with_cache_config(worker_registry: Arc<WorkerRegistry>, cache_config: CacheConfig) -> Self {
        Self {
            worker_registry,
            cache: Arc::new(DashMap::new()),
            cache_config,
        }
    }

    /// Get or create a tokenizer for the given model.
    async fn get_tokenizer(&self, model: &str) -> Result<Arc<dyn Tokenizer>> {
        // Check cache first
        if let Some(tokenizer) = self.cache.get(model) {
            return Ok(tokenizer.clone());
        }

        // Look up workers for this model
        let workers = self.worker_registry.get_by_model(model);
        if workers.is_empty() {
            return Err(anyhow!(
                "No workers found for model '{}'. Unable to determine tokenizer path.",
                model
            ));
        }

        // Get tokenizer path from first worker
        let worker = &workers[0];
        let tokenizer_path = worker.tokenizer_path(model).ok_or_else(|| {
            anyhow!(
                "Worker for model '{}' does not have a tokenizer path configured.",
                model
            )
        })?;

        debug!(
            model = %model,
            tokenizer_path = %tokenizer_path,
            "Creating tokenizer for model"
        );

        // Create tokenizer (this may download from HuggingFace Hub)
        let base_tokenizer = tokio::task::spawn_blocking({
            let tokenizer_path = tokenizer_path.to_string();
            move || create_tokenizer_with_chat_template_blocking(&tokenizer_path, None)
        })
        .await
        .map_err(|e| anyhow!("Failed to spawn tokenizer creation task: {}", e))?
        .map_err(|e| anyhow!("Failed to create tokenizer: {}", e))?;

        // Wrap with caching layer if enabled
        let tokenizer: Arc<dyn Tokenizer> =
            if self.cache_config.enable_l0 || self.cache_config.enable_l1 {
                Arc::new(CachedTokenizer::new(base_tokenizer, self.cache_config.clone()))
            } else {
                base_tokenizer
            };

        // Cache and return
        self.cache.insert(model.to_string(), tokenizer.clone());
        Ok(tokenizer)
    }

    /// Encode text using the tokenizer for the given model.
    pub async fn encode(&self, model: &str, text: &str) -> Result<Vec<u32>> {
        let tokenizer = self.get_tokenizer(model).await?;
        let encoding = tokenizer.encode(text)?;
        Ok(encoding.token_ids().to_vec())
    }

    /// Encode a batch of texts using the tokenizer for the given model.
    pub async fn encode_batch(&self, model: &str, texts: &[&str]) -> Result<Vec<Vec<u32>>> {
        let tokenizer = self.get_tokenizer(model).await?;
        let encodings = tokenizer.encode_batch(texts)?;
        Ok(encodings.into_iter().map(|e| e.token_ids().to_vec()).collect())
    }

    /// Decode token IDs using the tokenizer for the given model.
    pub async fn decode(
        &self,
        model: &str,
        token_ids: &[u32],
        skip_special_tokens: bool,
    ) -> Result<String> {
        let tokenizer = self.get_tokenizer(model).await?;
        tokenizer.decode(token_ids, skip_special_tokens)
    }
}
