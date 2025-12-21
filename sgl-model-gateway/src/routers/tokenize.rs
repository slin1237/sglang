//! Tokenize endpoint handlers for /v1/tokenize and /v1/detokenize.
//!
//! This module provides handlers for tokenization and detokenization requests,
//! using the TokenizerCache to look up and cache tokenizers by model ID.

use anyhow::Result;

use crate::{
    protocols::tokenize::{
        CountOutput, DetokenizeRequest, DetokenizeResponse, StringOrArray, TextOutput,
        TokenizeRequest, TokenizeResponse, TokensInput, TokensOutput,
    },
    tokenizer::TokenizerCache,
};

/// Default max model length when not available from tokenizer
const DEFAULT_MAX_MODEL_LEN: i64 = -1;

/// Handle a tokenize request using the provided tokenizer cache.
pub async fn handle_tokenize(
    cache: &TokenizerCache,
    request: &TokenizeRequest,
) -> Result<TokenizeResponse> {
    let max_model_len = DEFAULT_MAX_MODEL_LEN;

    match &request.prompt {
        StringOrArray::String(text) => {
            let token_ids = cache.encode(&request.model, text).await?;
            let count = token_ids.len();

            Ok(TokenizeResponse {
                tokens: TokensOutput::Single(token_ids),
                count: CountOutput::Single(count),
                max_model_len,
            })
        }
        StringOrArray::Array(texts) => {
            let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
            let token_ids_batch = cache.encode_batch(&request.model, &refs).await?;
            let counts: Vec<usize> = token_ids_batch.iter().map(|ids| ids.len()).collect();

            Ok(TokenizeResponse {
                tokens: TokensOutput::Batch(token_ids_batch),
                count: CountOutput::Batch(counts),
                max_model_len,
            })
        }
    }
}

/// Handle a detokenize request using the provided tokenizer cache.
pub async fn handle_detokenize(
    cache: &TokenizerCache,
    request: &DetokenizeRequest,
) -> Result<DetokenizeResponse> {
    match &request.tokens {
        TokensInput::Single(token_ids) => {
            let text = cache
                .decode(&request.model, token_ids, request.skip_special_tokens)
                .await?;
            Ok(DetokenizeResponse {
                text: TextOutput::Single(text),
            })
        }
        TokensInput::Batch(token_ids_batch) => {
            let mut texts = Vec::with_capacity(token_ids_batch.len());
            for token_ids in token_ids_batch {
                let text = cache
                    .decode(&request.model, token_ids, request.skip_special_tokens)
                    .await?;
                texts.push(text);
            }
            Ok(DetokenizeResponse {
                text: TextOutput::Batch(texts),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::core::WorkerRegistry;

    #[tokio::test]
    async fn test_handle_tokenize_missing_model() {
        let registry = Arc::new(WorkerRegistry::new());
        let cache = TokenizerCache::new(registry);

        let request = TokenizeRequest {
            model: "nonexistent-model".to_string(),
            prompt: StringOrArray::String("hello".to_string()),
            add_special_tokens: true,
        };

        // This should fail because the model doesn't exist and can't be downloaded
        let result = handle_tokenize(&cache, &request).await;
        assert!(result.is_err());
    }
}
