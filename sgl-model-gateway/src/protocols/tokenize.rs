use serde::{Deserialize, Serialize};

// ============================================================================
// Tokenize API (/v1/tokenize and /tokenize)
// ============================================================================

/// Request schema for the /tokenize endpoint.
/// Compatible with the Python SGLang implementation.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TokenizeRequest {
    /// ID of the model to use for tokenization
    #[serde(default = "default_model_name")]
    pub model: String,

    /// The prompt(s) to tokenize. Can be a single string or a list of strings.
    pub prompt: StringOrArray,

    /// Whether to add model-specific special tokens (e.g. BOS/EOS) during encoding.
    #[serde(default = "default_true")]
    pub add_special_tokens: bool,
}

/// Response schema for the /tokenize endpoint.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TokenizeResponse {
    /// Token IDs for the input prompt(s).
    /// If input was a single string, returns List[int].
    /// If input was a list of strings, returns List[List[int]].
    pub tokens: TokensOutput,

    /// Token count(s) for the input prompt(s).
    /// If input was a single string, returns int.
    /// If input was a list of strings, returns List[int].
    pub count: CountOutput,

    /// Maximum model context length, if available.
    #[serde(default)]
    pub max_model_len: i64,
}

// ============================================================================
// Detokenize API (/v1/detokenize and /detokenize)
// ============================================================================

/// Request schema for the /detokenize endpoint.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DetokenizeRequest {
    /// ID of the model to use for detokenization
    #[serde(default = "default_model_name")]
    pub model: String,

    /// The token IDs to decode. Can be a list of integers or a list of lists of integers.
    pub tokens: TokensInput,

    /// Whether to exclude special tokens (e.g. padding or EOS) during decoding.
    #[serde(default = "default_true")]
    pub skip_special_tokens: bool,
}

/// Response schema for the /detokenize endpoint.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DetokenizeResponse {
    /// Decoded text for the input token(s).
    /// If input was List[int], returns str.
    /// If input was List[List[int]], returns List[str].
    pub text: TextOutput,
}

// ============================================================================
// Helper Types
// ============================================================================

/// Represents either a single string or an array of strings
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum StringOrArray {
    String(String),
    Array(Vec<String>),
}

/// Represents either a single list of token IDs or a list of lists of token IDs
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum TokensInput {
    Single(Vec<u32>),
    Batch(Vec<Vec<u32>>),
}

/// Represents the output tokens - either a single list or a list of lists
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum TokensOutput {
    Single(Vec<u32>),
    Batch(Vec<Vec<u32>>),
}

/// Represents the count output - either a single count or a list of counts
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum CountOutput {
    Single(usize),
    Batch(Vec<usize>),
}

/// Represents the text output - either a single string or a list of strings
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum TextOutput {
    Single(String),
    Batch(Vec<String>),
}

// ============================================================================
// Default values
// ============================================================================

fn default_model_name() -> String {
    "default".to_string()
}

fn default_true() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize_request_deserialize_single_string() {
        let json = r#"{"prompt": "Hello, world!"}"#;
        let req: TokenizeRequest = serde_json::from_str(json).unwrap();
        assert!(matches!(req.prompt, StringOrArray::String(_)));
        assert_eq!(req.model, "default");
        assert!(req.add_special_tokens);
    }

    #[test]
    fn test_tokenize_request_deserialize_array() {
        let json = r#"{"model": "gpt-4", "prompt": ["Hello", "World"], "add_special_tokens": false}"#;
        let req: TokenizeRequest = serde_json::from_str(json).unwrap();
        assert!(matches!(req.prompt, StringOrArray::Array(_)));
        assert_eq!(req.model, "gpt-4");
        assert!(!req.add_special_tokens);
    }

    #[test]
    fn test_tokenize_response_serialize_single() {
        let resp = TokenizeResponse {
            tokens: TokensOutput::Single(vec![1, 2, 3]),
            count: CountOutput::Single(3),
            max_model_len: 4096,
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("[1,2,3]"));
        assert!(json.contains("\"count\":3"));
    }

    #[test]
    fn test_tokenize_response_serialize_batch() {
        let resp = TokenizeResponse {
            tokens: TokensOutput::Batch(vec![vec![1, 2], vec![3, 4, 5]]),
            count: CountOutput::Batch(vec![2, 3]),
            max_model_len: 4096,
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("[[1,2],[3,4,5]]"));
        assert!(json.contains("[2,3]"));
    }

    #[test]
    fn test_detokenize_request_deserialize() {
        let json = r#"{"tokens": [1, 2, 3]}"#;
        let req: DetokenizeRequest = serde_json::from_str(json).unwrap();
        assert!(matches!(req.tokens, TokensInput::Single(_)));
        assert!(req.skip_special_tokens);
    }

    #[test]
    fn test_detokenize_request_deserialize_batch() {
        let json = r#"{"tokens": [[1, 2], [3, 4, 5]], "skip_special_tokens": false}"#;
        let req: DetokenizeRequest = serde_json::from_str(json).unwrap();
        assert!(matches!(req.tokens, TokensInput::Batch(_)));
        assert!(!req.skip_special_tokens);
    }
}
