//! Runtime detection step for local workers.
//!
//! Detects whether a worker is running SGLang or vLLM by checking
//! the /metrics endpoint for distinctive metric prefixes:
//! - SGLang: metrics have "sgl_" prefix
//! - vLLM: metrics have "vllm:" or "vllm_" prefix
//!
//! This is the most reliable detection method because metric prefixes
//! are stable - changing them would break existing dashboards and monitoring.

use std::{sync::Arc, time::Duration};

use async_trait::async_trait;
use once_cell::sync::Lazy;
use reqwest::Client;
use tracing::{debug, warn};

use crate::{
    core::{ConnectionMode, RuntimeType},
    protocols::worker_spec::WorkerConfigRequest,
    workflow::{StepExecutor, StepResult, WorkflowContext, WorkflowError, WorkflowResult},
};

// HTTP client for runtime detection
static HTTP_CLIENT: Lazy<Client> = Lazy::new(|| {
    Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .expect("Failed to create HTTP client")
});

/// Detect runtime type by checking /metrics endpoint prefixes.
///
/// This is the most reliable detection method:
/// - SGLang metrics use "sgl_" prefix
/// - vLLM metrics use "vllm:" or "vllm_" prefix
///
/// Returns None if metrics endpoint is unavailable or prefixes not found.
pub async fn detect_runtime_from_metrics(url: &str, api_key: Option<&str>) -> Option<RuntimeType> {
    let base_url = url.trim_end_matches('/');
    let metrics_url = format!("{}/metrics", base_url);

    let mut req = HTTP_CLIENT.get(&metrics_url);
    if let Some(key) = api_key {
        req = req.bearer_auth(key);
    }

    let response = match req.send().await {
        Ok(resp) => resp,
        Err(e) => {
            debug!("Failed to fetch metrics from {}: {}", metrics_url, e);
            return None;
        }
    };

    if !response.status().is_success() {
        debug!(
            "Metrics endpoint returned status {} for {}",
            response.status(),
            metrics_url
        );
        return None;
    }

    let text = match response.text().await {
        Ok(t) => t,
        Err(e) => {
            debug!("Failed to read metrics response from {}: {}", metrics_url, e);
            return None;
        }
    };

    // Check for vLLM metric prefixes (vllm: or vllm_)
    // vLLM uses "vllm:" for newer metrics and "vllm_" for some older ones
    if text.contains("vllm:") || text.contains("vllm_") {
        debug!(
            "Detected vLLM runtime for {} (metrics contain vllm prefix)",
            url
        );
        return Some(RuntimeType::Vllm);
    }

    // Check for SGLang metric prefix (sgl_)
    if text.contains("sgl_") {
        debug!(
            "Detected SGLang runtime for {} (metrics contain sgl_ prefix)",
            url
        );
        return Some(RuntimeType::Sglang);
    }

    debug!(
        "Could not determine runtime from metrics for {} (no known prefixes found)",
        url
    );
    None
}

/// Step to detect runtime type (SGLang vs vLLM) for HTTP workers.
///
/// This step runs after connection mode detection and before metadata discovery.
/// It uses the /metrics endpoint to reliably detect the runtime type.
pub struct DetectRuntimeStep;

#[async_trait]
impl StepExecutor for DetectRuntimeStep {
    async fn execute(&self, context: &mut WorkflowContext) -> WorkflowResult<StepResult> {
        let config: Arc<WorkerConfigRequest> = context.get_or_err("worker_config")?;
        let connection_mode: Arc<ConnectionMode> = context.get_or_err("connection_mode")?;

        // Only detect runtime for HTTP workers
        // gRPC workers use protocol-based detection in the gRPC client
        if !matches!(connection_mode.as_ref(), ConnectionMode::Http) {
            debug!(
                "Skipping runtime detection for non-HTTP worker {}",
                config.url
            );
            // Default to SGLang for gRPC (will be refined by gRPC client)
            context.set("detected_runtime_type", "sglang".to_string());
            return Ok(StepResult::Success);
        }

        // Check if runtime is explicitly specified in config
        if let Some(ref explicit_runtime) = config.runtime {
            debug!(
                "Using explicitly specified runtime type '{}' for {}",
                explicit_runtime, config.url
            );
            context.set("detected_runtime_type", explicit_runtime.clone());
            return Ok(StepResult::Success);
        }

        // Detect runtime from metrics
        let detected = detect_runtime_from_metrics(&config.url, config.api_key.as_deref()).await;

        let runtime_str = match detected {
            Some(RuntimeType::Vllm) => "vllm",
            Some(RuntimeType::Sglang) => "sglang",
            Some(RuntimeType::External) => "external",
            None => {
                // Default to SGLang if detection fails
                warn!(
                    "Could not detect runtime type for {}, defaulting to SGLang",
                    config.url
                );
                "sglang"
            }
        };

        debug!("Detected runtime type '{}' for {}", runtime_str, config.url);
        context.set("detected_runtime_type", runtime_str.to_string());

        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        true
    }
}
