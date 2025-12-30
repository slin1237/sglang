//! vLLM PD (Prefill-Decode) Router Implementation
//!
//! This router handles prefill-decode disaggregated inference for vLLM backends.
//! Unlike SGLang's PD mode which uses bootstrap_host/port/room for KV cache transfer,
//! vLLM uses kv_transfer_params for coordination between prefill and decode instances.
//!
//! Key differences from SGLang PD router:
//! 1. Uses `kv_transfer_params` with `do_remote_decode`/`do_remote_prefill` flags
//! 2. Two-stage flow: prefill returns `kv_transfer_params`, decode uses those params
//! 3. Special request ID format sent via X-Request-Id header for P2P coordination
//! 4. Prefill stage uses `max_tokens=1` to generate KV cache without output tokens

use std::{sync::Arc, time::Instant};

use async_trait::async_trait;
use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
};
use futures_util::StreamExt;
use memchr::memmem;
use reqwest::Client;
use serde::Serialize;
use serde_json::{json, Value};
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::{debug, error, warn};
use uuid::Uuid;

use super::pd_types::api_path;
use crate::{
    config::types::RetryConfig,
    core::{
        is_retryable_status, HashRing, RetryExecutor, Worker, WorkerLoadGuard, WorkerRegistry,
        WorkerType, UNKNOWN_MODEL_ID,
    },
    observability::{
        events::{self, Event},
        metrics::{bool_to_static_str, metrics_labels, Metrics},
        otel_trace::inject_trace_context_http,
    },
    policies::{LoadBalancingPolicy, PolicyRegistry, SelectWorkerInfo},
    protocols::{
        chat::{ChatCompletionRequest, ChatMessage, MessageContent},
        common::StringOrArray,
        completion::CompletionRequest,
        generate::GenerateRequest,
        rerank::RerankRequest,
    },
    routers::{
        error,
        grpc::utils::{error_type_from_status, route_to_endpoint},
        header_utils, RouterTrait,
    },
};

/// vLLM PD Router for disaggregated prefill-decode inference
#[derive(Debug)]
pub struct VllmPDRouter {
    pub worker_registry: Arc<WorkerRegistry>,
    pub policy_registry: Arc<PolicyRegistry>,
    pub client: Client,
    pub retry_config: RetryConfig,
    pub api_key: Option<String>,
    pub enable_igw: bool,
}

#[derive(Clone)]
struct VllmPDRequestContext<'a> {
    route: &'static str,
    is_stream: bool,
    return_logprob: bool,
    request_text: Option<String>,
    model_id: Option<&'a str>,
    headers: Option<HeaderMap>,
}

impl VllmPDRouter {
    /// Generate a vLLM-style request ID for PD coordination
    /// Format: ___prefill_addr_{prefill_addr}___decode_addr_{decode_addr}_{uuid}
    /// This is sent via X-Request-Id header for P2P coordination between instances
    #[inline]
    fn generate_vllm_request_id(prefill_addr: &str, decode_addr: &str) -> String {
        let uuid = Uuid::new_v4().simple();
        format!(
            "___prefill_addr_{}___decode_addr_{}_{}",
            prefill_addr, decode_addr, uuid
        )
    }

    /// Extract HTTP address from worker URL (without scheme)
    #[inline]
    fn get_worker_http_addr(worker: &dyn Worker) -> &str {
        let url = worker.url();
        url.strip_prefix("https://")
            .or_else(|| url.strip_prefix("http://"))
            .unwrap_or(url)
    }

    async fn proxy_to_first_prefill_worker(
        &self,
        endpoint: &str,
        headers: Option<Vec<(String, String)>>,
    ) -> Response {
        let workers = self.worker_registry.get_prefill_workers();
        let first_worker_url = workers.first().map(|w| w.url().to_string());

        if let Some(worker_url) = first_worker_url {
            self.proxy_to_worker(worker_url, endpoint, headers).await
        } else {
            error::service_unavailable("no_prefill_servers", "No prefill servers available")
        }
    }

    async fn proxy_to_worker(
        &self,
        worker_url: String,
        endpoint: &str,
        headers: Option<Vec<(String, String)>>,
    ) -> Response {
        let url = format!("{}/{}", worker_url, endpoint);
        let mut request_builder = self.client.get(&url);

        if let Some(headers) = headers {
            for (name, value) in headers {
                request_builder = request_builder.header(name, value);
            }
        }

        match request_builder.send().await {
            Ok(res) if res.status().is_success() => {
                let response_headers = header_utils::preserve_response_headers(res.headers());

                match res.bytes().await {
                    Ok(body) => {
                        let mut response = Response::new(Body::from(body));
                        *response.status_mut() = StatusCode::OK;
                        *response.headers_mut() = response_headers;
                        response
                    }
                    Err(e) => {
                        error!(error = %e, "Failed to read response body");
                        error::internal_error(
                            "read_response_body_failed",
                            format!("Failed to read response body: {}", e),
                        )
                    }
                }
            }
            Ok(res) => {
                let status = StatusCode::from_u16(res.status().as_u16())
                    .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                Self::status_to_error_response(
                    status,
                    "server",
                    format!("Server returned status: {}", res.status()),
                )
            }
            Err(e) => {
                error!(error = %e, "Failed to proxy request to server");
                error::internal_error(
                    "proxy_request_failed",
                    format!("Failed to proxy request: {}", e),
                )
            }
        }
    }

    pub async fn new(ctx: &Arc<crate::app_context::AppContext>) -> Result<Self, String> {
        Ok(VllmPDRouter {
            worker_registry: Arc::clone(&ctx.worker_registry),
            policy_registry: Arc::clone(&ctx.policy_registry),
            client: ctx.client.clone(),
            retry_config: ctx.router_config.effective_retry_config(),
            api_key: ctx.router_config.api_key.clone(),
            enable_igw: ctx.router_config.enable_igw,
        })
    }

    #[inline]
    fn handle_server_selection_error(error: String) -> Response {
        error!(error = %error, "Failed to select PD pair");
        error::service_unavailable(
            "server_selection_failed",
            format!("No available servers: {}", error),
        )
    }

    #[inline]
    fn handle_serialization_error(error: impl std::fmt::Display) -> Response {
        error!(error = %error, "Failed to serialize request");
        error::internal_error("serialization_failed", "Failed to serialize request")
    }

    /// Map HTTP status code to appropriate error response
    #[inline]
    fn status_to_error_response(
        status: StatusCode,
        error_code_prefix: &str,
        message: impl Into<String>,
    ) -> Response {
        let msg = message.into();
        match status {
            StatusCode::BAD_REQUEST => {
                error::bad_request(&format!("{}_bad_request", error_code_prefix), msg)
            }
            StatusCode::NOT_FOUND => {
                error::not_found(&format!("{}_not_found", error_code_prefix), msg)
            }
            StatusCode::INTERNAL_SERVER_ERROR => {
                error::internal_error(&format!("{}_internal_error", error_code_prefix), msg)
            }
            StatusCode::SERVICE_UNAVAILABLE => {
                error::service_unavailable(&format!("{}_unavailable", error_code_prefix), msg)
            }
            StatusCode::BAD_GATEWAY => {
                error::bad_gateway(&format!("{}_bad_gateway", error_code_prefix), msg)
            }
            _ => error::internal_error(&format!("{}_error", error_code_prefix), msg),
        }
    }

    /// Generate the vLLM request ID for P2P coordination between prefill and decode
    #[inline]
    fn generate_request_id_for_workers(
        prefill_worker: &dyn Worker,
        decode_worker: &dyn Worker,
    ) -> String {
        let prefill_addr = Self::get_worker_http_addr(prefill_worker);
        let decode_addr = Self::get_worker_http_addr(decode_worker);
        Self::generate_vllm_request_id(prefill_addr, decode_addr)
    }

    /// Prepare the prefill request for vLLM
    /// vLLM prefill uses max_tokens=1 to just generate KV cache
    fn prepare_prefill_request(request: &Value) -> Value {
        let mut prefill_req = request.clone();
        if let Some(obj) = prefill_req.as_object_mut() {
            // Set max_tokens=1 for prefill (just generate KV cache, minimal output)
            obj.insert("max_tokens".to_string(), json!(1));

            // Also set max_completion_tokens=1 if present (OpenAI API compatibility)
            if obj.contains_key("max_completion_tokens") {
                obj.insert("max_completion_tokens".to_string(), json!(1));
            }

            // Disable streaming for prefill to get JSON response with kv_transfer_params
            obj.insert("stream".to_string(), json!(false));

            // Remove stream_options since we're setting stream=false
            obj.remove("stream_options");

            // Add kv_transfer_params for prefill stage
            // This tells vLLM to prepare for remote decode
            obj.insert(
                "kv_transfer_params".to_string(),
                json!({
                    "do_remote_decode": true,
                    "do_remote_prefill": false,
                    "remote_engine_id": Value::Null,
                    "remote_block_ids": Value::Null,
                    "remote_host": Value::Null,
                    "remote_port": Value::Null
                }),
            );
        }
        prefill_req
    }

    /// Prepare the decode request with kv_transfer_params from prefill response
    fn prepare_decode_request(original_request: &Value, kv_transfer_params: Option<Value>) -> Value {
        let mut decode_req = original_request.clone();
        if let Some(obj) = decode_req.as_object_mut() {
            // Add kv_transfer_params from prefill response if available
            if let Some(params) = kv_transfer_params {
                obj.insert("kv_transfer_params".to_string(), params);
            }
        }
        decode_req
    }

    async fn execute_dual_dispatch<T: Serialize + Clone>(
        &self,
        headers: Option<&HeaderMap>,
        original_request: &T,
        context: VllmPDRequestContext<'_>,
    ) -> Response {
        let start_time = Instant::now();

        let route = context.route;
        let model = context.model_id.unwrap_or("default");
        let endpoint = route_to_endpoint(route);

        // Record request start (Layer 2)
        Metrics::record_router_request(
            metrics_labels::ROUTER_HTTP,
            metrics_labels::BACKEND_PD,
            metrics_labels::CONNECTION_HTTP,
            model,
            endpoint,
            bool_to_static_str(context.is_stream),
        );

        let shared_request = Arc::new(original_request.clone());
        let response = RetryExecutor::execute_response_with_retry(
            &self.retry_config,
            {
                move |attempt: u32| {
                    let shared_request = Arc::clone(&shared_request);
                    let context = context.clone();
                    async move {
                        let (prefill, decode) = match self
                            .select_pd_pair(
                                context.request_text.as_deref(),
                                context.model_id,
                                context.headers.as_ref(),
                            )
                            .await
                        {
                            Ok(pair) => pair,
                            Err(e) => {
                                return Self::handle_server_selection_error(e);
                            }
                        };

                        debug!(
                            "vLLM PD retry attempt {} using prefill={} decode={}",
                            attempt,
                            prefill.url(),
                            decode.url()
                        );

                        let json_request = match serde_json::to_value(shared_request.as_ref()) {
                            Ok(v) => v,
                            Err(e) => return Self::handle_serialization_error(e),
                        };

                        // Generate request ID for P2P coordination (sent via header)
                        let request_id = Self::generate_request_id_for_workers(
                            prefill.as_ref(),
                            decode.as_ref(),
                        );

                        let response = self
                            .execute_vllm_dual_dispatch_internal(
                                headers,
                                json_request,
                                context,
                                Arc::clone(&prefill),
                                Arc::clone(&decode),
                                &request_id,
                            )
                            .await;

                        let status = response.status();
                        let not_error = status.is_success() || status.is_client_error();
                        prefill.record_outcome(not_error);
                        decode.record_outcome(not_error);

                        if status.is_server_error() {
                            let error_type = error_type_from_status(status);
                            Metrics::record_worker_error(
                                metrics_labels::WORKER_PREFILL,
                                metrics_labels::CONNECTION_HTTP,
                                error_type,
                            );
                            Metrics::record_worker_error(
                                metrics_labels::WORKER_DECODE,
                                metrics_labels::CONNECTION_HTTP,
                                error_type,
                            );
                        }

                        response
                    }
                }
            },
            |res, _attempt| is_retryable_status(res.status()),
            |delay, attempt| {
                Metrics::record_worker_retry(metrics_labels::WORKER_PREFILL, endpoint);
                Metrics::record_worker_retry(metrics_labels::WORKER_DECODE, endpoint);
                Metrics::record_worker_retry_backoff(attempt, delay);
            },
            || {
                Metrics::record_worker_retries_exhausted(metrics_labels::WORKER_PREFILL, endpoint);
                Metrics::record_worker_retries_exhausted(metrics_labels::WORKER_DECODE, endpoint);
            },
        )
        .await;

        let duration = start_time.elapsed();
        if response.status().is_success() {
            Metrics::record_router_duration(
                metrics_labels::ROUTER_HTTP,
                metrics_labels::BACKEND_PD,
                metrics_labels::CONNECTION_HTTP,
                model,
                endpoint,
                duration,
            );
        } else if !is_retryable_status(response.status()) {
            Metrics::record_router_error(
                metrics_labels::ROUTER_HTTP,
                metrics_labels::BACKEND_PD,
                metrics_labels::CONNECTION_HTTP,
                model,
                endpoint,
                error_type_from_status(response.status()),
            );
        }

        response
    }

    /// Execute vLLM two-stage PD dispatch
    /// Stage 1: Send prefill request with max_tokens=1 and kv_transfer_params
    /// Stage 2: Send decode request with kv_transfer_params from prefill response
    async fn execute_vllm_dual_dispatch_internal(
        &self,
        headers: Option<&HeaderMap>,
        json_request: Value,
        context: VllmPDRequestContext<'_>,
        prefill: Arc<dyn Worker>,
        decode: Arc<dyn Worker>,
        request_id: &str,
    ) -> Response {
        // For non-streaming: use guard for automatic load management
        // For streaming: load will be managed in create_streaming_response
        let _prefill_guard = (!context.is_stream).then(|| WorkerLoadGuard::new(prefill.clone()));
        let _decode_guard = (!context.is_stream).then(|| WorkerLoadGuard::new(decode.clone()));

        let mut headers_with_trace = headers.cloned().unwrap_or_default();
        inject_trace_context_http(&mut headers_with_trace);
        // Add vLLM P2P coordination header
        if let Ok(header_value) = HeaderValue::from_str(request_id) {
            headers_with_trace.insert("x-request-id", header_value);
        }
        let headers = Some(&headers_with_trace);

        debug!(
            "vLLM PD: Starting two-stage dispatch with request_id={}",
            request_id
        );

        // Prepare prefill request (max_tokens=1, stream=false, kv_transfer_params)
        let prefill_request_json = Self::prepare_prefill_request(&json_request);

        // Build prefill request with X-Request-Id header for P2P coordination
        let prefill_request = self.build_post_with_headers(
            &self.client,
            prefill.url(),
            context.route,
            &prefill_request_json,
            headers,
            false,
        );

        events::RequestPDSentEvent {
            prefill_url: prefill.url(),
            decode_url: decode.url(),
        }
        .emit();

        debug!(
            "vLLM PD Stage 1: Sending prefill request to {} with max_tokens=1",
            prefill.url()
        );

        // Stage 1: Execute prefill
        let prefill_result = prefill_request.send().await;

        // Process prefill response and extract kv_transfer_params
        let kv_params = match self
            .process_vllm_prefill_response(prefill_result, prefill.url())
            .await
        {
            Ok(params) => params,
            Err(error_response) => return error_response,
        };

        // Stage 2: Build decode request with kv_transfer_params from prefill response
        let decode_request_json = Self::prepare_decode_request(&json_request, kv_params);

        debug!(
            "vLLM PD Stage 2: Sending decode request to {}",
            decode.url()
        );

        let decode_request = self.build_post_with_headers(
            &self.client,
            decode.url(),
            context.route,
            &decode_request_json,
            headers,
            false,
        );

        // Execute decode request
        let decode_result = decode_request.send().await;

        events::RequestReceivedEvent {}.emit();

        // Process decode response
        match decode_result {
            Ok(res) => {
                let status = StatusCode::from_u16(res.status().as_u16())
                    .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                debug!(status = %status, "Decode response received");

                if !status.is_success() {
                    error!(
                        decode_url = %decode.url(),
                        status = %status,
                        "Decode server returned error status"
                    );
                    return self
                        .handle_decode_error_response(res, &context, prefill, decode)
                        .await;
                }

                if context.is_stream {
                    let response_headers = header_utils::preserve_response_headers(res.headers());
                    self.create_streaming_response(
                        res.bytes_stream(),
                        status,
                        None,
                        context.return_logprob,
                        None,
                        Some(response_headers),
                        prefill,
                        decode,
                    )
                } else {
                    // Direct passthrough for non-streaming
                    let response_headers = header_utils::preserve_response_headers(res.headers());
                    match res.bytes().await {
                        Ok(decode_body) => {
                            let mut response = Response::new(Body::from(decode_body));
                            *response.status_mut() = status;
                            *response.headers_mut() = response_headers;
                            response
                        }
                        Err(e) => {
                            error!(error = %e, "Failed to read decode response");
                            error::internal_error(
                                "read_response_failed",
                                "Failed to read response",
                            )
                        }
                    }
                }
            }
            Err(e) => {
                error!(
                    decode_url = %decode.url(),
                    error = %e,
                    "Decode request failed"
                );
                error::bad_gateway("decode_server_error", format!("Decode server error: {}", e))
            }
        }
    }

    /// Process vLLM prefill response and extract KV transfer params
    async fn process_vllm_prefill_response(
        &self,
        prefill_result: Result<reqwest::Response, reqwest::Error>,
        prefill_url: &str,
    ) -> Result<Option<Value>, Response> {
        let prefill_response = match prefill_result {
            Ok(response) => response,
            Err(e) => {
                error!(
                    prefill_url = %prefill_url,
                    error = %e,
                    "Prefill server failed (CRITICAL). Decode will fail without prefill KV cache"
                );
                return Err(error::bad_gateway(
                    "prefill_server_error",
                    format!(
                        "Prefill server error: {}. This will cause decode failure.",
                        e
                    ),
                ));
            }
        };

        let prefill_status = StatusCode::from_u16(prefill_response.status().as_u16())
            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

        if !prefill_status.is_success() {
            let error_msg = prefill_response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown prefill error".to_string());

            error!(
                prefill_url = %prefill_url,
                status = %prefill_status,
                body = %error_msg,
                "Prefill server returned error status"
            );

            return Err(Self::status_to_error_response(
                prefill_status,
                "prefill",
                format!("Prefill server error ({}): {}", prefill_status, error_msg),
            ));
        }

        // Read prefill body to extract KV transfer params
        match prefill_response.bytes().await {
            Ok(body) => {
                // Try to parse and extract kv_transfer_params
                if let Ok(json) = serde_json::from_slice::<Value>(&body) {
                    if let Some(kv_params) = json.get("kv_transfer_params") {
                        debug!("Extracted kv_transfer_params from prefill response");
                        return Ok(Some(kv_params.clone()));
                    }
                }
                debug!("No kv_transfer_params in prefill response, using injected params");
                Ok(None)
            }
            Err(e) => {
                warn!(error = %e, "Failed to read prefill response body");
                Ok(None)
            }
        }
    }

    async fn handle_decode_error_response(
        &self,
        res: reqwest::Response,
        context: &VllmPDRequestContext<'_>,
        prefill: Arc<dyn Worker>,
        decode: Arc<dyn Worker>,
    ) -> Response {
        let status = res.status();

        if context.is_stream {
            let response_headers = header_utils::preserve_response_headers(res.headers());
            let error_payload = match res.bytes().await {
                Ok(error_body) => {
                    if let Ok(error_json) = serde_json::from_slice::<Value>(&error_body) {
                        json!({ "message": error_json, "status": status.as_u16() })
                    } else {
                        json!({ "message": String::from_utf8_lossy(&error_body).to_string(), "status": status.as_u16() })
                    }
                }
                Err(e) => {
                    json!({ "message": format!("Decode server error: {}", e), "status": status.as_u16() })
                }
            };

            let sse_data = format!(
                "data: {{'error': {}}}",
                serde_json::to_string(&error_payload).unwrap_or_default()
            );
            let error_stream = tokio_stream::once(Ok(axum::body::Bytes::from(sse_data)));

            let decode_url = decode.url().to_string();
            self.create_streaming_response(
                error_stream,
                status,
                None,
                context.return_logprob,
                Some(decode_url),
                Some(response_headers),
                prefill,
                decode,
            )
        } else {
            let status_code =
                StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

            match res.bytes().await {
                Ok(error_body) => {
                    let error_message = if let Ok(error_json) =
                        serde_json::from_slice::<Value>(&error_body)
                    {
                        error_json
                            .get("error")
                            .and_then(|e| e.get("message"))
                            .and_then(|m| m.as_str())
                            .or_else(|| error_json.get("message").and_then(|m| m.as_str()))
                            .map(String::from)
                            .unwrap_or_else(|| String::from_utf8_lossy(&error_body).into_owned())
                    } else {
                        String::from_utf8_lossy(&error_body).into_owned()
                    };

                    Self::status_to_error_response(status_code, "decode", error_message)
                }
                Err(e) => Self::status_to_error_response(
                    status_code,
                    "decode_read_failed",
                    format!("Decode server error: {}", e),
                ),
            }
        }
    }

    #[inline]
    fn policies_need_request_text(&self) -> bool {
        let prefill_policy = self.policy_registry.get_prefill_policy();
        let decode_policy = self.policy_registry.get_decode_policy();
        prefill_policy.needs_request_text() || decode_policy.needs_request_text()
    }

    async fn select_pd_pair(
        &self,
        request_text: Option<&str>,
        model_id: Option<&str>,
        headers: Option<&HeaderMap>,
    ) -> Result<(Arc<dyn Worker>, Arc<dyn Worker>), String> {
        let effective_model_id = if !self.enable_igw { None } else { model_id };

        debug!(
            "Selecting PD pair: enable_igw={}, model_id={:?}, effective_model_id={:?}",
            self.enable_igw, model_id, effective_model_id
        );

        let prefill_workers = if let Some(model) = effective_model_id {
            self.worker_registry
                .get_by_model_fast(model)
                .iter()
                .filter(|w| matches!(w.worker_type(), WorkerType::Prefill { .. }))
                .cloned()
                .collect()
        } else {
            self.worker_registry.get_prefill_workers()
        };

        let decode_workers = if let Some(model) = effective_model_id {
            self.worker_registry
                .get_by_model_fast(model)
                .iter()
                .filter(|w| matches!(w.worker_type(), WorkerType::Decode))
                .cloned()
                .collect()
        } else {
            self.worker_registry.get_decode_workers()
        };

        let prefill_policy = self.policy_registry.get_prefill_policy();
        let decode_policy = self.policy_registry.get_decode_policy();

        let hash_ring = self
            .worker_registry
            .get_hash_ring(effective_model_id.unwrap_or(UNKNOWN_MODEL_ID));

        let prefill = Self::pick_worker_by_policy_arc(
            &prefill_workers,
            &*prefill_policy,
            request_text,
            headers,
            hash_ring.clone(),
            "prefill",
        )?;

        let decode = Self::pick_worker_by_policy_arc(
            &decode_workers,
            &*decode_policy,
            request_text,
            headers,
            hash_ring,
            "decode",
        )?;

        let model = model_id.unwrap_or("default");
        Metrics::record_worker_selection(
            metrics_labels::WORKER_PREFILL,
            metrics_labels::CONNECTION_HTTP,
            model,
            prefill_policy.name(),
        );
        Metrics::record_worker_selection(
            metrics_labels::WORKER_DECODE,
            metrics_labels::CONNECTION_HTTP,
            model,
            decode_policy.name(),
        );

        Ok((prefill, decode))
    }

    fn pick_worker_by_policy_arc(
        workers: &[Arc<dyn Worker>],
        policy: &dyn LoadBalancingPolicy,
        request_text: Option<&str>,
        headers: Option<&HeaderMap>,
        hash_ring: Option<Arc<HashRing>>,
        worker_type: &str,
    ) -> Result<Arc<dyn Worker>, String> {
        if workers.is_empty() {
            return Err(format!(
                "No {} workers available. Please check if {} servers are configured and healthy.",
                worker_type, worker_type
            ));
        }

        let available_workers: Vec<Arc<dyn Worker>> = workers
            .iter()
            .filter(|w| w.is_available())
            .cloned()
            .collect();

        if available_workers.is_empty() {
            return Err(format!(
                "No available {} workers (all circuits open or unhealthy)",
                worker_type
            ));
        }

        let selected_idx = policy
            .select_worker(
                &available_workers,
                &SelectWorkerInfo {
                    request_text,
                    tokens: None,
                    headers,
                    hash_ring,
                },
            )
            .ok_or_else(|| {
                format!(
                    "Policy {} failed to select a {} worker",
                    policy.name(),
                    worker_type
                )
            })?;

        Ok(available_workers[selected_idx].clone())
    }

    #[allow(clippy::too_many_arguments)]
    fn create_streaming_response(
        &self,
        stream: impl futures_util::Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Send + 'static,
        status: StatusCode,
        prefill_logprobs: Option<Value>,
        return_logprob: bool,
        decode_url: Option<String>,
        headers: Option<HeaderMap>,
        prefill: Arc<dyn Worker>,
        decode: Arc<dyn Worker>,
    ) -> Response {
        use crate::core::attach_guards_to_response;

        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

        tokio::spawn(async move {
            futures_util::pin_mut!(stream);
            while let Some(chunk_result) = stream.next().await {
                match chunk_result {
                    Ok(chunk) => {
                        let is_done = memmem::find(&chunk, b"data: [DONE]").is_some();

                        let result = if return_logprob && prefill_logprobs.is_some() {
                            Self::merge_streaming_logprobs(prefill_logprobs.clone(), &chunk)
                                .unwrap_or(chunk)
                        } else {
                            chunk
                        };

                        if tx.send(Ok(result)).is_err() {
                            break;
                        }

                        if is_done {
                            break;
                        }
                    }
                    Err(e) => {
                        if let Some(ref url) = decode_url {
                            error!(decode_url = %url, error = %e, "Stream error from decode server");
                        }
                        let _ = tx.send(Err(format!("Stream error: {}", e)));
                        break;
                    }
                }
            }
        });

        let stream = UnboundedReceiverStream::new(rx);
        let body = Body::from_stream(stream);

        let mut response = Response::new(body);
        *response.status_mut() = status;

        let mut headers = headers.unwrap_or_default();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
        *response.headers_mut() = headers;

        let guards = vec![WorkerLoadGuard::new(prefill), WorkerLoadGuard::new(decode)];
        attach_guards_to_response(guards, response)
    }

    fn build_post_with_headers(
        &self,
        client: &Client,
        url: &str,
        route: &'static str,
        json_request: &Value,
        headers: Option<&HeaderMap>,
        connection_close: bool,
    ) -> reqwest::RequestBuilder {
        let mut request = client.post(api_path(url, route)).json(json_request);
        if connection_close {
            request = request.header("Connection", "close");
        }
        if let Some(headers) = headers {
            for (name, value) in headers.iter() {
                let name_str = name.as_str();
                let forward = name_str.eq_ignore_ascii_case("authorization")
                    || name_str.eq_ignore_ascii_case("x-request-id")
                    || name_str.eq_ignore_ascii_case("x-correlation-id")
                    || name_str.eq_ignore_ascii_case("traceparent")
                    || name_str.eq_ignore_ascii_case("tracestate")
                    || name_str
                        .get(..13)
                        .is_some_and(|prefix| prefix.eq_ignore_ascii_case("x-request-id-"));
                if forward {
                    if let Ok(val) = value.to_str() {
                        request = request.header(name, val);
                    }
                }
            }
        }
        request
    }

    fn merge_streaming_logprobs(
        prefill_logprobs: Option<Value>,
        decode_chunk: &[u8],
    ) -> Result<bytes::Bytes, ()> {
        let chunk_str = std::str::from_utf8(decode_chunk).map_err(|_| ())?;
        if !chunk_str.starts_with("data: ") || chunk_str.contains("[DONE]") {
            return Err(());
        }

        let json_str = chunk_str.trim_start_matches("data: ").trim();
        let mut decode_json: Value = serde_json::from_str(json_str).map_err(|_| ())?;

        if let Some(ref p_logprobs) = prefill_logprobs {
            if let Some(meta) = decode_json.get_mut("meta_info") {
                if let Some(d_logprobs) = meta.get_mut("input_token_logprobs") {
                    if let Some(p_arr) = p_logprobs.as_array() {
                        let decode_arr = std::mem::take(d_logprobs);
                        if let Value::Array(d_vec) = decode_arr {
                            let mut merged = Vec::with_capacity(p_arr.len() + d_vec.len());
                            merged.extend(p_arr.iter().cloned());
                            merged.extend(d_vec);
                            *d_logprobs = Value::Array(merged);
                        }
                    }
                }
            }
        }

        let merged_str = format!(
            "data: {}\n\n",
            serde_json::to_string(&decode_json).unwrap_or_default()
        );
        Ok(bytes::Bytes::from(merged_str))
    }
}

#[async_trait]
impl RouterTrait for VllmPDRouter {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn health_generate(&self, _req: Request<Body>) -> Response {
        let (prefill, decode) = match self.select_pd_pair(None, None, None).await {
            Ok(pair) => pair,
            Err(e) => {
                return error::service_unavailable(
                    "no_healthy_worker_pair",
                    format!("No healthy worker pair available: {}", e),
                );
            }
        };

        let prefill_url = format!("{}/health_generate", prefill.url());
        let (prefill_result, decode_result) = tokio::join!(
            self.client.get(&prefill_url).send(),
            self.client
                .get(format!("{}/health_generate", decode.url()))
                .send()
        );

        let mut errors = Vec::new();

        match prefill_result {
            Ok(res) if res.status().is_success() => {
                debug!(
                    "Health generate passed for prefill server: {}",
                    prefill.url()
                );
            }
            Ok(res) => {
                errors.push(format!(
                    "Prefill {} returned status {}",
                    prefill.url(),
                    res.status()
                ));
            }
            Err(e) => {
                errors.push(format!("Prefill {} error: {}", prefill.url(), e));
            }
        }

        match decode_result {
            Ok(res) if res.status().is_success() => {
                debug!("Health generate passed for decode server: {}", decode.url());
            }
            Ok(res) => {
                errors.push(format!(
                    "Decode {} returned status {}",
                    decode.url(),
                    res.status()
                ));
            }
            Err(e) => {
                errors.push(format!("Decode {} error: {}", decode.url(), e));
            }
        }

        if errors.is_empty() {
            (
                StatusCode::OK,
                format!(
                    "Health generate passed on selected pair: prefill={}, decode={}",
                    prefill.url(),
                    decode.url()
                ),
            )
                .into_response()
        } else {
            error::service_unavailable(
                "health_generate_failed",
                format!("Health generate failed: {:?}", errors),
            )
        }
    }

    async fn get_server_info(&self, _req: Request<Body>) -> Response {
        self.proxy_to_first_prefill_worker("get_server_info", None)
            .await
    }

    async fn get_models(&self, req: Request<Body>) -> Response {
        let headers = header_utils::copy_request_headers(&req);
        self.proxy_to_first_prefill_worker("v1/models", Some(headers))
            .await
    }

    async fn get_model_info(&self, req: Request<Body>) -> Response {
        let headers = header_utils::copy_request_headers(&req);
        self.proxy_to_first_prefill_worker("get_model_info", Some(headers))
            .await
    }

    async fn route_generate(
        &self,
        headers: Option<&HeaderMap>,
        body: &GenerateRequest,
        model_id: Option<&str>,
    ) -> Response {
        let is_stream = body.stream;
        let return_logprob = body.return_logprob.unwrap_or(false);

        let request_text = if self.policies_need_request_text() {
            body.text.as_deref().map(|s| s.to_string())
        } else {
            None
        };

        let context = VllmPDRequestContext {
            route: "/generate",
            is_stream,
            return_logprob,
            request_text,
            model_id,
            headers: headers.cloned(),
        };

        self.execute_dual_dispatch(headers, body, context).await
    }

    async fn route_chat(
        &self,
        headers: Option<&HeaderMap>,
        body: &ChatCompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        let is_stream = body.stream;
        let return_logprob = body.logprobs;

        let request_text = if self.policies_need_request_text() {
            body.messages.first().and_then(|msg| match msg {
                ChatMessage::User { content, .. } => match content {
                    MessageContent::Text(text) => Some(text.clone()),
                    MessageContent::Parts(_) => None,
                },
                ChatMessage::Developer { content, .. } => match content {
                    MessageContent::Text(text) => Some(text.clone()),
                    MessageContent::Parts(_) => None,
                },
                ChatMessage::System { content, .. } => Some(content.to_simple_string()),
                _ => None,
            })
        } else {
            None
        };

        let context = VllmPDRequestContext {
            route: "/v1/chat/completions",
            is_stream,
            return_logprob,
            request_text,
            model_id,
            headers: headers.cloned(),
        };

        self.execute_dual_dispatch(headers, body, context).await
    }

    async fn route_completion(
        &self,
        headers: Option<&HeaderMap>,
        body: &CompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        let is_stream = body.stream;
        let return_logprob = body.logprobs.is_some();

        let request_text = if self.policies_need_request_text() {
            match &body.prompt {
                StringOrArray::String(s) => Some(s.clone()),
                StringOrArray::Array(v) => v.first().map(|s| s.to_string()),
            }
        } else {
            None
        };

        let context = VllmPDRequestContext {
            route: "/v1/completions",
            is_stream,
            return_logprob,
            request_text,
            model_id,
            headers: headers.cloned(),
        };

        self.execute_dual_dispatch(headers, body, context).await
    }

    async fn route_rerank(
        &self,
        headers: Option<&HeaderMap>,
        body: &RerankRequest,
        model_id: Option<&str>,
    ) -> Response {
        let req_text = if self.policies_need_request_text() {
            Some(body.query.clone())
        } else {
            None
        };

        let context = VllmPDRequestContext {
            route: "/v1/rerank",
            is_stream: false,
            return_logprob: false,
            request_text: req_text,
            model_id,
            headers: headers.cloned(),
        };

        self.execute_dual_dispatch(headers, body, context).await
    }

    fn router_type(&self) -> &'static str {
        "vllm_pd"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorkerBuilder, WorkerType};

    fn create_test_vllm_pd_router() -> VllmPDRouter {
        let worker_registry = Arc::new(WorkerRegistry::new());
        let policy_registry =
            Arc::new(PolicyRegistry::new(crate::config::PolicyConfig::RoundRobin));

        VllmPDRouter {
            worker_registry,
            policy_registry,
            client: Client::new(),
            retry_config: RetryConfig::default(),
            api_key: Some("test_api_key".to_string()),
            enable_igw: false,
        }
    }

    fn create_test_worker(url: String, worker_type: WorkerType, healthy: bool) -> Box<dyn Worker> {
        let worker = BasicWorkerBuilder::new(url)
            .worker_type(worker_type)
            .build();
        worker.set_healthy(healthy);
        Box::new(worker)
    }

    #[test]
    fn test_generate_vllm_request_id() {
        let request_id = VllmPDRouter::generate_vllm_request_id(
            "10.0.0.1:8000",
            "10.0.0.2:8000",
        );

        // Format: ___prefill_addr_{addr}___decode_addr_{addr}_{uuid}
        assert!(request_id.starts_with("___prefill_addr_10.0.0.1:8000___decode_addr_10.0.0.2:8000_"));
        assert!(request_id.len() > 60); // Should include UUID (32 chars without dashes)
    }

    #[test]
    fn test_prepare_prefill_request() {
        let original = json!({
            "prompt": "Hello, world!",
            "max_tokens": 100,
            "max_completion_tokens": 100,
            "stream": true,
            "stream_options": {"include_usage": true}
        });

        let prefill_req = VllmPDRouter::prepare_prefill_request(&original);

        // max_tokens should be set to 1
        assert_eq!(prefill_req.get("max_tokens").unwrap(), &json!(1));
        // max_completion_tokens should also be set to 1
        assert_eq!(prefill_req.get("max_completion_tokens").unwrap(), &json!(1));
        // stream should be false for prefill
        assert_eq!(prefill_req.get("stream").unwrap(), &json!(false));
        // stream_options should be removed
        assert!(prefill_req.get("stream_options").is_none());
        // kv_transfer_params should be added with do_remote_decode=true
        let kv_params = prefill_req.get("kv_transfer_params").unwrap();
        assert_eq!(kv_params.get("do_remote_decode").unwrap(), &json!(true));
        assert_eq!(kv_params.get("do_remote_prefill").unwrap(), &json!(false));
        // Original prompt should be preserved
        assert_eq!(prefill_req.get("prompt").unwrap(), &json!("Hello, world!"));
    }

    #[test]
    fn test_prepare_decode_request() {
        let original = json!({
            "prompt": "Hello, world!",
            "max_tokens": 100,
            "stream": true
        });

        let kv_params = json!({
            "do_remote_decode": false,
            "do_remote_prefill": true,
            "remote_engine_id": "engine-123"
        });

        let decode_req = VllmPDRouter::prepare_decode_request(&original, Some(kv_params.clone()));

        // kv_transfer_params should be added from prefill response
        assert_eq!(decode_req.get("kv_transfer_params").unwrap(), &kv_params);
        // Original fields should be preserved
        assert_eq!(decode_req.get("prompt").unwrap(), &json!("Hello, world!"));
        assert_eq!(decode_req.get("max_tokens").unwrap(), &json!(100));
        assert_eq!(decode_req.get("stream").unwrap(), &json!(true));
    }

    #[tokio::test]
    async fn test_select_healthy_prefill_worker() {
        let router = create_test_vllm_pd_router();

        let healthy_worker = create_test_worker(
            "http://healthy".to_string(),
            WorkerType::Prefill {
                bootstrap_port: None,
            },
            true,
        );
        let unhealthy_worker = create_test_worker(
            "http://unhealthy".to_string(),
            WorkerType::Prefill {
                bootstrap_port: None,
            },
            false,
        );
        let decode_worker =
            create_test_worker("http://decode".to_string(), WorkerType::Decode, true);

        router.worker_registry.register(Arc::from(unhealthy_worker));
        router.worker_registry.register(Arc::from(healthy_worker));
        router.worker_registry.register(Arc::from(decode_worker));

        let result = router.select_pd_pair(None, None, None).await;

        assert!(result.is_ok());
        let (prefill, _decode) = result.unwrap();

        assert_eq!(prefill.url(), "http://healthy");
        assert!(prefill.is_healthy());
    }

    #[tokio::test]
    async fn test_empty_worker_lists() {
        let router = create_test_vllm_pd_router();

        let result = router.select_pd_pair(None, None, None).await;

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("No prefill workers available"));
    }

    #[test]
    fn test_router_type() {
        let router = create_test_vllm_pd_router();
        assert_eq!(router.router_type(), "vllm_pd");
    }
}
