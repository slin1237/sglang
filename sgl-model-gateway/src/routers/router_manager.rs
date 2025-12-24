//! Router Manager for coordinating multiple routers and workers
//!
//! Provides centralized management based on enable_igw flag:
//! - Single Router Mode (enable_igw=false): Router owns workers directly
//! - Multi-Router Mode (enable_igw=true): RouterManager coordinates everything

use std::sync::Arc;

use arc_swap::ArcSwap;
use async_trait::async_trait;
use axum::{
    body::Body,
    extract::Request,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use dashmap::DashMap;
use serde_json::Value;
use tracing::{debug, info, warn};

use crate::{
    app_context::AppContext,
    config::RoutingMode,
    core::{ConnectionMode, WorkerRegistry, WorkerType},
    protocols::{
        chat::ChatCompletionRequest,
        classify::ClassifyRequest,
        completion::CompletionRequest,
        embedding::EmbeddingRequest,
        generate::GenerateRequest,
        rerank::RerankRequest,
        responses::{ResponsesGetParams, ResponsesRequest},
    },
    routers::RouterTrait,
    server::ServerConfig,
};

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct RouterId(String);

impl RouterId {
    pub fn new(id: String) -> Self {
        Self(id)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Pre-allocated router IDs to avoid heap allocations in hot path
mod router_ids {
    use super::RouterId;
    use std::sync::LazyLock;

    pub static GRPC_PD: LazyLock<RouterId> = LazyLock::new(|| RouterId::new("grpc-pd".to_string()));
    pub static HTTP_PD: LazyLock<RouterId> = LazyLock::new(|| RouterId::new("http-pd".to_string()));
    pub static GRPC_REGULAR: LazyLock<RouterId> =
        LazyLock::new(|| RouterId::new("grpc-regular".to_string()));
    pub static HTTP_REGULAR: LazyLock<RouterId> =
        LazyLock::new(|| RouterId::new("http-regular".to_string()));
}

/// Worker availability flags for router selection
#[derive(Default, Clone, Copy)]
struct WorkerAvailability {
    has_grpc_pd: bool,
    has_http_pd: bool,
    has_grpc_regular: bool,
    has_http_regular: bool,
}

impl WorkerAvailability {
    /// Check if all possible worker types have been found (early termination)
    #[inline]
    fn all_found(&self) -> bool {
        self.has_grpc_pd && self.has_http_pd && self.has_grpc_regular && self.has_http_regular
    }

    /// Categorize a worker and update availability flags
    #[inline]
    fn categorize(&mut self, is_grpc: bool, is_pd: bool) {
        match (is_grpc, is_pd) {
            (true, true) => self.has_grpc_pd = true,
            (true, false) => self.has_grpc_regular = true,
            (false, true) => self.has_http_pd = true,
            (false, false) => self.has_http_regular = true,
        }
    }

    /// Get the best router ID based on priority: grpc-pd > http-pd > grpc-regular > http-regular
    #[inline]
    fn best_router_id(&self) -> Option<&'static RouterId> {
        if self.has_grpc_pd {
            Some(&router_ids::GRPC_PD)
        } else if self.has_http_pd {
            Some(&router_ids::HTTP_PD)
        } else if self.has_grpc_regular {
            Some(&router_ids::GRPC_REGULAR)
        } else if self.has_http_regular {
            Some(&router_ids::HTTP_REGULAR)
        } else {
            None
        }
    }
}

pub struct RouterManager {
    worker_registry: Arc<WorkerRegistry>,
    routers: Arc<DashMap<RouterId, Arc<dyn RouterTrait>>>,
    routers_snapshot: ArcSwap<Vec<Arc<dyn RouterTrait>>>,
    default_router: Arc<std::sync::RwLock<Option<RouterId>>>,
    enable_igw: bool,
}

impl RouterManager {
    pub fn new(worker_registry: Arc<WorkerRegistry>) -> Self {
        Self {
            worker_registry,
            routers: Arc::new(DashMap::new()),
            routers_snapshot: ArcSwap::from_pointee(Vec::new()),
            default_router: Arc::new(std::sync::RwLock::new(None)),
            enable_igw: false, // Will be set properly in from_config
        }
    }

    pub async fn from_config(
        config: &ServerConfig,
        app_context: &Arc<AppContext>,
    ) -> Result<Arc<Self>, String> {
        use crate::routers::RouterFactory;

        let mut manager = Self::new(app_context.worker_registry.clone());
        manager.enable_igw = config.router_config.enable_igw;
        let manager = Arc::new(manager);

        if config.router_config.enable_igw {
            info!("Initializing RouterManager in multi-router mode (IGW)");

            // In IGW mode, create all router types to support dynamic worker registration
            // with any transport (HTTP/gRPC) and mode (regular/PD)

            // Create HTTP Regular router
            match RouterFactory::create_regular_router(app_context).await {
                Ok(http_regular) => {
                    info!("Created HTTP Regular router");
                    manager.register_router(
                        router_ids::HTTP_REGULAR.clone(),
                        Arc::from(http_regular),
                    );
                }
                Err(e) => {
                    warn!("Failed to create HTTP Regular router: {e}");
                }
            }

            // Create gRPC Regular router
            match RouterFactory::create_grpc_router(app_context).await {
                Ok(grpc_regular) => {
                    info!("Created gRPC Regular router");
                    manager.register_router(
                        router_ids::GRPC_REGULAR.clone(),
                        Arc::from(grpc_regular),
                    );
                }
                Err(e) => {
                    warn!("Failed to create gRPC Regular router: {e}");
                }
            }

            // Create HTTP PD router
            match RouterFactory::create_pd_router(
                None,
                None,
                &config.router_config.policy,
                app_context,
            )
            .await
            {
                Ok(http_pd) => {
                    info!("Created HTTP PD router");
                    manager.register_router(router_ids::HTTP_PD.clone(), Arc::from(http_pd));
                }
                Err(e) => {
                    warn!("Failed to create HTTP PD router: {e}");
                }
            }

            // Create gRPC PD router
            match RouterFactory::create_grpc_pd_router(
                None,
                None,
                &config.router_config.policy,
                app_context,
            )
            .await
            {
                Ok(grpc_pd) => {
                    info!("Created gRPC PD router");
                    manager.register_router(router_ids::GRPC_PD.clone(), Arc::from(grpc_pd));
                }
                Err(e) => {
                    warn!("Failed to create gRPC PD router: {e}");
                }
            }

            info!(
                "RouterManager initialized with {} routers for multi-router mode",
                manager.router_count()
            );
        } else {
            info!("Initializing RouterManager in single-router mode");

            let single_router = Arc::from(RouterFactory::create_router(app_context).await?);
            let router_id = Self::determine_router_id(
                &config.router_config.mode,
                &config.router_config.connection_mode,
            );

            info!("Created single router with ID: {}", router_id.as_str());
            manager.register_router(router_id.clone(), single_router);
            manager.set_default_router(router_id);
        }

        if manager.router_count() == 0 {
            return Err("No routers could be initialized".to_string());
        }

        Ok(manager)
    }

    pub fn determine_router_id(
        routing_mode: &RoutingMode,
        connection_mode: &ConnectionMode,
    ) -> RouterId {
        match (connection_mode, routing_mode) {
            (ConnectionMode::Http, RoutingMode::Regular { .. }) => {
                RouterId::new("http-regular".to_string())
            }
            (ConnectionMode::Http, RoutingMode::PrefillDecode { .. }) => {
                RouterId::new("http-pd".to_string())
            }
            (ConnectionMode::Http, RoutingMode::OpenAI { .. }) => {
                RouterId::new("http-openai".to_string())
            }
            (ConnectionMode::Grpc { .. }, RoutingMode::Regular { .. }) => {
                RouterId::new("grpc-regular".to_string())
            }
            (ConnectionMode::Grpc { .. }, RoutingMode::PrefillDecode { .. }) => {
                RouterId::new("grpc-pd".to_string())
            }
            (ConnectionMode::Grpc { .. }, RoutingMode::OpenAI { .. }) => {
                RouterId::new("grpc-regular".to_string())
            }
        }
    }

    pub fn register_router(&self, id: RouterId, router: Arc<dyn RouterTrait>) {
        self.routers.insert(id.clone(), router);

        // Update the lock-free snapshot for fast per-request iteration
        let new_snapshot: Vec<_> = self.routers.iter().map(|e| e.value().clone()).collect();
        self.routers_snapshot.store(Arc::new(new_snapshot));

        let mut default_router = self.default_router.write().unwrap();
        if default_router.is_none() {
            *default_router = Some(id.clone());
            info!("Set default router to {}", id.as_str());
        }
    }

    pub fn set_default_router(&self, id: RouterId) {
        let mut default_router = self.default_router.write().unwrap();
        *default_router = Some(id);
    }

    pub fn router_count(&self) -> usize {
        self.routers.len()
    }

    /// Resolve model_id for a request, inferring from available workers if not specified
    /// - If model_id is provided, use it directly
    /// - If not provided and only one model exists, use it as implicit default
    /// - If not provided and multiple models exist, return error requiring specification
    /// - If no models exist, return service unavailable error
    fn resolve_model_id(&self, model_id: Option<&str>) -> Result<String, Box<Response>> {
        // If model_id is provided, use it
        if let Some(id) = model_id {
            return Ok(id.to_string());
        }

        // Get all available models from worker registry
        let available_models = self.worker_registry.get_models();

        match available_models.len() {
            0 => Err(Box::new(
                (
                    StatusCode::SERVICE_UNAVAILABLE,
                    "No models available - no workers registered",
                )
                    .into_response(),
            )),
            1 => {
                // Single model: use it as implicit default
                debug!(
                    "Model not specified, using implicit default: {}",
                    available_models[0]
                );
                Ok(available_models[0].clone())
            }
            _ => {
                // Multiple models: require explicit model specification
                Err(Box::new(
                    (
                        StatusCode::BAD_REQUEST,
                        format!(
                            "Model must be specified. Available models: {}",
                            available_models.join(", ")
                        ),
                    )
                        .into_response(),
                ))
            }
        }
    }

    /// Get router for a model based on worker connection mode and type.
    ///
    /// Selection priority: grpc-pd > http-pd > grpc-regular > http-regular
    /// This priority prefers gRPC over HTTP (for performance) and PD over regular
    /// (when PD workers are available for disaggregated inference).
    pub fn get_router_for_model(&self, model_id: &str) -> Option<Arc<dyn RouterTrait>> {
        let workers = self.worker_registry.get_by_model(model_id);

        if !workers.is_empty() {
            let mut availability = WorkerAvailability::default();

            for worker in workers.iter() {
                let is_pd = matches!(
                    worker.worker_type(),
                    WorkerType::Prefill { .. } | WorkerType::Decode
                );
                let is_grpc = matches!(worker.connection_mode(), ConnectionMode::Grpc { .. });
                availability.categorize(is_grpc, is_pd);

                // Early termination: if we found grpc-pd (highest priority), no need to continue
                if availability.has_grpc_pd {
                    break;
                }
            }

            if let Some(router_id) = availability.best_router_id() {
                if let Some(router) = self.routers.get(router_id) {
                    return Some(router.clone());
                }
            }
        }

        // Fallback to default router (handle poisoned lock gracefully)
        self.default_router
            .read()
            .ok()
            .and_then(|guard| guard.as_ref().and_then(|id| self.routers.get(id).map(|r| r.clone())))
    }

    /// Select a router for a request based on headers and model.
    ///
    /// Selection priority: grpc-pd > http-pd > grpc-regular > http-regular
    pub fn select_router_for_request(
        &self,
        _headers: Option<&HeaderMap>,
        model_id: Option<&str>,
    ) -> Option<Arc<dyn RouterTrait>> {
        // In single-router mode (enable_igw=false), always use the default router
        if !self.enable_igw {
            return self
                .default_router
                .read()
                .ok()
                .and_then(|guard| {
                    guard.as_ref().map(|id| {
                        debug!(
                            "Single-router mode: using default router {} for model {:?}",
                            id.as_str(),
                            model_id
                        );
                        id.clone()
                    })
                })
                .and_then(|id| self.routers.get(&id).map(|r| r.clone()));
        }

        // When model_id is provided, use get_router_for_model which checks actual worker availability
        if let Some(model) = model_id {
            if let Some(router) = self.get_router_for_model(model) {
                return Some(router);
            }
        }

        // When no model_id, categorize all healthy workers and select best router
        let workers = self.worker_registry.get_all();
        let mut availability = WorkerAvailability::default();

        for worker in workers.iter() {
            if !worker.is_healthy() {
                continue;
            }
            let is_pd = matches!(
                worker.worker_type(),
                WorkerType::Prefill { .. } | WorkerType::Decode
            );
            let is_grpc = matches!(worker.connection_mode(), ConnectionMode::Grpc { .. });
            availability.categorize(is_grpc, is_pd);

            // Early termination: if all types found, no need to continue
            if availability.all_found() {
                break;
            }
        }

        // Get best router based on priority
        if let Some(router_id) = availability.best_router_id() {
            if let Some(router) = self.routers.get(router_id) {
                return Some(router.clone());
            }
        }

        // Fallback to default router
        self.default_router
            .read()
            .ok()
            .and_then(|guard| guard.as_ref().and_then(|id| self.routers.get(id).map(|r| r.clone())))
    }
}

#[async_trait]
impl RouterTrait for RouterManager {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn health_generate(&self, _req: Request<Body>) -> Response {
        // Check if any worker is healthy - report ready if at least one healthy worker exists
        let workers = self.worker_registry.get_all();
        let healthy_workers: Vec<_> = workers.iter().filter(|w| w.is_healthy()).collect();

        if healthy_workers.is_empty() {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                serde_json::json!({
                    "status": "unhealthy",
                    "reason": "no healthy workers available",
                    "total_workers": workers.len(),
                    "healthy_workers": 0
                })
                .to_string(),
            )
                .into_response()
        } else {
            (
                StatusCode::OK,
                serde_json::json!({
                    "status": "healthy",
                    "total_workers": workers.len(),
                    "healthy_workers": healthy_workers.len(),
                    "routers_count": self.routers.len()
                })
                .to_string(),
            )
                .into_response()
        }
    }

    async fn get_server_info(&self, _req: Request<Body>) -> Response {
        // TODO: Aggregate info from all routers with healthy workers
        (
            StatusCode::OK,
            serde_json::json!({
                "router_manager": true,
                "routers_count": self.routers.len(),
                "workers_count": self.worker_registry.get_all().len()
            })
            .to_string(),
        )
            .into_response()
    }

    async fn get_models(&self, _req: Request<Body>) -> Response {
        let model_names = self.worker_registry.get_models();

        if model_names.is_empty() {
            (StatusCode::SERVICE_UNAVAILABLE, "No models available").into_response()
        } else {
            // Convert model names to OpenAI-compatible model objects
            let models: Vec<Value> = model_names
                .iter()
                .map(|name| {
                    serde_json::json!({
                        "id": name,
                        "object": "model",
                        "owned_by": "local"
                    })
                })
                .collect();

            (
                StatusCode::OK,
                serde_json::json!({
                    "object": "list",
                    "data": models
                })
                .to_string(),
            )
                .into_response()
        }
    }

    async fn get_model_info(&self, req: Request<Body>) -> Response {
        // Route to default router or first available router
        let router = self
            .default_router
            .read()
            .ok()
            .and_then(|guard| guard.as_ref().and_then(|id| self.routers.get(id).map(|r| r.clone())))
            .or_else(|| self.routers.iter().next().map(|r| r.value().clone()));

        if let Some(router) = router {
            router.get_model_info(req).await
        } else {
            (StatusCode::SERVICE_UNAVAILABLE, "No routers available").into_response()
        }
    }

    async fn route_generate(
        &self,
        headers: Option<&HeaderMap>,
        body: &GenerateRequest,
        model_id: Option<&str>,
    ) -> Response {
        // Resolve model_id intelligently instead of falling back to "unknown"
        let resolved_model_id = match self.resolve_model_id(model_id) {
            Ok(id) => id,
            Err(err_response) => return *err_response,
        };

        let router = self.select_router_for_request(headers, Some(&resolved_model_id));

        if let Some(router) = router {
            router
                .route_generate(headers, body, Some(&resolved_model_id))
                .await
        } else {
            (
                StatusCode::NOT_FOUND,
                "No router available for this request",
            )
                .into_response()
        }
    }

    async fn route_chat(
        &self,
        headers: Option<&HeaderMap>,
        body: &ChatCompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        let router = self.select_router_for_request(headers, model_id);

        if let Some(router) = router {
            router.route_chat(headers, body, model_id).await
        } else {
            (
                StatusCode::NOT_FOUND,
                format!("Model '{}' not found or no router available", body.model),
            )
                .into_response()
        }
    }

    async fn route_completion(
        &self,
        headers: Option<&HeaderMap>,
        body: &CompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        let router = self.select_router_for_request(headers, model_id);

        if let Some(router) = router {
            router.route_completion(headers, body, model_id).await
        } else {
            (
                StatusCode::NOT_FOUND,
                format!("Model '{}' not found or no router available", body.model),
            )
                .into_response()
        }
    }

    async fn route_responses(
        &self,
        headers: Option<&HeaderMap>,
        body: &ResponsesRequest,
        model_id: Option<&str>,
    ) -> Response {
        let selected_model = model_id.or(Some(body.model.as_str()));
        let router = self.select_router_for_request(headers, selected_model);

        if let Some(router) = router {
            router.route_responses(headers, body, selected_model).await
        } else {
            (
                StatusCode::NOT_FOUND,
                "No router available to handle responses request",
            )
                .into_response()
        }
    }

    async fn get_response(
        &self,
        headers: Option<&HeaderMap>,
        response_id: &str,
        params: &ResponsesGetParams,
    ) -> Response {
        let router = self.select_router_for_request(headers, None);
        if let Some(router) = router {
            router.get_response(headers, response_id, params).await
        } else {
            (
                StatusCode::NOT_FOUND,
                format!("No router available to get response '{}'", response_id),
            )
                .into_response()
        }
    }

    async fn cancel_response(&self, headers: Option<&HeaderMap>, response_id: &str) -> Response {
        let router = self.select_router_for_request(headers, None);
        if let Some(router) = router {
            router.cancel_response(headers, response_id).await
        } else {
            (
                StatusCode::NOT_FOUND,
                format!("No router available to cancel response '{}'", response_id),
            )
                .into_response()
        }
    }

    async fn delete_response(&self, _headers: Option<&HeaderMap>, _response_id: &str) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "responses api not yet implemented in inference gateway mode",
        )
            .into_response()
    }

    async fn list_response_input_items(
        &self,
        headers: Option<&HeaderMap>,
        response_id: &str,
    ) -> Response {
        // Delegate to the default router (typically http-regular)
        // Response storage is shared across all routers via AppContext
        let router = self.select_router_for_request(headers, None);
        if let Some(router) = router {
            router.list_response_input_items(headers, response_id).await
        } else {
            (
                StatusCode::NOT_FOUND,
                "No router available to list response input items",
            )
                .into_response()
        }
    }

    async fn route_embeddings(
        &self,
        headers: Option<&HeaderMap>,
        body: &EmbeddingRequest,
        model_id: Option<&str>,
    ) -> Response {
        let router = self.select_router_for_request(headers, model_id);

        if let Some(router) = router {
            router.route_embeddings(headers, body, model_id).await
        } else {
            (
                StatusCode::NOT_FOUND,
                format!("Model '{}' not found or no router available", body.model),
            )
                .into_response()
        }
    }

    async fn route_classify(
        &self,
        headers: Option<&HeaderMap>,
        body: &ClassifyRequest,
        model_id: Option<&str>,
    ) -> Response {
        let router = self.select_router_for_request(headers, model_id);

        if let Some(router) = router {
            router.route_classify(headers, body, model_id).await
        } else {
            (
                StatusCode::NOT_FOUND,
                format!("Model '{}' not found or no router available", body.model),
            )
                .into_response()
        }
    }

    async fn route_rerank(
        &self,
        headers: Option<&HeaderMap>,
        body: &RerankRequest,
        model_id: Option<&str>,
    ) -> Response {
        let router = self.select_router_for_request(headers, model_id);

        if let Some(router) = router {
            router.route_rerank(headers, body, model_id).await
        } else {
            (
                StatusCode::NOT_FOUND,
                "No router available for rerank request",
            )
                .into_response()
        }
    }

    fn router_type(&self) -> &'static str {
        "manager"
    }
}

impl std::fmt::Debug for RouterManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let default_router = self
            .default_router
            .read()
            .ok()
            .and_then(|guard| guard.clone());

        f.debug_struct("RouterManager")
            .field("routers_count", &self.routers.len())
            .field("workers_count", &self.worker_registry.get_all().len())
            .field("default_router", &default_router)
            .finish()
    }
}
