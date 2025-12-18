//! gRPC service implementation for the standalone RadixTree service.
//!
//! This module provides both server and client implementations for the RadixTree
//! gRPC service, enabling:
//!
//! - **Horizontal Scaling**: Run radix tree as a separate scalable service
//! - **Shared State**: Multiple gateway instances can share cache state
//! - **Persistence**: Cache state survives gateway restarts
//! - **Offloaded CPU**: Heavy tree operations run on dedicated service
//!
//! ## Usage
//!
//! ### As a standalone server:
//! ```ignore
//! let service = RadixTreeGrpcServer::new();
//! service.serve("0.0.0.0:50051").await?;
//! ```
//!
//! ### As a client in the gateway:
//! ```ignore
//! let client = RadixTreeGrpcClient::connect("http://radix-tree-service:50051").await?;
//! let result = client.prefix_match("model-1", "Hello world").await?;
//! ```

use std::{
    collections::HashMap,
    net::SocketAddr,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::Instant,
};

use dashmap::DashMap;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{transport::Server, Request, Response, Status, Streaming};
use tracing::{debug, error, info};

use super::tree_optimized::OptimizedTree;

// Include generated protobuf code
pub mod proto {
    tonic::include_proto!("sglang.grpc.radixtree");
}

use proto::{
    radix_tree_service_server::{RadixTreeService, RadixTreeServiceServer},
    BatchInsertRequest, BatchInsertResponse, BatchPrefixMatchRequest, BatchPrefixMatchResponse,
    EvictRequest, EvictResponse, GetStatsRequest, GetStatsResponse, HealthCheckRequest,
    HealthCheckResponse, InsertRequest, InsertResponse, ModelStats, PrefixMatchRequest,
    PrefixMatchResponse, PrefixMatchResult, PrefixMatchTenantRequest, PrefixMatchTenantResponse,
    RemoveTenantRequest, RemoveTenantResponse, RouteRequest, RouteResponse, TenantScore,
};

/// gRPC server implementation for the RadixTree service.
#[derive(Debug)]
pub struct RadixTreeGrpcServer {
    /// Per-model trees
    trees: Arc<DashMap<String, Arc<OptimizedTree>>>,
    /// Server start time for uptime tracking
    start_time: Instant,
    /// Total requests processed
    requests_processed: AtomicU64,
}

impl Default for RadixTreeGrpcServer {
    fn default() -> Self {
        Self::new()
    }
}

impl RadixTreeGrpcServer {
    /// Create a new RadixTree gRPC server.
    pub fn new() -> Self {
        RadixTreeGrpcServer {
            trees: Arc::new(DashMap::new()),
            start_time: Instant::now(),
            requests_processed: AtomicU64::new(0),
        }
    }

    /// Get or create a tree for the given model.
    fn get_or_create_tree(&self, model_id: &str) -> Arc<OptimizedTree> {
        let model_key = if model_id.is_empty() || model_id == "unknown" {
            "default".to_string()
        } else {
            model_id.to_string()
        };

        self.trees
            .entry(model_key)
            .or_insert_with(|| Arc::new(OptimizedTree::new()))
            .clone()
    }

    /// Increment request counter.
    fn inc_requests(&self) {
        self.requests_processed.fetch_add(1, Ordering::Relaxed);
    }

    /// Start the gRPC server.
    pub async fn serve(self, addr: &str) -> Result<(), Box<dyn std::error::Error>> {
        let addr: SocketAddr = addr.parse()?;
        info!("RadixTree gRPC server listening on {}", addr);

        Server::builder()
            .add_service(RadixTreeServiceServer::new(self))
            .serve(addr)
            .await?;

        Ok(())
    }
}

#[tonic::async_trait]
impl RadixTreeService for RadixTreeGrpcServer {
    async fn insert(
        &self,
        request: Request<InsertRequest>,
    ) -> Result<Response<InsertResponse>, Status> {
        self.inc_requests();
        let req = request.into_inner();

        let tree = self.get_or_create_tree(&req.model_id);
        let before_count = tree
            .tenant_char_count
            .get(req.tenant_id.as_str())
            .map(|v| *v)
            .unwrap_or(0);

        tree.insert(&req.text, &req.tenant_id);

        let after_count = tree
            .tenant_char_count
            .get(req.tenant_id.as_str())
            .map(|v| *v)
            .unwrap_or(0);

        let chars_added = after_count.saturating_sub(before_count);

        Ok(Response::new(InsertResponse {
            success: true,
            chars_added: chars_added as u64,
        }))
    }

    async fn batch_insert(
        &self,
        request: Request<BatchInsertRequest>,
    ) -> Result<Response<BatchInsertResponse>, Status> {
        self.inc_requests();
        let req = request.into_inner();

        let tree = self.get_or_create_tree(&req.model_id);
        let mut total_chars_added = 0u64;
        let entries_count = req.entries.len() as u32;

        for entry in req.entries {
            let before = tree
                .tenant_char_count
                .get(entry.tenant_id.as_str())
                .map(|v| *v)
                .unwrap_or(0);

            tree.insert(&entry.text, &entry.tenant_id);

            let after = tree
                .tenant_char_count
                .get(entry.tenant_id.as_str())
                .map(|v| *v)
                .unwrap_or(0);

            total_chars_added += after.saturating_sub(before) as u64;
        }

        Ok(Response::new(BatchInsertResponse {
            success: true,
            total_chars_added,
            entries_processed: entries_count,
        }))
    }

    async fn prefix_match(
        &self,
        request: Request<PrefixMatchRequest>,
    ) -> Result<Response<PrefixMatchResponse>, Status> {
        self.inc_requests();
        let req = request.into_inner();

        let tree = self.get_or_create_tree(&req.model_id);
        let (matched_text, tenant_id) = tree.prefix_match(&req.text);

        let match_ratio = if req.text.is_empty() {
            0.0
        } else {
            matched_text.chars().count() as f32 / req.text.chars().count() as f32
        };

        Ok(Response::new(PrefixMatchResponse {
            matched_text,
            tenant_id,
            match_ratio,
        }))
    }

    async fn batch_prefix_match(
        &self,
        request: Request<BatchPrefixMatchRequest>,
    ) -> Result<Response<BatchPrefixMatchResponse>, Status> {
        self.inc_requests();
        let req = request.into_inner();

        let tree = self.get_or_create_tree(&req.model_id);
        let results: Vec<PrefixMatchResult> = req
            .texts
            .iter()
            .map(|text| {
                let (matched_text, tenant_id) = tree.prefix_match(text);
                let match_ratio = if text.is_empty() {
                    0.0
                } else {
                    matched_text.chars().count() as f32 / text.chars().count() as f32
                };
                PrefixMatchResult {
                    matched_text,
                    tenant_id,
                    match_ratio,
                }
            })
            .collect();

        Ok(Response::new(BatchPrefixMatchResponse { results }))
    }

    async fn prefix_match_tenant(
        &self,
        request: Request<PrefixMatchTenantRequest>,
    ) -> Result<Response<PrefixMatchTenantResponse>, Status> {
        self.inc_requests();
        let req = request.into_inner();

        let tree = self.get_or_create_tree(&req.model_id);
        let matched_text = tree.prefix_match_tenant(&req.text, &req.tenant_id);

        let match_ratio = if req.text.is_empty() {
            0.0
        } else {
            matched_text.chars().count() as f32 / req.text.chars().count() as f32
        };

        Ok(Response::new(PrefixMatchTenantResponse {
            matched_text,
            match_ratio,
        }))
    }

    async fn remove_tenant(
        &self,
        request: Request<RemoveTenantRequest>,
    ) -> Result<Response<RemoveTenantResponse>, Status> {
        self.inc_requests();
        let req = request.into_inner();

        let tree = self.get_or_create_tree(&req.model_id);
        let before = tree
            .tenant_char_count
            .get(req.tenant_id.as_str())
            .map(|v| *v)
            .unwrap_or(0);

        tree.remove_tenant(&req.tenant_id);

        Ok(Response::new(RemoveTenantResponse {
            success: true,
            chars_removed: before as u64,
        }))
    }

    async fn evict(
        &self,
        request: Request<EvictRequest>,
    ) -> Result<Response<EvictResponse>, Status> {
        self.inc_requests();
        let req = request.into_inner();

        let tree = self.get_or_create_tree(&req.model_id);

        let before_counts: HashMap<String, usize> = tree.get_tenant_char_count();

        tree.evict_tenant_by_size(req.max_size_per_tenant as usize);

        let after_counts = tree.get_tenant_char_count();

        let chars_evicted: HashMap<String, u64> = before_counts
            .iter()
            .map(|(tenant, &before)| {
                let after = after_counts.get(tenant).copied().unwrap_or(0);
                (tenant.clone(), before.saturating_sub(after) as u64)
            })
            .filter(|(_, evicted)| *evicted > 0)
            .collect();

        Ok(Response::new(EvictResponse {
            success: true,
            chars_evicted,
        }))
    }

    async fn get_stats(
        &self,
        request: Request<GetStatsRequest>,
    ) -> Result<Response<GetStatsResponse>, Status> {
        self.inc_requests();
        let req = request.into_inner();

        let mut total_nodes = 0u64;
        let mut total_tenants = 0u64;
        let mut total_chars = 0u64;
        let mut tenant_char_counts: HashMap<String, u64> = HashMap::new();
        let mut model_stats: HashMap<String, ModelStats> = HashMap::new();

        let filter_model = req.model_id;

        for tree_ref in self.trees.iter() {
            let model_id = tree_ref.key().clone();

            // Skip if filtering and model doesn't match
            if let Some(ref filter) = filter_model {
                if &model_id != filter {
                    continue;
                }
            }

            let tree = tree_ref.value();
            let stats = tree.stats();

            total_nodes += stats.node_count as u64;
            total_tenants += stats.tenant_count as u64;
            total_chars += stats.total_chars as u64;

            for (tenant, count) in tree.get_tenant_char_count() {
                *tenant_char_counts.entry(tenant).or_insert(0) += count as u64;
            }

            model_stats.insert(
                model_id,
                ModelStats {
                    node_count: stats.node_count as u64,
                    tenant_count: stats.tenant_count as u64,
                    total_chars: stats.total_chars as u64,
                },
            );
        }

        Ok(Response::new(GetStatsResponse {
            total_nodes,
            total_tenants,
            total_chars,
            tenant_char_counts,
            model_stats,
        }))
    }

    async fn health_check(
        &self,
        _request: Request<HealthCheckRequest>,
    ) -> Result<Response<HealthCheckResponse>, Status> {
        let uptime = self.start_time.elapsed().as_secs();
        let requests = self.requests_processed.load(Ordering::Relaxed);

        Ok(Response::new(HealthCheckResponse {
            healthy: true,
            message: "RadixTree service is healthy".to_string(),
            uptime_secs: uptime,
            requests_processed: requests,
        }))
    }

    type RouteRequestsStream = ReceiverStream<Result<RouteResponse, Status>>;

    async fn route_requests(
        &self,
        request: Request<Streaming<RouteRequest>>,
    ) -> Result<Response<Self::RouteRequestsStream>, Status> {
        let mut stream = request.into_inner();
        let trees = Arc::clone(&self.trees);
        let (tx, rx) = mpsc::channel(128);

        tokio::spawn(async move {
            while let Ok(Some(req)) = stream.message().await {
                let model_key = if req.model_id.is_empty() || req.model_id == "unknown" {
                    "default".to_string()
                } else {
                    req.model_id.clone()
                };

                let response = if let Some(tree_ref) = trees.get(&model_key) {
                    let tree = tree_ref.value();

                    // If candidate tenants provided, match against each
                    if !req.candidate_tenants.is_empty() {
                        let mut best_tenant = String::new();
                        let mut best_score = 0.0f32;
                        let mut alternatives = Vec::new();

                        for tenant in &req.candidate_tenants {
                            let matched = tree.prefix_match_tenant(&req.text, tenant);
                            let score = if req.text.is_empty() {
                                0.0
                            } else {
                                matched.chars().count() as f32 / req.text.chars().count() as f32
                            };

                            if score > best_score {
                                if !best_tenant.is_empty() {
                                    alternatives.push(TenantScore {
                                        tenant_id: best_tenant.clone(),
                                        score: best_score,
                                    });
                                }
                                best_tenant = tenant.clone();
                                best_score = score;
                            } else {
                                alternatives.push(TenantScore {
                                    tenant_id: tenant.clone(),
                                    score,
                                });
                            }
                        }

                        RouteResponse {
                            request_id: req.request_id,
                            recommended_tenant: best_tenant,
                            match_score: best_score,
                            alternatives,
                        }
                    } else {
                        // No candidates, use global match
                        let (matched, tenant) = tree.prefix_match(&req.text);
                        let score = if req.text.is_empty() {
                            0.0
                        } else {
                            matched.chars().count() as f32 / req.text.chars().count() as f32
                        };

                        RouteResponse {
                            request_id: req.request_id,
                            recommended_tenant: tenant,
                            match_score: score,
                            alternatives: vec![],
                        }
                    }
                } else {
                    // No tree for model, return empty
                    RouteResponse {
                        request_id: req.request_id,
                        recommended_tenant: String::new(),
                        match_score: 0.0,
                        alternatives: vec![],
                    }
                };

                if tx.send(Ok(response)).await.is_err() {
                    debug!("Client disconnected from route_requests stream");
                    break;
                }
            }
        });

        Ok(Response::new(ReceiverStream::new(rx)))
    }
}

// =====================
// Client Implementation
// =====================

use proto::radix_tree_service_client::RadixTreeServiceClient;
use tonic::transport::Channel;

/// gRPC client for connecting to a remote RadixTree service.
#[derive(Debug, Clone)]
pub struct RadixTreeGrpcClient {
    client: RadixTreeServiceClient<Channel>,
}

impl RadixTreeGrpcClient {
    /// Connect to a RadixTree gRPC server.
    pub async fn connect(addr: &str) -> Result<Self, tonic::transport::Error> {
        let client = RadixTreeServiceClient::connect(addr.to_string()).await?;
        Ok(RadixTreeGrpcClient { client })
    }

    /// Insert text for a tenant.
    pub async fn insert(
        &mut self,
        model_id: &str,
        text: &str,
        tenant_id: &str,
    ) -> Result<u64, Status> {
        let request = InsertRequest {
            model_id: model_id.to_string(),
            text: text.to_string(),
            tenant_id: tenant_id.to_string(),
        };
        let response = self.client.insert(request).await?;
        Ok(response.into_inner().chars_added)
    }

    /// Find the best prefix match.
    pub async fn prefix_match(
        &mut self,
        model_id: &str,
        text: &str,
    ) -> Result<(String, String, f32), Status> {
        let request = PrefixMatchRequest {
            model_id: model_id.to_string(),
            text: text.to_string(),
        };
        let response = self.client.prefix_match(request).await?;
        let inner = response.into_inner();
        Ok((inner.matched_text, inner.tenant_id, inner.match_ratio))
    }

    /// Batch prefix match for high throughput.
    pub async fn batch_prefix_match(
        &mut self,
        model_id: &str,
        texts: Vec<String>,
    ) -> Result<Vec<(String, String, f32)>, Status> {
        let request = BatchPrefixMatchRequest {
            model_id: model_id.to_string(),
            texts,
        };
        let response = self.client.batch_prefix_match(request).await?;
        Ok(response
            .into_inner()
            .results
            .into_iter()
            .map(|r| (r.matched_text, r.tenant_id, r.match_ratio))
            .collect())
    }

    /// Remove a tenant from the tree.
    pub async fn remove_tenant(
        &mut self,
        model_id: &str,
        tenant_id: &str,
    ) -> Result<u64, Status> {
        let request = RemoveTenantRequest {
            model_id: model_id.to_string(),
            tenant_id: tenant_id.to_string(),
        };
        let response = self.client.remove_tenant(request).await?;
        Ok(response.into_inner().chars_removed)
    }

    /// Trigger eviction.
    pub async fn evict(
        &mut self,
        model_id: &str,
        max_size_per_tenant: u64,
    ) -> Result<HashMap<String, u64>, Status> {
        let request = EvictRequest {
            model_id: model_id.to_string(),
            max_size_per_tenant,
        };
        let response = self.client.evict(request).await?;
        Ok(response.into_inner().chars_evicted)
    }

    /// Get statistics.
    pub async fn get_stats(
        &mut self,
        model_id: Option<&str>,
    ) -> Result<GetStatsResponse, Status> {
        let request = GetStatsRequest {
            model_id: model_id.map(|s| s.to_string()),
        };
        let response = self.client.get_stats(request).await?;
        Ok(response.into_inner())
    }

    /// Health check.
    pub async fn health_check(&mut self) -> Result<HealthCheckResponse, Status> {
        let request = HealthCheckRequest {};
        let response = self.client.health_check(request).await?;
        Ok(response.into_inner())
    }
}

// =====================
// Configuration
// =====================

/// Configuration for the RadixTree gRPC service.
#[derive(Debug, Clone)]
pub struct RadixTreeGrpcConfig {
    /// Whether to use gRPC mode (remote service) or local mode
    pub enabled: bool,
    /// Server address (for client) or bind address (for server)
    pub address: String,
    /// Connection timeout in milliseconds
    pub connect_timeout_ms: u64,
    /// Request timeout in milliseconds
    pub request_timeout_ms: u64,
    /// Maximum retry attempts for failed requests
    pub max_retries: u32,
}

impl Default for RadixTreeGrpcConfig {
    fn default() -> Self {
        RadixTreeGrpcConfig {
            enabled: false,
            address: "http://localhost:50052".to_string(),
            connect_timeout_ms: 5000,
            request_timeout_ms: 1000,
            max_retries: 3,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_server_creation() {
        let server = RadixTreeGrpcServer::new();
        assert_eq!(server.trees.len(), 0);
    }

    #[test]
    fn test_get_or_create_tree() {
        let server = RadixTreeGrpcServer::new();

        // Create tree for model1
        let tree1 = server.get_or_create_tree("model1");
        assert_eq!(server.trees.len(), 1);

        // Get same tree
        let tree1_again = server.get_or_create_tree("model1");
        assert_eq!(server.trees.len(), 1);
        assert!(Arc::ptr_eq(&tree1, &tree1_again));

        // Create tree for model2
        let _tree2 = server.get_or_create_tree("model2");
        assert_eq!(server.trees.len(), 2);

        // Empty model id should use "default"
        let tree_default = server.get_or_create_tree("");
        assert_eq!(server.trees.len(), 3);
        assert!(server.trees.contains_key("default"));
    }

    #[tokio::test]
    async fn test_config_defaults() {
        let config = RadixTreeGrpcConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.address, "http://localhost:50052");
        assert_eq!(config.connect_timeout_ms, 5000);
        assert_eq!(config.request_timeout_ms, 1000);
        assert_eq!(config.max_retries, 3);
    }
}
