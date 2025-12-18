//! Hybrid Cache-Aware Load Balancing Router with optional gRPC backend.
//!
//! This module provides a cache-aware routing policy that can operate in two modes:
//!
//! 1. **Local Mode** (default): Uses the optimized in-process radix tree
//! 2. **gRPC Mode**: Delegates tree operations to a remote gRPC service
//!
//! ## When to Use gRPC Mode
//!
//! Use gRPC mode when:
//! - Running multiple gateway instances that need shared cache state
//! - The radix tree operations are causing CPU bottlenecks (>200k req/s)
//! - You need cache state persistence across gateway restarts
//! - You want to scale the tree service independently
//!
//! ## Configuration
//!
//! ```ignore
//! // Local mode (default)
//! let config = CacheAwareHybridConfig::default();
//!
//! // gRPC mode
//! let config = CacheAwareHybridConfig {
//!     use_grpc: true,
//!     grpc_address: "http://radix-tree-service:50052".to_string(),
//!     ..Default::default()
//! };
//! ```

use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread,
    time::Duration,
};

use dashmap::DashMap;
use rand::Rng;
use tokio::runtime::Handle;
use tracing::{debug, error, warn};

use super::{
    get_healthy_worker_indices, tree_optimized::OptimizedTree, CacheAwareConfig,
    LoadBalancingPolicy,
};
use crate::core::Worker;

#[cfg(feature = "grpc-server")]
use super::radix_tree_grpc::RadixTreeGrpcClient;

/// Configuration for the hybrid cache-aware policy.
#[derive(Debug, Clone)]
pub struct CacheAwareHybridConfig {
    /// Base cache-aware configuration
    pub base: CacheAwareConfig,
    /// Whether to use gRPC backend
    pub use_grpc: bool,
    /// gRPC server address (only used if use_grpc is true)
    pub grpc_address: String,
    /// Connection timeout in milliseconds
    pub grpc_connect_timeout_ms: u64,
    /// Request timeout in milliseconds
    pub grpc_request_timeout_ms: u64,
    /// Fallback to local on gRPC failure
    pub fallback_to_local: bool,
}

impl Default for CacheAwareHybridConfig {
    fn default() -> Self {
        CacheAwareHybridConfig {
            base: CacheAwareConfig::default(),
            use_grpc: false,
            grpc_address: "http://localhost:50052".to_string(),
            grpc_connect_timeout_ms: 5000,
            grpc_request_timeout_ms: 100, // Low timeout for routing decisions
            fallback_to_local: true,
        }
    }
}

/// Tree backend abstraction for local or remote operations.
enum TreeBackend {
    /// Local optimized tree
    Local(Arc<OptimizedTree>),
    /// Remote gRPC client (with local fallback)
    #[cfg(feature = "grpc-server")]
    Remote {
        client: tokio::sync::RwLock<Option<RadixTreeGrpcClient>>,
        address: String,
        fallback: Option<Arc<OptimizedTree>>,
    },
}

impl std::fmt::Debug for TreeBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TreeBackend::Local(_) => write!(f, "TreeBackend::Local"),
            #[cfg(feature = "grpc-server")]
            TreeBackend::Remote { address, .. } => {
                write!(f, "TreeBackend::Remote({})", address)
            }
        }
    }
}

/// Hybrid cache-aware policy supporting local and gRPC modes.
#[derive(Debug)]
pub struct CacheAwareHybridPolicy {
    config: CacheAwareHybridConfig,
    /// Per-model tree backends
    backends: Arc<DashMap<String, TreeBackend>>,
    /// Background eviction thread handle
    eviction_handle: Option<thread::JoinHandle<()>>,
    /// Shutdown flag for eviction thread
    shutdown_flag: Arc<AtomicBool>,
    /// Tokio runtime handle for async operations in sync context
    runtime_handle: Option<Handle>,
}

impl CacheAwareHybridPolicy {
    /// Create a new hybrid policy with default configuration.
    pub fn new() -> Self {
        Self::with_config(CacheAwareHybridConfig::default())
    }

    /// Create a new hybrid policy with the given configuration.
    pub fn with_config(config: CacheAwareHybridConfig) -> Self {
        let backends: Arc<DashMap<String, TreeBackend>> = Arc::new(DashMap::new());
        let shutdown_flag = Arc::new(AtomicBool::new(false));

        // Try to get the current tokio runtime handle
        let runtime_handle = Handle::try_current().ok();

        // Start eviction thread for local mode
        let eviction_handle = if !config.use_grpc && config.base.eviction_interval_secs > 0 {
            let backends_clone = Arc::clone(&backends);
            let shutdown_clone = Arc::clone(&shutdown_flag);
            let max_tree_size = config.base.max_tree_size;
            let interval = config.base.eviction_interval_secs;

            Some(thread::spawn(move || {
                let check_interval_ms = 100;
                let total_sleep_ms = interval * 1000;

                loop {
                    let mut slept_ms = 0u64;
                    while slept_ms < total_sleep_ms {
                        if shutdown_clone.load(Ordering::Relaxed) {
                            return;
                        }
                        thread::sleep(Duration::from_millis(check_interval_ms));
                        slept_ms += check_interval_ms;
                    }

                    if shutdown_clone.load(Ordering::Relaxed) {
                        return;
                    }

                    // Evict from all local backends
                    for backend_ref in backends_clone.iter() {
                        if let TreeBackend::Local(tree) = backend_ref.value() {
                            tree.evict_tenant_by_size(max_tree_size);
                        }
                    }
                }
            }))
        } else {
            None
        };

        CacheAwareHybridPolicy {
            config,
            backends,
            eviction_handle,
            shutdown_flag,
            runtime_handle,
        }
    }

    /// Get or create a backend for the given model.
    fn get_or_create_backend(&self, model_id: &str) -> dashmap::mapref::one::Ref<'_, String, TreeBackend> {
        let model_key = if model_id.is_empty() || model_id == "unknown" {
            "default".to_string()
        } else {
            model_id.to_string()
        };

        // Check if backend exists
        if !self.backends.contains_key(&model_key) {
            let backend = if self.config.use_grpc {
                #[cfg(feature = "grpc-server")]
                {
                    TreeBackend::Remote {
                        client: tokio::sync::RwLock::new(None),
                        address: self.config.grpc_address.clone(),
                        fallback: if self.config.fallback_to_local {
                            Some(Arc::new(OptimizedTree::new()))
                        } else {
                            None
                        },
                    }
                }
                #[cfg(not(feature = "grpc-server"))]
                {
                    warn!("gRPC mode requested but grpc-server feature not enabled, using local");
                    TreeBackend::Local(Arc::new(OptimizedTree::new()))
                }
            } else {
                TreeBackend::Local(Arc::new(OptimizedTree::new()))
            };
            self.backends.insert(model_key.clone(), backend);
        }

        self.backends.get(&model_key).unwrap()
    }

    /// Insert into tree (local or remote).
    fn tree_insert(&self, model_id: &str, text: &str, tenant: &str) {
        let backend = self.get_or_create_backend(model_id);

        match backend.value() {
            TreeBackend::Local(tree) => {
                tree.insert(text, tenant);
            }
            #[cfg(feature = "grpc-server")]
            TreeBackend::Remote {
                client,
                address,
                fallback,
            } => {
                // Try async insert via gRPC
                if let Some(handle) = &self.runtime_handle {
                    let address = address.clone();
                    let model_id = model_id.to_string();
                    let text = text.to_string();
                    let tenant = tenant.to_string();
                    let fallback = fallback.clone();

                    // Spawn async task (fire and forget for insert)
                    handle.spawn(async move {
                        match RadixTreeGrpcClient::connect(&address).await {
                            Ok(mut client) => {
                                if let Err(e) = client.insert(&model_id, &text, &tenant).await {
                                    debug!("gRPC insert failed: {}, using fallback", e);
                                    if let Some(fb) = fallback {
                                        fb.insert(&text, &tenant);
                                    }
                                }
                            }
                            Err(e) => {
                                debug!("gRPC connect failed: {}, using fallback", e);
                                if let Some(fb) = fallback {
                                    fb.insert(&text, &tenant);
                                }
                            }
                        }
                    });
                } else if let Some(fb) = fallback {
                    // No runtime, use fallback
                    fb.insert(text, tenant);
                }
            }
        }
    }

    /// Prefix match (local or remote).
    fn tree_prefix_match(&self, model_id: &str, text: &str) -> (String, String, f32) {
        let backend = self.get_or_create_backend(model_id);

        match backend.value() {
            TreeBackend::Local(tree) => {
                let (matched, tenant) = tree.prefix_match(text);
                let ratio = if text.is_empty() {
                    0.0
                } else {
                    matched.chars().count() as f32 / text.chars().count() as f32
                };
                (matched, tenant, ratio)
            }
            #[cfg(feature = "grpc-server")]
            TreeBackend::Remote {
                address, fallback, ..
            } => {
                // For prefix match, we need synchronous result
                // Try blocking call with timeout
                if let Some(handle) = &self.runtime_handle {
                    let address = address.clone();
                    let model_id = model_id.to_string();
                    let text = text.to_string();
                    let timeout = Duration::from_millis(self.config.grpc_request_timeout_ms);

                    // Use block_on with timeout for synchronous result
                    let result = handle.block_on(async {
                        tokio::time::timeout(timeout, async {
                            match RadixTreeGrpcClient::connect(&address).await {
                                Ok(mut client) => client.prefix_match(&model_id, &text).await.ok(),
                                Err(_) => None,
                            }
                        })
                        .await
                        .ok()
                        .flatten()
                    });

                    if let Some((matched, tenant, ratio)) = result {
                        return (matched, tenant, ratio);
                    }
                }

                // Fallback to local
                if let Some(fb) = fallback {
                    let (matched, tenant) = fb.prefix_match(text);
                    let ratio = if text.is_empty() {
                        0.0
                    } else {
                        matched.chars().count() as f32 / text.chars().count() as f32
                    };
                    (matched, tenant, ratio)
                } else {
                    (String::new(), "empty".to_string(), 0.0)
                }
            }
        }
    }

    /// Remove tenant from tree.
    pub fn remove_tenant(&self, model_id: &str, tenant: &str) {
        if let Some(backend) = self.backends.get(model_id) {
            match backend.value() {
                TreeBackend::Local(tree) => {
                    tree.remove_tenant(tenant);
                }
                #[cfg(feature = "grpc-server")]
                TreeBackend::Remote {
                    address, fallback, ..
                } => {
                    if let Some(handle) = &self.runtime_handle {
                        let address = address.clone();
                        let model_id = model_id.to_string();
                        let tenant = tenant.to_string();
                        let fallback = fallback.clone();

                        handle.spawn(async move {
                            match RadixTreeGrpcClient::connect(&address).await {
                                Ok(mut client) => {
                                    if let Err(e) =
                                        client.remove_tenant(&model_id, &tenant).await
                                    {
                                        error!("gRPC remove_tenant failed: {}", e);
                                    }
                                }
                                Err(e) => {
                                    error!("gRPC connect failed: {}", e);
                                }
                            }
                            if let Some(fb) = fallback {
                                fb.remove_tenant(&tenant);
                            }
                        });
                    } else if let Some(fb) = fallback {
                        fb.remove_tenant(tenant);
                    }
                }
            }
        }
    }

    /// Initialize workers.
    pub fn init_workers(&self, workers: &[Arc<dyn Worker>]) {
        for worker in workers {
            let model_id = worker.model_id();
            let tree_key = if model_id.is_empty() || model_id == "unknown" {
                "default"
            } else {
                model_id
            };
            self.tree_insert(tree_key, "", worker.url());
        }
    }

    /// Add a worker.
    pub fn add_worker(&self, worker: &dyn Worker) {
        let model_id = worker.model_id();
        let tree_key = if model_id.is_empty() || model_id == "unknown" {
            "default"
        } else {
            model_id
        };
        self.tree_insert(tree_key, "", worker.url());
    }

    /// Remove a worker.
    pub fn remove_worker(&self, worker: &dyn Worker) {
        let model_id = worker.model_id();
        let tree_key = if model_id.is_empty() || model_id == "unknown" {
            "default"
        } else {
            model_id
        };
        self.remove_tenant(tree_key, worker.url());
    }
}

impl LoadBalancingPolicy for CacheAwareHybridPolicy {
    fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        request_text: Option<&str>,
    ) -> Option<usize> {
        let healthy_indices = get_healthy_worker_indices(workers);

        if healthy_indices.is_empty() {
            return None;
        }

        // Determine model from first worker
        let first_model = workers[healthy_indices[0]].model_id();
        let model_id = if first_model.is_empty() || first_model == "unknown" {
            "default"
        } else {
            first_model
        };

        // Get load statistics
        let (min_load, max_load) = workers.iter().fold((usize::MAX, 0usize), |(min, max), w| {
            let load = w.load();
            (min.min(load), max.max(load))
        });
        let min_load = if min_load == usize::MAX { 0 } else { min_load };

        // Check for imbalance
        let is_imbalanced = max_load.saturating_sub(min_load) > self.config.base.balance_abs_threshold
            && (max_load as f32) > (min_load as f32 * self.config.base.balance_rel_threshold);

        if is_imbalanced {
            // Use shortest queue when imbalanced
            let min_idx = healthy_indices
                .iter()
                .min_by_key(|&&idx| workers[idx].load())
                .copied()?;

            // Still update tree for cache state
            if let Some(text) = request_text {
                self.tree_insert(model_id, text, workers[min_idx].url());
            }

            workers[min_idx].increment_processed();
            return Some(min_idx);
        }

        // Cache-aware routing when balanced
        let text = request_text.unwrap_or("");
        let (matched_text, matched_worker, match_ratio) = self.tree_prefix_match(model_id, text);

        let selected_url = if match_ratio > self.config.base.cache_threshold {
            matched_worker
        } else {
            // Low match, use worker with smallest tree (most cache space)
            let min_idx = *healthy_indices
                .iter()
                .min_by_key(|&&idx| workers[idx].load())?;
            workers[min_idx].url().to_string()
        };

        // Find worker index
        if let Some(idx) = workers.iter().position(|w| w.url() == selected_url) {
            if workers[idx].is_healthy() {
                self.tree_insert(model_id, text, &selected_url);
                workers[idx].increment_processed();
                return Some(idx);
            }
        }

        // Fallback to first healthy
        healthy_indices.first().copied()
    }

    fn on_request_complete(&self, worker_url: &str, success: bool) {
        if !success {
            debug!(
                "Request to {} completed with success={}",
                worker_url, success
            );
        }
    }

    fn name(&self) -> &'static str {
        if self.config.use_grpc {
            "cache_aware_grpc"
        } else {
            "cache_aware_optimized"
        }
    }

    fn needs_request_text(&self) -> bool {
        true
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Default for CacheAwareHybridPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for CacheAwareHybridPolicy {
    fn drop(&mut self) {
        self.shutdown_flag.store(true, Ordering::Relaxed);

        if let Some(handle) = self.eviction_handle.take() {
            let _ = handle.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = CacheAwareHybridConfig::default();
        assert!(!config.use_grpc);
        assert!(config.fallback_to_local);
    }

    #[test]
    fn test_local_mode_policy() {
        let policy = CacheAwareHybridPolicy::new();
        assert_eq!(policy.name(), "cache_aware_optimized");
        assert!(policy.needs_request_text());
    }

    #[test]
    fn test_grpc_mode_name() {
        let config = CacheAwareHybridConfig {
            use_grpc: true,
            ..Default::default()
        };
        let policy = CacheAwareHybridPolicy::with_config(config);
        // Will be "cache_aware_grpc" if grpc feature enabled, otherwise falls back
        let name = policy.name();
        assert!(name.starts_with("cache_aware"));
    }
}
