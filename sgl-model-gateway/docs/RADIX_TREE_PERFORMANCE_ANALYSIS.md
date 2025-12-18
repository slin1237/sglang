# Radix Tree Performance Analysis

## Executive Summary

The cache-aware load balancing radix tree (`tree.rs`) exhibits significant CPU usage under high load (~200k requests). This document analyzes the root causes and proposes solutions, including a standalone gRPC service architecture.

## Architecture Overview

```
                    ┌─────────────────────────────────┐
                    │     CacheAwarePolicy            │
                    │  ┌───────────────────────────┐  │
                    │  │   DashMap<Model, Tree>    │  │
                    │  └───────────────────────────┘  │
                    └─────────────────────────────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │          Tree (Radix)          │
                    │  ┌─────────────────────────┐  │
                    │  │   root: Arc<Node>       │  │
                    │  │   tenant_char_count     │  │
                    │  └─────────────────────────┘  │
                    └───────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
  ┌─────▼─────┐              ┌─────▼─────┐              ┌─────▼─────┐
  │   Node    │              │   Node    │              │   Node    │
  │ ┌───────┐ │              │ ┌───────┐ │              │ ┌───────┐ │
  │ │RwLock │ │──children──▶ │ │RwLock │ │──children──▶ │ │RwLock │ │
  │ │ text  │ │              │ │ text  │ │              │ │ text  │ │
  │ └───────┘ │              │ └───────┘ │              │ └───────┘ │
  │ ┌───────┐ │              │ ┌───────┐ │              │ ┌───────┐ │
  │ │DashMap│ │              │ │DashMap│ │              │ │DashMap│ │
  │ │tenant │ │              │ │tenant │ │              │ │tenant │ │
  │ │ times │ │              │ │ times │ │              │ │ times │ │
  │ └───────┘ │              │ └───────┘ │              │ └───────┘ │
  └───────────┘              └───────────┘              └───────────┘
```

## Identified Performance Bottlenecks

### 1. **RwLock on Node Text (Critical)**

**Location:** `tree.rs:207`
```rust
text: RwLock<NodeText>,
```

**Problem:** Every `insert()` and `prefix_match()` operation acquires a read lock on each node's text during traversal. Under 200k concurrent requests:
- Read locks contend when multiple threads traverse similar paths
- Write locks during node splits block ALL readers on that node
- Lock acquisition overhead accumulates: O(path_length) locks per operation

**Impact:** At 200k requests with average path depth of 10, that's 2M lock acquisitions just for reads.

### 2. **Timestamp Syscall on Every Operation (Critical)**

**Location:** `tree.rs:182-199`
```rust
fn get_timestamp_ms() -> u128 {
    let cached = CURRENT_TIMESTAMP_MS.load(Ordering::Relaxed);
    // STILL MAKES SYSCALL EVERY TIME!
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;
    // ...
}
```

**Problem:** Despite the "optimization" comment, every call to `get_timestamp_ms()` makes a syscall to get the current time. The "caching" only avoids the atomic store when within 5ms, but the expensive `SystemTime::now()` call happens unconditionally.

**Impact:** 200k syscalls for timestamps alone, plus context switches.

### 3. **Parent Pointer Traversal for Timestamp Updates (High)**

**Location:** `tree.rs:532-538`
```rust
let mut current_node = Some(curr);
while let Some(node) = current_node {
    node.tenant_last_access_time.insert(Arc::clone(tenant_id), timestamp_ms);
    current_node = node.parent.read().unwrap().clone();
}
```

**Problem:** On every `prefix_match()`, timestamps propagate from matched node up to root:
- O(tree_depth) RwLock acquisitions on parent pointers
- O(tree_depth) DashMap insertions for timestamps
- Arc cloning at each level

**Impact:** For a tree depth of 20 and 200k matches, that's 4M additional lock acquisitions and hashmap insertions.

### 4. **Vec<char> Allocation per Operation (Medium)**

**Location:** `tree.rs:64-67`
```rust
fn new(text: &str) -> Self {
    Self {
        chars: text.chars().collect(),  // Allocates new Vec every time
    }
}
```

**Problem:** Every `insert()` and `prefix_match()` creates a `CharIndexedText`, which allocates a `Vec<char>`. For a 1000-character prompt, that's 4KB allocated and deallocated per operation.

**Impact:** 200k allocations/deallocations, memory allocator pressure, potential false sharing.

### 5. **Arc Cloning During Traversal (Medium)**

**Locations:** Throughout traversal code
```rust
curr = Arc::clone(&self.root);
prev = Arc::clone(&matched_node);
```

**Problem:** Each `Arc::clone()` performs an atomic increment. Under high contention, this causes cache-line bouncing between CPU cores.

**Impact:** Millions of atomic operations with potential cache coherency traffic.

### 6. **O(n) Eviction DFS (Low - but periodic)**

**Location:** `tree.rs:655-735`

**Problem:** Eviction traverses the entire tree via DFS, collecting all leaves into a priority queue. With 200k nodes, this is O(n) with significant allocation.

**Impact:** Periodic latency spikes every 30 seconds during eviction.

## Quantified Impact Analysis

| Bottleneck | Operations per Request | At 200k Requests |
|------------|----------------------|------------------|
| RwLock text reads | ~10 (avg path) | 2,000,000 |
| Timestamp syscalls | 1-2 | 200,000-400,000 |
| Parent traversal | ~10 (match) | 2,000,000 |
| Vec<char> allocations | 1-2 | 200,000-400,000 |
| Arc clone atomics | ~20 | 4,000,000 |

## Proposed Optimizations

### Phase 1: Quick Wins (Minimal Code Changes)

#### 1.1 True Timestamp Caching
```rust
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

static CACHED_TIMESTAMP: AtomicU64 = AtomicU64::new(0);
static TIMESTAMP_BASE: OnceLock<Instant> = OnceLock::new();

// Background thread updates this every 1-5ms
fn start_timestamp_updater() {
    std::thread::spawn(|| {
        let base = TIMESTAMP_BASE.get_or_init(Instant::now);
        loop {
            let elapsed = base.elapsed().as_millis() as u64;
            CACHED_TIMESTAMP.store(elapsed, Ordering::Relaxed);
            std::thread::sleep(Duration::from_millis(1));
        }
    });
}

#[inline(always)]
fn get_timestamp_ms() -> u128 {
    CACHED_TIMESTAMP.load(Ordering::Relaxed) as u128
}
```

**Expected improvement:** Eliminates syscall overhead (~5-10% CPU reduction)

#### 1.2 Lazy Timestamp Updates
```rust
// Only update timestamps probabilistically or on cache hits
fn prefix_match(&self, text: &str) -> (String, String) {
    // ... match logic ...

    // Probabilistic timestamp update (1 in 10 requests)
    if rand::random::<u8>() < 25 {  // ~10% probability
        self.update_timestamps_to_root(curr, &tenant_id);
    }

    // Or batch: collect and update in background
}
```

**Expected improvement:** 90% reduction in timestamp propagation overhead

#### 1.3 Thread-Local CharIndexedText Pool
```rust
thread_local! {
    static CHAR_BUFFER: RefCell<Vec<char>> = RefCell::new(Vec::with_capacity(4096));
}

fn with_indexed_text<R>(text: &str, f: impl FnOnce(&[char]) -> R) -> R {
    CHAR_BUFFER.with(|buf| {
        let mut buf = buf.borrow_mut();
        buf.clear();
        buf.extend(text.chars());
        f(&buf)
    })
}
```

**Expected improvement:** Eliminates per-operation allocation (~10-15% reduction)

### Phase 2: Architectural Improvements

#### 2.1 Immutable Node Text with Copy-on-Write
```rust
struct Node {
    // Instead of RwLock, use immutable Arc<str>
    text: Arc<str>,  // Immutable - no lock needed for reads
    char_count: usize,  // Cached, immutable

    children: DashMap<char, Arc<Node>, CharHasherBuilder>,
    tenant_last_access_time: DashMap<TenantId, u128>,
    parent: AtomicPtr<Node>,  // Lock-free parent pointer
}

// For splits, create new nodes (copy-on-write)
fn split_node(&self, node: &Arc<Node>, split_idx: usize) -> Arc<Node> {
    let (prefix, suffix) = node.text.split_at(split_idx);
    // Create new nodes with new immutable text
    // ...
}
```

**Expected improvement:** Eliminates RwLock contention on reads (~20-30% improvement)

#### 2.2 Separate Read/Write Trees (Double Buffering)
```rust
struct Tree {
    // Two trees: one for reads, one for writes
    read_tree: Arc<RwLock<Arc<TreeInner>>>,
    write_tree: Mutex<TreeInner>,
}

impl Tree {
    fn prefix_match(&self, text: &str) -> (String, String) {
        // Lock-free read path
        let tree = self.read_tree.read().unwrap().clone();
        tree.prefix_match_inner(text)
    }

    fn insert(&self, text: &str, tenant: &str) {
        let mut write = self.write_tree.lock().unwrap();
        write.insert_inner(text, tenant);
        // Periodically swap trees (every N inserts or time interval)
    }
}
```

#### 2.3 Sharded Trees by Prefix
```rust
struct ShardedTree {
    shards: [Arc<Tree>; 256],  // Shard by first byte
}

impl ShardedTree {
    fn get_shard(&self, text: &str) -> &Arc<Tree> {
        let idx = text.bytes().next().unwrap_or(0) as usize;
        &self.shards[idx]
    }
}
```

**Expected improvement:** Reduces contention by 256x for unrelated prefixes

### Phase 3: Standalone gRPC Service

For ultimate scalability, extract the radix tree into a dedicated service.

## Standalone gRPC Service Design

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Model Gateway                            │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              CacheAwarePolicy (client)               │   │
│  │  ┌────────────────────────────────────────────────┐  │   │
│  │  │  Local Cache (LRU, TTL: 100ms)                 │  │   │
│  │  │  - Caches prefix_match results                 │  │   │
│  │  │  - Reduces RPC calls for hot paths             │  │   │
│  │  └────────────────────────────────────────────────┘  │   │
│  │                         │                             │   │
│  │                   gRPC (batch)                        │   │
│  │                         │                             │   │
│  └─────────────────────────┼────────────────────────────┘   │
└─────────────────────────────┼───────────────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │   Load Balancer   │
                    └─────────┬─────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
  ┌─────▼─────┐        ┌─────▼─────┐        ┌─────▼─────┐
  │  Tree     │        │  Tree     │        │  Tree     │
  │  Service  │        │  Service  │        │  Service  │
  │  (shard 0)│        │  (shard 1)│        │  (shard 2)│
  └───────────┘        └───────────┘        └───────────┘
```

### Protocol Buffer Definition

```protobuf
syntax = "proto3";

package sglang.cache;

service CacheTreeService {
    // Single operations
    rpc Insert(InsertRequest) returns (InsertResponse);
    rpc PrefixMatch(PrefixMatchRequest) returns (PrefixMatchResponse);
    rpc PrefixMatchTenant(PrefixMatchTenantRequest) returns (PrefixMatchResponse);
    rpc RemoveTenant(RemoveTenantRequest) returns (RemoveTenantResponse);

    // Batch operations (for efficiency)
    rpc BatchInsert(BatchInsertRequest) returns (BatchInsertResponse);
    rpc BatchPrefixMatch(BatchPrefixMatchRequest) returns (BatchPrefixMatchResponse);

    // Streaming for high-throughput
    rpc StreamInsert(stream InsertRequest) returns (StreamInsertResponse);
    rpc StreamPrefixMatch(stream PrefixMatchRequest) returns (stream PrefixMatchResponse);

    // Admin operations
    rpc GetStats(GetStatsRequest) returns (GetStatsResponse);
    rpc Evict(EvictRequest) returns (EvictResponse);
    rpc HealthCheck(HealthCheckRequest) returns (HealthCheckResponse);
}

message InsertRequest {
    string model_id = 1;
    string text = 2;
    string tenant_id = 3;
    int64 timestamp_ms = 4;  // Optional: client-provided timestamp
}

message InsertResponse {
    bool success = 1;
    int64 new_size = 2;  // Current tree size for tenant
}

message PrefixMatchRequest {
    string model_id = 1;
    string text = 2;
    optional string tenant_id = 3;  // If set, tenant-specific match
}

message PrefixMatchResponse {
    string matched_text = 1;
    string matched_tenant = 2;
    float match_ratio = 3;  // matched_len / total_len
}

message BatchInsertRequest {
    repeated InsertRequest requests = 1;
}

message BatchInsertResponse {
    repeated InsertResponse responses = 1;
}

message BatchPrefixMatchRequest {
    repeated PrefixMatchRequest requests = 1;
}

message BatchPrefixMatchResponse {
    repeated PrefixMatchResponse responses = 1;
}

message GetStatsRequest {
    optional string model_id = 1;
}

message GetStatsResponse {
    map<string, ModelStats> model_stats = 1;
}

message ModelStats {
    int64 total_nodes = 1;
    int64 total_chars = 2;
    map<string, int64> tenant_char_counts = 3;
    int64 inserts_per_sec = 4;
    int64 matches_per_sec = 5;
    double avg_match_latency_us = 6;
}

message EvictRequest {
    string model_id = 1;
    int64 max_size = 2;
}

message EvictResponse {
    int64 evicted_count = 1;
    int64 new_size = 2;
}

message RemoveTenantRequest {
    string model_id = 1;
    string tenant_id = 2;
}

message RemoveTenantResponse {
    bool success = 1;
}

message HealthCheckRequest {}

message HealthCheckResponse {
    bool healthy = 1;
    string version = 2;
    int64 uptime_seconds = 3;
}
```

### Service Implementation Skeleton

```rust
// src/bin/cache_tree_service.rs

use tonic::{transport::Server, Request, Response, Status};
use dashmap::DashMap;
use std::sync::Arc;

mod proto {
    tonic::include_proto!("sglang.cache");
}

use proto::cache_tree_service_server::{CacheTreeService, CacheTreeServiceServer};
use proto::*;

pub struct CacheTreeServiceImpl {
    trees: Arc<DashMap<String, Arc<Tree>>>,
    metrics: Arc<Metrics>,
}

#[tonic::async_trait]
impl CacheTreeService for CacheTreeServiceImpl {
    async fn insert(
        &self,
        request: Request<InsertRequest>,
    ) -> Result<Response<InsertResponse>, Status> {
        let req = request.into_inner();
        let tree = self.get_or_create_tree(&req.model_id);

        tree.insert(&req.text, &req.tenant_id);

        let new_size = tree.tenant_char_count
            .get(&req.tenant_id)
            .map(|v| *v as i64)
            .unwrap_or(0);

        self.metrics.record_insert();

        Ok(Response::new(InsertResponse {
            success: true,
            new_size,
        }))
    }

    async fn prefix_match(
        &self,
        request: Request<PrefixMatchRequest>,
    ) -> Result<Response<PrefixMatchResponse>, Status> {
        let start = std::time::Instant::now();
        let req = request.into_inner();

        let tree = self.trees.get(&req.model_id)
            .ok_or_else(|| Status::not_found("Model tree not found"))?;

        let (matched_text, matched_tenant) = if let Some(tenant) = req.tenant_id {
            let matched = tree.prefix_match_tenant(&req.text, &tenant);
            (matched, tenant)
        } else {
            tree.prefix_match(&req.text)
        };

        let match_ratio = if req.text.is_empty() {
            0.0
        } else {
            matched_text.chars().count() as f32 / req.text.chars().count() as f32
        };

        self.metrics.record_match(start.elapsed());

        Ok(Response::new(PrefixMatchResponse {
            matched_text,
            matched_tenant,
            match_ratio,
        }))
    }

    async fn batch_insert(
        &self,
        request: Request<BatchInsertRequest>,
    ) -> Result<Response<BatchInsertResponse>, Status> {
        let requests = request.into_inner().requests;
        let mut responses = Vec::with_capacity(requests.len());

        for req in requests {
            let tree = self.get_or_create_tree(&req.model_id);
            tree.insert(&req.text, &req.tenant_id);

            let new_size = tree.tenant_char_count
                .get(&req.tenant_id)
                .map(|v| *v as i64)
                .unwrap_or(0);

            responses.push(InsertResponse {
                success: true,
                new_size,
            });
        }

        self.metrics.record_batch_insert(responses.len());

        Ok(Response::new(BatchInsertResponse { responses }))
    }

    async fn batch_prefix_match(
        &self,
        request: Request<BatchPrefixMatchRequest>,
    ) -> Result<Response<BatchPrefixMatchResponse>, Status> {
        let start = std::time::Instant::now();
        let requests = request.into_inner().requests;
        let mut responses = Vec::with_capacity(requests.len());

        for req in requests {
            let response = if let Some(tree) = self.trees.get(&req.model_id) {
                let (matched_text, matched_tenant) = if let Some(tenant) = req.tenant_id {
                    let matched = tree.prefix_match_tenant(&req.text, &tenant);
                    (matched, tenant)
                } else {
                    tree.prefix_match(&req.text)
                };

                let match_ratio = if req.text.is_empty() {
                    0.0
                } else {
                    matched_text.chars().count() as f32 / req.text.chars().count() as f32
                };

                PrefixMatchResponse {
                    matched_text,
                    matched_tenant,
                    match_ratio,
                }
            } else {
                PrefixMatchResponse {
                    matched_text: String::new(),
                    matched_tenant: "empty".to_string(),
                    match_ratio: 0.0,
                }
            };

            responses.push(response);
        }

        self.metrics.record_batch_match(responses.len(), start.elapsed());

        Ok(Response::new(BatchPrefixMatchResponse { responses }))
    }

    // ... other methods
}

impl CacheTreeServiceImpl {
    fn get_or_create_tree(&self, model_id: &str) -> Arc<Tree> {
        self.trees
            .entry(model_id.to_string())
            .or_insert_with(|| Arc::new(Tree::new()))
            .clone()
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr = "[::]:50051".parse()?;

    let service = CacheTreeServiceImpl {
        trees: Arc::new(DashMap::new()),
        metrics: Arc::new(Metrics::new()),
    };

    println!("CacheTreeService listening on {}", addr);

    Server::builder()
        .add_service(CacheTreeServiceServer::new(service))
        .serve(addr)
        .await?;

    Ok(())
}
```

### Client Integration

```rust
// In CacheAwarePolicy, replace direct Tree usage with gRPC client

use tokio::sync::RwLock;
use lru::LruCache;

pub struct CacheAwarePolicy {
    client: CacheTreeClient<tonic::transport::Channel>,
    local_cache: RwLock<LruCache<String, (String, String, Instant)>>,
    cache_ttl: Duration,
}

impl CacheAwarePolicy {
    async fn prefix_match(&self, model_id: &str, text: &str) -> (String, String) {
        // Check local cache first
        let cache_key = format!("{}:{}", model_id, &text[..text.len().min(100)]);

        {
            let cache = self.local_cache.read().await;
            if let Some((matched, tenant, time)) = cache.get(&cache_key) {
                if time.elapsed() < self.cache_ttl {
                    return (matched.clone(), tenant.clone());
                }
            }
        }

        // RPC call
        let response = self.client
            .prefix_match(PrefixMatchRequest {
                model_id: model_id.to_string(),
                text: text.to_string(),
                tenant_id: None,
            })
            .await
            .unwrap()
            .into_inner();

        // Update local cache
        {
            let mut cache = self.local_cache.write().await;
            cache.put(
                cache_key,
                (response.matched_text.clone(), response.matched_tenant.clone(), Instant::now())
            );
        }

        (response.matched_text, response.matched_tenant)
    }

    async fn insert_async(&self, model_id: &str, text: &str, tenant: &str) {
        // Fire-and-forget insert (don't block routing decision)
        let client = self.client.clone();
        let request = InsertRequest {
            model_id: model_id.to_string(),
            text: text.to_string(),
            tenant_id: tenant.to_string(),
            timestamp_ms: 0,
        };

        tokio::spawn(async move {
            let _ = client.insert(request).await;
        });
    }
}
```

### Benefits of Standalone Service

1. **Horizontal Scalability**: Can run multiple tree service instances behind a load balancer
2. **Resource Isolation**: Tree CPU usage doesn't compete with gateway's HTTP handling
3. **Independent Deployment**: Can upgrade tree service without touching gateway
4. **State Persistence**: Can add Redis/disk persistence for crash recovery
5. **Multi-Gateway Support**: Multiple gateway instances can share one tree service
6. **Specialized Optimization**: Can use optimized data structures (e.g., C++ radix tree)

### Trade-offs

1. **Latency**: Adds 0.5-2ms network round-trip per operation
2. **Complexity**: More components to operate and monitor
3. **Failure Modes**: Network partitions, service unavailability
4. **Resource Overhead**: Additional containers/processes

### Mitigation Strategies

1. **Batching**: Batch multiple requests per RPC (reduces overhead by 10-100x)
2. **Local Caching**: Cache hot paths in gateway (100ms TTL)
3. **Async Inserts**: Fire-and-forget inserts, only block on matches
4. **Connection Pooling**: Reuse gRPC connections
5. **Circuit Breakers**: Fall back to random routing on service failure

## Recommended Implementation Order

1. **Week 1**: Implement Phase 1 optimizations (timestamp, lazy updates, pooling)
   - Expected: 30-40% CPU reduction

2. **Week 2-3**: Implement Phase 2 (immutable text, sharding)
   - Expected: Additional 30-40% reduction

3. **Week 4-6**: Implement gRPC service if still needed
   - Expected: Near-linear horizontal scaling

## Metrics to Track

- `tree_insert_latency_us`: Insert operation latency
- `tree_match_latency_us`: Prefix match latency
- `tree_lock_contention_count`: Lock acquisition failures/retries
- `tree_node_count`: Total nodes per model
- `tree_eviction_duration_ms`: Time spent in eviction
- `tree_cpu_percent`: CPU usage attributed to tree operations

## Conclusion

The primary bottlenecks are:
1. **RwLock contention** on node text (fixable with immutable Arc<str>)
2. **Syscall overhead** for timestamps (fixable with background updater)
3. **Memory allocation** per operation (fixable with pooling)

For most workloads, Phase 1+2 optimizations should be sufficient. The gRPC service is recommended only if:
- Multiple gateway instances need shared cache state
- Tree operations exceed 50% of gateway CPU
- Horizontal scaling of tree service is required
