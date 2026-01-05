# vLLM Cache-Aware Load Balancing Design

## Table of Contents

1. [Overview](#1-overview)
2. [Background: vLLM KV Cache Management](#2-background-vllm-kv-cache-management)
3. [Background: SGLang Radix Tree](#3-background-sglang-radix-tree)
4. [Comparison: Hash Table vs Radix Tree](#4-comparison-hash-table-vs-radix-tree)
5. [Gateway Design: Block Cache Tracker](#5-gateway-design-block-cache-tracker)
6. [Data Structures](#6-data-structures)
7. [Core Algorithms](#7-core-algorithms)
8. [Optimizations](#8-optimizations)
9. [Integration](#9-integration)
10. [Configuration](#10-configuration)
11. [References](#11-references)

---

## 1. Overview

This document describes a cache-aware load balancing strategy for SGLang gateway routing to vLLM backends. The design mirrors vLLM's Automatic Prefix Caching (APC) mechanism to accurately predict which backend has cached KV blocks for incoming requests.

### Goals

- **Accuracy**: Match vLLM's block-level caching granularity
- **Performance**: O(log(n/B)) lookup instead of O(n) traversal
- **Scalability**: Handle millions of cached blocks across multiple backends
- **Compatibility**: Work with vLLM's configurable block sizes (16, 32, 128)

---

## 2. Background: vLLM KV Cache Management

### 2.1 PagedAttention Architecture

vLLM uses PagedAttention to manage KV cache as fixed-size blocks:

```
┌─────────────────────────────────────────────────────────────┐
│                    vLLM Memory Layout                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Logical View:     [Token 0-15] [Token 16-31] [Token 32-47] │
│                         ↓            ↓             ↓        │
│  Block Table:      Block #42    Block #87     Block #23     │
│                         ↓            ↓             ↓        │
│  Physical GPU:     [GPU Mem]    [GPU Mem]     [GPU Mem]     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Reference**: [vLLM PagedAttention Design](https://docs.vllm.ai/en/stable/design/paged_attention/)

### 2.2 Block Size Configuration

| Platform | Default Block Size | Supported Sizes |
|----------|-------------------|-----------------|
| CUDA     | 16                | 8, 16, 32       |
| HPU      | 128               | 128             |

**Reference**: [vLLM Config](https://docs.vllm.ai/en/stable/api/vllm/config/)

```python
# vLLM engine arguments
--block-size 16  # Tokens per block
```

### 2.3 Automatic Prefix Caching (APC)

vLLM identifies cached blocks using content-based hashing:

```python
# Conceptual block hash computation (vLLM)
def compute_block_hash(parent_hash: Optional[int],
                       block_tokens: Tuple[int, ...],
                       extra_hash: Optional[int] = None) -> int:
    """
    Reference: vllm/core/block_manager.py

    Components:
    - parent_hash: Hash of previous block (chain integrity)
    - block_tokens: Tuple of token IDs in this block
    - extra_hash: LoRA ID, cache salt, multimodal hashes
    """
    return hash((parent_hash, block_tokens, extra_hash))
```

**Key Properties**:
1. Only **complete blocks** are cached (partial blocks ignored)
2. Hash includes **parent block hash** for chain integrity
3. Blocks with **ref_count = 0** are evictable
4. **LRU eviction** with prefix-length tiebreaker

**Reference**: [vLLM APC RFC](https://github.com/vllm-project/vllm/issues/2614)

### 2.4 vLLM Block Data Structure

```python
# Conceptual block metadata
class BlockMetadata:
    block_hash: int           # Content-based hash
    ref_count: int            # Active sequences using this block
    last_accessed: float      # Timestamp for LRU
    num_hashed_tokens: int    # Prefix length (for eviction tiebreaker)
```

### 2.5 vLLM Eviction Policy

Priority for eviction (lowest = evict first):

```
1. ref_count == 0           # Only evict unreferenced blocks
2. last_accessed (oldest)   # LRU ordering
3. num_hashed_tokens (longest prefix first)  # Tiebreaker
```

---

## 3. Background: SGLang Radix Tree

### 3.1 Token-Level Radix Tree

SGLang uses a radix tree for KV cache management at token granularity:

```
┌─────────────────────────────────────────────────────────────┐
│                  SGLang Radix Tree                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│                        [root]                                │
│                       /      \                               │
│               [1,2,3]         [5,6,7]                       │
│              /       \             \                         │
│        [4,5,6]    [7,8,9]       [8,9,10]                    │
│                                                              │
│  Each node stores:                                           │
│  - Token sequence (key)                                      │
│  - KV cache indices (value)                                  │
│  - Lock reference count                                      │
│  - Last access time                                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Reference**: `python/sglang/srt/mem_cache/radix_cache.py`

### 3.2 SGLang TreeNode Structure

```python
# Reference: radix_cache.py:88-113
class TreeNode:
    children: defaultdict(TreeNode)
    key: RadixKey              # Token IDs + extra_key
    value: torch.Tensor        # KV cache indices (GPU memory pointers)
    lock_ref: int              # Reference count for active requests
    last_access_time: float    # For LRU eviction
    hit_count: int             # For LFU eviction
    priority: int              # For priority-based eviction
    host_value: torch.Tensor   # CPU offload storage
    hash_value: List[str]      # For disaggregated KV cache
```

### 3.3 SGLang Gateway Radix Tree

The gateway uses a separate radix tree for routing (text-based):

```rust
// Reference: sgl-model-gateway/src/policies/tree.rs:229-243
struct Node {
    children: DashMap<char, NodeRef>,           // First-char indexed
    text: RwLock<NodeText>,                     // Node text
    tenant_last_access_time: DashMap<TenantId, u64>,  // Per-backend LRU
    parent: RwLock<Option<NodeRef>>,            // Parent pointer
    last_tenant: RwLock<Option<TenantId>>,      // Cached last accessor
}
```

**Reference**: `sgl-model-gateway/src/policies/tree.rs`

---

## 4. Comparison: Hash Table vs Radix Tree

### 4.1 Complexity Analysis

| Operation | Radix Tree | Hash Table (Chained) | Hash Table (Position) |
|-----------|-----------|---------------------|----------------------|
| Prefix Match | O(m) | O(m/B) sequential | O(log(m/B)) binary search |
| Insert | O(m) | O(m/B) | O(m/B) |
| Memory | O(unique tokens) | O(unique blocks) | O(unique blocks) |
| Parallelizable | No | No (chain dependency) | Yes |

Where: `m` = token count, `B` = block size

### 4.2 Hash Chain Dependency Problem

**vLLM-style chained hashing**:
```
hash₀ = hash(∅, tokens[0:16])
hash₁ = hash(hash₀, tokens[16:32])    ← Depends on hash₀
hash₂ = hash(hash₁, tokens[32:48])    ← Depends on hash₁
```

This creates sequential dependency, requiring O(m/B) lookups.

**Position-based hashing** (our optimization):
```
hash₀ = hash(position=0, tokens[0:16])     ← Independent
hash₁ = hash(position=1, tokens[16:32])    ← Independent
hash₂ = hash(position=2, tokens[32:48])    ← Independent
```

Enables parallel computation and binary search.

### 4.3 When Each Approach Wins

| Scenario | Better Approach | Reason |
|----------|----------------|--------|
| Long prompts (>100 tokens) | Hash Table | O(log(m/B)) vs O(m) |
| Shared system prompts | Both similar | Tree shares nodes, hash shares blocks |
| Short prefixes (<block_size) | Radix Tree | Can match partial |
| vLLM backend | Hash Table | Matches actual cache behavior |
| SGLang backend | Radix Tree | Token-level granularity |

---

## 5. Gateway Design: Block Cache Tracker

### 5.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Gateway                                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                   BlockCacheTracker                          │   │
│  ├─────────────────────────────────────────────────────────────┤   │
│  │                                                              │   │
│  │   blocks: DashMap<BlockHash, BlockEntry>                    │   │
│  │     │                                                        │   │
│  │     └─► BlockEntry                                          │   │
│  │           ├── backends: DashMap<BackendId, AccessEpoch>     │   │
│  │           └── position: usize                                │   │
│  │                                                              │   │
│  │   backend_stats: DashMap<BackendId, BackendStats>           │   │
│  │     └─► BackendStats { block_count, last_seen }             │   │
│  │                                                              │   │
│  │   epoch: AtomicU64                                          │   │
│  │   config: BlockCacheConfig                                  │   │
│  │                                                              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                 VllmCacheAwarePolicy                         │   │
│  │                                                              │   │
│  │   fn select_worker(workers, info) -> Option<usize>          │   │
│  │     1. find_best_backend(tokens) → binary search            │   │
│  │     2. check load balance                                    │   │
│  │     3. record_request(tokens, selected_backend)             │   │
│  │                                                              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
        ┌──────────────┬──────────────┬──────────────┐
        │   vLLM #1    │   vLLM #2    │   vLLM #3    │
        │  block_size  │  block_size  │  block_size  │
        │     =16      │     =16      │     =16      │
        └──────────────┴──────────────┴──────────────┘
```

### 5.2 Request Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Request Flow                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. Request arrives                                                  │
│     └─► tokens = tokenizer.encode(prompt)                           │
│                                                                      │
│  2. Find best backend (O(log(n/B)))                                 │
│     └─► (backend, hit_ratio) = tracker.find_best_backend(tokens)    │
│                                                                      │
│  3. Routing decision                                                 │
│     ├─► hit_ratio > threshold?                                      │
│     │     └─► Route to cached backend (if healthy & load OK)        │
│     └─► Otherwise                                                    │
│           └─► Route to least-loaded backend                          │
│                                                                      │
│  4. Record for future routing                                        │
│     └─► tracker.record_request(tokens, selected_backend)            │
│                                                                      │
│  5. Forward request                                                  │
│     └─► Send to selected vLLM backend                               │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 6. Data Structures

### 6.1 BlockHash

```rust
/// 64-bit hash uniquely identifying a block by position and content.
/// Uses xxHash for speed (non-cryptographic).
pub type BlockHash = u64;
```

### 6.2 BackendId

```rust
/// Backend identifier (URL).
/// Using Arc<str> for cheap cloning and comparison.
pub type BackendId = Arc<str>;
```

### 6.3 BlockEntry

```rust
/// Information about a cached block across backends.
#[derive(Debug)]
pub struct BlockEntry {
    /// Which backends have this block cached, with access epochs.
    /// Key: backend URL, Value: last access epoch
    pub backends: DashMap<BackendId, u64>,

    /// Block position in sequence (for validation).
    pub position: usize,
}
```

### 6.4 BackendStats

```rust
/// Per-backend statistics for eviction and monitoring.
#[derive(Debug)]
pub struct BackendStats {
    /// Number of blocks tracked for this backend.
    pub block_count: usize,

    /// Last time this backend was selected.
    pub last_seen: u64,
}
```

### 6.5 BlockCacheTracker

```rust
/// Main cache tracker mirroring vLLM's block hash table.
#[derive(Debug)]
pub struct BlockCacheTracker {
    /// Global block registry: hash → entry
    /// Mirrors vLLM's: hash_table[block_hash] → physical_block
    blocks: DashMap<BlockHash, BlockEntry>,

    /// Per-backend statistics
    backend_stats: DashMap<BackendId, BackendStats>,

    /// Monotonic epoch counter for LRU (no syscalls)
    epoch: AtomicU64,

    /// Configuration
    config: BlockCacheConfig,
}
```

### 6.6 BlockCacheConfig

```rust
/// Configuration for the block cache tracker.
#[derive(Debug, Clone)]
pub struct BlockCacheConfig {
    /// Tokens per block (must match vLLM backend).
    /// CUDA default: 16, HPU default: 128
    pub block_size: usize,

    /// Maximum blocks tracked per backend (memory bound).
    pub max_blocks_per_backend: usize,

    /// Background eviction interval in seconds.
    pub eviction_interval_secs: u64,

    /// Minimum cache hit ratio to prefer cached backend.
    pub cache_threshold: f32,

    /// Load factor for bounded load balancing.
    /// Route to cached backend only if load < avg * load_factor
    pub load_factor: f32,
}

impl Default for BlockCacheConfig {
    fn default() -> Self {
        Self {
            block_size: 16,
            max_blocks_per_backend: 100_000,  // ~1.6M tokens
            eviction_interval_secs: 60,
            cache_threshold: 0.3,
            load_factor: 1.25,
        }
    }
}
```

---

## 7. Core Algorithms

### 7.1 Block Hash Computation

**Position-based hashing** (enables parallelism and binary search):

```rust
impl BlockCacheTracker {
    /// Compute hash for a block at a given position.
    ///
    /// Uses position-based hashing instead of chained hashing to enable:
    /// 1. Parallel hash computation
    /// 2. Binary search for longest match
    ///
    /// Trade-off: Loses chain integrity verification, but this is acceptable
    /// for routing purposes (we trust the backends).
    #[inline]
    fn compute_block_hash(position: usize, tokens: &[u32]) -> BlockHash {
        use xxhash_rust::xxh3::xxh3_64;

        // Serialize position + tokens
        let mut buffer = Vec::with_capacity(8 + tokens.len() * 4);
        buffer.extend_from_slice(&position.to_le_bytes());
        buffer.extend_from_slice(bytemuck::cast_slice(tokens));

        xxh3_64(&buffer)
    }

    /// Compute all block hashes for a token sequence.
    /// Can be parallelized with rayon if needed.
    fn compute_all_hashes(&self, tokens: &[u32]) -> Vec<BlockHash> {
        let block_size = self.config.block_size;
        let num_blocks = tokens.len() / block_size;

        (0..num_blocks)
            .map(|i| {
                let start = i * block_size;
                let end = start + block_size;
                Self::compute_block_hash(i, &tokens[start..end])
            })
            .collect()
    }
}
```

### 7.2 Record Request

Called after routing decision to update cache state:

```rust
impl BlockCacheTracker {
    /// Record that a backend processed a request with these tokens.
    /// Assumes vLLM will cache complete blocks via APC.
    pub fn record_request(&self, tokens: &[u32], backend: &BackendId) {
        let block_size = self.config.block_size;
        let epoch = self.epoch.fetch_add(1, Ordering::Relaxed);

        let mut new_blocks = 0usize;

        for (position, chunk) in tokens.chunks(block_size).enumerate() {
            // Only complete blocks are cached (matching vLLM behavior)
            if chunk.len() < block_size {
                break;
            }

            let block_hash = Self::compute_block_hash(position, chunk);

            // Upsert block entry
            let entry = self.blocks
                .entry(block_hash)
                .or_insert_with(|| BlockEntry {
                    backends: DashMap::new(),
                    position,
                });

            // Track if this is a new block for this backend
            let is_new = !entry.backends.contains_key(backend);

            // Update access time
            entry.backends.insert(Arc::clone(backend), epoch);

            if is_new {
                new_blocks += 1;
            }
        }

        // Update backend stats
        if new_blocks > 0 {
            self.backend_stats
                .entry(Arc::clone(backend))
                .and_modify(|s| s.block_count += new_blocks)
                .or_insert(BackendStats {
                    block_count: new_blocks,
                    last_seen: epoch,
                });
        }
    }
}
```

### 7.3 Find Best Backend (Binary Search)

O(log(n/B)) lookup for longest cached prefix:

```rust
impl BlockCacheTracker {
    /// Find the backend with the longest cached prefix.
    ///
    /// Returns: Option<(backend_id, num_matched_blocks, total_blocks)>
    ///
    /// Complexity: O(log(n/B)) where n = tokens, B = block_size
    pub fn find_best_backend(&self, tokens: &[u32]) -> Option<(BackendId, usize, usize)> {
        let block_size = self.config.block_size;
        let num_blocks = tokens.len() / block_size;

        if num_blocks == 0 {
            return None;
        }

        // Precompute all block hashes
        let hashes = self.compute_all_hashes(tokens);

        // Binary search for longest contiguous match
        let (best_backend, match_len) = self.binary_search_match(&hashes)?;

        // Update access times for matched blocks
        let epoch = self.epoch.fetch_add(1, Ordering::Relaxed);
        for hash in &hashes[..match_len] {
            if let Some(entry) = self.blocks.get(hash) {
                if let Some(mut access) = entry.backends.get_mut(&best_backend) {
                    *access = epoch;
                }
            }
        }

        Some((best_backend, match_len, num_blocks))
    }

    /// Binary search to find longest contiguous prefix match.
    fn binary_search_match(&self, hashes: &[BlockHash]) -> Option<(BackendId, usize)> {
        if hashes.is_empty() {
            return None;
        }

        // First, check if block 0 exists at all
        let first_entry = self.blocks.get(&hashes[0])?;

        // Collect candidate backends (those that have block 0)
        let candidates: Vec<BackendId> = first_entry
            .backends
            .iter()
            .map(|e| Arc::clone(e.key()))
            .collect();

        if candidates.is_empty() {
            return None;
        }

        // For each candidate, binary search for their longest match
        let mut best_backend = None;
        let mut best_len = 0;

        for backend in candidates {
            let len = self.binary_search_backend_match(hashes, &backend);
            if len > best_len {
                best_len = len;
                best_backend = Some(backend);
            }
        }

        best_backend.map(|b| (b, best_len))
    }

    /// Binary search for a specific backend's match length.
    fn binary_search_backend_match(&self, hashes: &[BlockHash], backend: &BackendId) -> usize {
        let mut lo = 0;
        let mut hi = hashes.len();

        while lo < hi {
            let mid = lo + (hi - lo + 1) / 2;

            // Check if all blocks [0..mid] are cached by this backend
            if self.has_contiguous_blocks(hashes, backend, mid) {
                lo = mid;  // Can match at least 'mid' blocks
            } else {
                hi = mid - 1;  // Match is shorter
            }
        }

        lo
    }

    /// Check if backend has all blocks [0..count] cached.
    #[inline]
    fn has_contiguous_blocks(&self, hashes: &[BlockHash], backend: &BackendId, count: usize) -> bool {
        if count == 0 {
            return true;
        }

        // Only need to check the last block (blocks are contiguous)
        // If block N-1 exists, blocks 0..N-1 must also exist
        // (because we only insert contiguously in record_request)
        self.blocks
            .get(&hashes[count - 1])
            .map(|entry| entry.backends.contains_key(backend))
            .unwrap_or(false)
    }
}
```

### 7.4 Eviction (LRU per Backend)

```rust
impl BlockCacheTracker {
    /// Evict oldest blocks for a backend to stay under limit.
    pub fn evict_backend(&self, backend: &BackendId, max_blocks: usize) {
        let current = self.backend_stats
            .get(backend)
            .map(|s| s.block_count)
            .unwrap_or(0);

        if current <= max_blocks {
            return;
        }

        // Collect blocks owned by this backend with access times
        let mut owned: Vec<(BlockHash, u64)> = Vec::with_capacity(current);

        for entry in self.blocks.iter() {
            if let Some(epoch) = entry.backends.get(backend) {
                owned.push((*entry.key(), *epoch));
            }
        }

        // Sort by access time (oldest first)
        owned.sort_unstable_by_key(|(_, epoch)| *epoch);

        // Evict oldest blocks
        let to_evict = current.saturating_sub(max_blocks);
        let mut evicted = 0;

        for (hash, _) in owned.into_iter().take(to_evict) {
            if let Some(entry) = self.blocks.get(&hash) {
                entry.backends.remove(backend);
                evicted += 1;

                // Remove entry if no backends own it
                if entry.backends.is_empty() {
                    drop(entry);
                    self.blocks.remove(&hash);
                }
            }
        }

        // Update stats
        if let Some(mut stats) = self.backend_stats.get_mut(backend) {
            stats.block_count = stats.block_count.saturating_sub(evicted);
        }
    }

    /// Remove all blocks for a backend (when backend goes offline).
    pub fn remove_backend(&self, backend: &BackendId) {
        // Remove from all block entries
        self.blocks.retain(|_, entry| {
            entry.backends.remove(backend);
            !entry.backends.is_empty()
        });

        // Remove stats
        self.backend_stats.remove(backend);
    }
}
```

---

## 8. Optimizations

### 8.1 Position-Based Hashing

**Problem**: vLLM uses chained hashing where each block's hash depends on the parent:
```
hash₁ = hash(hash₀, tokens)  // Sequential dependency
```

**Solution**: Use position-based hashing:
```
hash₁ = hash(position=1, tokens)  // Independent
```

**Benefits**:
- Parallel hash computation
- Binary search instead of linear scan
- O(log(n/B)) vs O(n/B)

**Trade-off**:
- Loses chain integrity verification
- Acceptable for routing (we trust backends)

### 8.2 Binary Search for Longest Match

**Instead of**:
```
for each block 0..n:
    if exists: continue
    else: break
# O(n/B) lookups
```

**Use**:
```
binary_search(0, n):
    if block[mid] exists for backend:
        search upper half
    else:
        search lower half
# O(log(n/B)) lookups
```

### 8.3 Contiguous Block Assumption

**Insight**: Blocks are always inserted contiguously (0, 1, 2, ...).

**Optimization**: To check if blocks [0..k] exist, only need to check block k-1:
```rust
fn has_contiguous_blocks(hashes, backend, count) -> bool {
    // If block count-1 exists, all previous blocks must exist
    blocks.get(&hashes[count - 1])?.backends.contains_key(backend)
}
```

### 8.4 Epoch-Based LRU

**Instead of**: `SystemTime::now()` (syscall overhead)

**Use**: Atomic counter (no syscalls, perfectly monotonic)

```rust
static EPOCH: AtomicU64 = AtomicU64::new(0);

fn get_epoch() -> u64 {
    EPOCH.fetch_add(1, Ordering::Relaxed)
}
```

**Reference**: Same pattern used in `sgl-model-gateway/src/policies/tree.rs:218-227`

### 8.5 Probabilistic Timestamp Updates

**Problem**: Updating access time on every lookup causes contention.

**Solution**: Update probabilistically (1 in N matches):

```rust
let epoch = get_epoch();
if epoch & 0x7 == 0 {  // 1 in 8
    entry.last_access = epoch;
}
```

**Reference**: `sgl-model-gateway/src/policies/tree.rs:603-607`

---

## 9. Integration

### 9.1 VllmCacheAwarePolicy

```rust
/// Load balancing policy for vLLM backends with cache awareness.
#[derive(Debug)]
pub struct VllmCacheAwarePolicy {
    /// Block cache tracker
    tracker: Arc<BlockCacheTracker>,

    /// Per-model trackers (if different models have different block sizes)
    model_trackers: DashMap<String, Arc<BlockCacheTracker>>,

    /// Background eviction task
    eviction_task: Option<PeriodicTask>,
}

impl LoadBalancingPolicy for VllmCacheAwarePolicy {
    fn select_worker(&self, workers: &[Arc<dyn Worker>], info: &SelectWorkerInfo) -> Option<usize> {
        let tokens = info.tokens?;
        let healthy_workers = get_healthy_indices(workers);

        if healthy_workers.is_empty() {
            return None;
        }

        // Find best cached backend
        if let Some((backend, hits, total)) = self.tracker.find_best_backend(tokens) {
            let hit_ratio = hits as f32 / total as f32;

            if hit_ratio > self.tracker.config.cache_threshold {
                // Check if backend is healthy and load is acceptable
                if let Some(idx) = workers.iter().position(|w| w.url() == backend.as_ref()) {
                    if workers[idx].is_healthy() && self.load_acceptable(workers, idx) {
                        self.tracker.record_request(tokens, &backend);
                        return Some(idx);
                    }
                }
            }
        }

        // Fallback: least loaded worker
        let idx = self.select_least_loaded(workers, &healthy_workers)?;
        let backend: BackendId = Arc::from(workers[idx].url());
        self.tracker.record_request(tokens, &backend);
        Some(idx)
    }

    fn name(&self) -> &'static str {
        "vllm_cache_aware"
    }

    fn needs_tokens(&self) -> bool {
        true  // Requires tokenized input
    }
}
```

### 9.2 Configuration Integration

```yaml
# gateway config
routing:
  policy: vllm_cache_aware
  vllm_cache_aware:
    block_size: 16
    max_blocks_per_backend: 100000
    eviction_interval_secs: 60
    cache_threshold: 0.3
    load_factor: 1.25
```

---

## 10. Configuration

### 10.1 Block Size Matching

**Critical**: Gateway block size must match vLLM backend block size.

Options:
1. **Static config**: Set in gateway config, must match deployment
2. **Query backend**: GET `/health` or `/v1/models` for block size
3. **Per-model config**: Different models may have different block sizes

### 10.2 Recommended Defaults

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| block_size | 16 | CUDA default, most common |
| max_blocks_per_backend | 100,000 | ~1.6M tokens, ~100MB memory |
| eviction_interval_secs | 60 | Balance freshness vs overhead |
| cache_threshold | 0.3 | 30% prefix match triggers affinity |
| load_factor | 1.25 | Allow 25% above average load |

---

## 11. References

### vLLM Documentation
- [PagedAttention Design](https://docs.vllm.ai/en/stable/design/paged_attention/)
- [Automatic Prefix Caching](https://docs.vllm.ai/en/stable/design/prefix_caching/)
- [Hybrid KV Cache Manager](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/)
- [Config API](https://docs.vllm.ai/en/stable/api/vllm/config/)

### vLLM Source Code
- [APC RFC](https://github.com/vllm-project/vllm/issues/2614)
- [Block Size Limits](https://github.com/vllm-project/vllm/issues/2594)

### SGLang Source Code
- Radix Cache: `python/sglang/srt/mem_cache/radix_cache.py`
- Gateway Tree: `sgl-model-gateway/src/policies/tree.rs`
- Cache Aware Policy: `sgl-model-gateway/src/policies/cache_aware.rs`
- Prefix Hash Policy: `sgl-model-gateway/src/policies/prefix_hash.rs`

### Comparison Articles
- [SGLang vs vLLM Prefix Caching](https://medium.com/byte-sized-ai/prefix-caching-sglang-vs-vllm-token-level-radix-tree-vs-block-level-hashing-b99ece9977a1)
