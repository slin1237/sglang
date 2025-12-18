// Optimized Radix Tree Implementation
// This file contains key optimizations to address performance bottlenecks
// identified in the original tree.rs implementation.

use std::{
    collections::HashMap,
    hash::{BuildHasherDefault, Hasher},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, OnceLock,
    },
    thread,
    time::{Duration, Instant},
};

use dashmap::{mapref::entry::Entry, DashMap};

type NodeRef = Arc<Node>;

/// Interned tenant ID to avoid repeated string allocations.
pub type TenantId = Arc<str>;

// ============================================================================
// OPTIMIZATION 1: True Timestamp Caching with Background Updater
// ============================================================================

/// Global monotonic timestamp that's updated by a background thread.
/// This eliminates syscalls from the hot path entirely.
static CACHED_TIMESTAMP_MS: AtomicU64 = AtomicU64::new(0);
static TIMESTAMP_BASE: OnceLock<Instant> = OnceLock::new();
static TIMESTAMP_UPDATER_STARTED: OnceLock<()> = OnceLock::new();

/// Start the background timestamp updater thread.
/// Call this once during initialization.
pub fn start_timestamp_updater() {
    TIMESTAMP_UPDATER_STARTED.get_or_init(|| {
        let base = TIMESTAMP_BASE.get_or_init(Instant::now);
        // Store initial timestamp
        CACHED_TIMESTAMP_MS.store(base.elapsed().as_millis() as u64, Ordering::Relaxed);

        thread::spawn(move || {
            let base = *TIMESTAMP_BASE.get().unwrap();
            loop {
                let elapsed = base.elapsed().as_millis() as u64;
                CACHED_TIMESTAMP_MS.store(elapsed, Ordering::Relaxed);
                // Update every 1ms - fast enough for LRU, no syscall overhead
                thread::sleep(Duration::from_millis(1));
            }
        });
    });
}

/// Get current timestamp without ANY syscall.
/// Falls back to syscall if updater hasn't started yet.
#[inline(always)]
fn get_timestamp_ms() -> u128 {
    let ts = CACHED_TIMESTAMP_MS.load(Ordering::Relaxed);
    if ts == 0 {
        // Fallback: updater not started yet
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis()
    } else {
        ts as u128
    }
}

// ============================================================================
// OPTIMIZATION 2: Custom Fast Hasher for Char Keys
// ============================================================================

#[derive(Default)]
struct CharHasher(u64);

impl Hasher for CharHasher {
    #[inline(always)]
    fn finish(&self) -> u64 {
        self.0
    }

    #[inline(always)]
    fn write(&mut self, bytes: &[u8]) {
        // Fast path for 4-byte (char) writes
        if bytes.len() == 4 {
            let val = u32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            self.0 = (val as u64).wrapping_mul(0x9E3779B97F4A7C15);
            return;
        }
        for &byte in bytes {
            self.0 = self.0.wrapping_mul(0x100000001b3).wrapping_add(byte as u64);
        }
    }

    #[inline(always)]
    fn write_u32(&mut self, i: u32) {
        self.0 = (i as u64).wrapping_mul(0x9E3779B97F4A7C15);
    }
}

type CharHasherBuilder = BuildHasherDefault<CharHasher>;

// ============================================================================
// OPTIMIZATION 3: Immutable Node Text (No RwLock for reads!)
// ============================================================================

/// Immutable node text - no lock needed for reading.
/// Text is set once and never changes. For splits, we create new nodes.
#[derive(Debug, Clone)]
struct ImmutableText {
    /// The actual text stored in this node (immutable after creation)
    text: Arc<str>,
    /// Cached character count
    char_count: usize,
    /// Pre-indexed chars for O(1) access (lazy, computed once)
    chars: Arc<[char]>,
}

impl ImmutableText {
    fn new(text: &str) -> Self {
        let chars: Arc<[char]> = text.chars().collect::<Vec<_>>().into();
        Self {
            text: Arc::from(text),
            char_count: chars.len(),
            chars,
        }
    }

    fn empty() -> Self {
        Self {
            text: Arc::from(""),
            char_count: 0,
            chars: Arc::from([]),
        }
    }

    #[inline]
    fn char_count(&self) -> usize {
        self.char_count
    }

    #[inline]
    fn as_str(&self) -> &str {
        &self.text
    }

    #[inline]
    fn first_char(&self) -> Option<char> {
        self.chars.first().copied()
    }

    #[inline]
    fn get_char(&self, idx: usize) -> Option<char> {
        self.chars.get(idx).copied()
    }

    /// Split at character index, returns (prefix, suffix)
    fn split_at(&self, char_idx: usize) -> (Self, Self) {
        if char_idx == 0 {
            return (Self::empty(), self.clone());
        }
        if char_idx >= self.char_count {
            return (self.clone(), Self::empty());
        }

        let prefix_chars: Arc<[char]> = self.chars[..char_idx].into();
        let suffix_chars: Arc<[char]> = self.chars[char_idx..].into();

        let prefix_text: String = prefix_chars.iter().collect();
        let suffix_text: String = suffix_chars.iter().collect();

        (
            Self {
                text: Arc::from(prefix_text.as_str()),
                char_count: char_idx,
                chars: prefix_chars,
            },
            Self {
                text: Arc::from(suffix_text.as_str()),
                char_count: self.char_count - char_idx,
                chars: suffix_chars,
            },
        )
    }
}

// ============================================================================
// OPTIMIZATION 4: Lock-Free Node Structure
// ============================================================================

#[derive(Debug)]
struct Node {
    /// Children nodes indexed by first character.
    children: DashMap<char, NodeRef, CharHasherBuilder>,

    /// IMMUTABLE text - no lock needed!
    text: ImmutableText,

    /// Per-tenant last access timestamps
    tenant_last_access_time: DashMap<TenantId, u128>,

    /// Parent pointer (rarely modified, could use AtomicPtr but DashMap entry works)
    parent: DashMap<(), NodeRef>,  // Single entry map for optional parent
}

impl Node {
    fn new(text: ImmutableText, parent: Option<NodeRef>) -> Self {
        let node = Self {
            children: DashMap::with_hasher(CharHasherBuilder::default()),
            text,
            tenant_last_access_time: DashMap::new(),
            parent: DashMap::new(),
        };
        if let Some(p) = parent {
            node.parent.insert((), p);
        }
        node
    }

    fn get_parent(&self) -> Option<NodeRef> {
        self.parent.get(&()).map(|r| r.value().clone())
    }

    fn set_parent(&self, parent: NodeRef) {
        self.parent.insert((), parent);
    }
}

// ============================================================================
// OPTIMIZATION 5: Probabilistic Timestamp Updates
// ============================================================================

/// Counter for probabilistic updates
static UPDATE_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Only update timestamps every N operations (default: 10)
const TIMESTAMP_UPDATE_FREQUENCY: u64 = 10;

#[inline]
fn should_update_timestamps() -> bool {
    let count = UPDATE_COUNTER.fetch_add(1, Ordering::Relaxed);
    count % TIMESTAMP_UPDATE_FREQUENCY == 0
}

// ============================================================================
// OPTIMIZED TREE IMPLEMENTATION
// ============================================================================

#[derive(Debug)]
pub struct OptimizedTree {
    root: NodeRef,
    pub tenant_char_count: DashMap<TenantId, usize>,
}

impl Default for OptimizedTree {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizedTree {
    pub fn new() -> Self {
        // Ensure timestamp updater is running
        start_timestamp_updater();

        Self {
            root: Arc::new(Node::new(ImmutableText::empty(), None)),
            tenant_char_count: DashMap::new(),
        }
    }

    pub fn insert(&self, text: &str, tenant: &str) {
        if text.is_empty() {
            return;
        }

        // Pre-index text once
        let chars: Vec<char> = text.chars().collect();
        let text_count = chars.len();

        let mut curr = Arc::clone(&self.root);
        let mut curr_idx = 0;

        // Use cached timestamp (no syscall!)
        let timestamp_ms = get_timestamp_ms();

        // Intern tenant ID
        let tenant_id: TenantId = Arc::from(tenant);

        curr.tenant_last_access_time
            .insert(Arc::clone(&tenant_id), timestamp_ms);

        self.tenant_char_count
            .entry(Arc::clone(&tenant_id))
            .or_insert(0);

        while curr_idx < text_count {
            let first_char = chars[curr_idx];

            // Check if child exists first (releases borrow immediately)
            let existing_child = curr.children.get(&first_char).map(|e| e.value().clone());

            match existing_child {
                None => {
                    // No existing child - create new node with remaining text
                    let new_text: String = chars[curr_idx..].iter().collect();
                    let new_node = Arc::new(Node::new(
                        ImmutableText::new(&new_text),
                        Some(Arc::clone(&curr)),
                    ));

                    // Update tenant tracking
                    let added_chars = text_count - curr_idx;
                    self.tenant_char_count
                        .entry(Arc::clone(&tenant_id))
                        .and_modify(|c| *c += added_chars)
                        .or_insert(added_chars);

                    new_node
                        .tenant_last_access_time
                        .insert(Arc::clone(&tenant_id), timestamp_ms);

                    // Insert into children (may race with another insert, but DashMap handles it)
                    curr.children.insert(first_char, Arc::clone(&new_node));
                    break;
                }

                Some(matched_node) => {
                    let matched_text = &matched_node.text;
                    let matched_count = matched_text.char_count();

                    // Count shared prefix
                    let mut shared = 0;
                    while shared < matched_count && curr_idx + shared < text_count {
                        if matched_text.get_char(shared) != Some(chars[curr_idx + shared]) {
                            break;
                        }
                        shared += 1;
                    }

                    if shared < matched_count {
                        // Need to split node
                        let (prefix_text, suffix_text) = matched_text.split_at(shared);

                        // Create new intermediate node
                        let new_node = Arc::new(Node::new(prefix_text, Some(Arc::clone(&curr))));

                        // Copy tenant timestamps to new node
                        for item in matched_node.tenant_last_access_time.iter() {
                            new_node
                                .tenant_last_access_time
                                .insert(Arc::clone(item.key()), *item.value());
                        }

                        // Create contracted node with suffix
                        let contracted = Arc::new(Node::new(suffix_text, Some(Arc::clone(&new_node))));

                        // Move children and timestamps from original to contracted
                        for child in matched_node.children.iter() {
                            contracted.children.insert(*child.key(), child.value().clone());
                            // Update child's parent
                            child.value().set_parent(Arc::clone(&contracted));
                        }
                        for item in matched_node.tenant_last_access_time.iter() {
                            contracted
                                .tenant_last_access_time
                                .insert(Arc::clone(item.key()), *item.value());
                        }

                        // Link contracted as child of new node
                        if let Some(first) = contracted.text.first_char() {
                            new_node.children.insert(first, contracted);
                        }

                        // Replace in parent's children
                        curr.children.insert(first_char, Arc::clone(&new_node));

                        // Add current tenant to new node
                        if !new_node.tenant_last_access_time.contains_key(tenant_id.as_ref()) {
                            self.tenant_char_count
                                .entry(Arc::clone(&tenant_id))
                                .and_modify(|c| *c += shared)
                                .or_insert(shared);
                        }
                        new_node.tenant_last_access_time
                            .insert(Arc::clone(&tenant_id), timestamp_ms);

                        curr = new_node;
                        curr_idx += shared;
                    } else {
                        // Full match, continue to next node
                        if !matched_node.tenant_last_access_time.contains_key(tenant_id.as_ref()) {
                            self.tenant_char_count
                                .entry(Arc::clone(&tenant_id))
                                .and_modify(|c| *c += matched_count)
                                .or_insert(matched_count);
                        }
                        matched_node.tenant_last_access_time
                            .insert(Arc::clone(&tenant_id), timestamp_ms);

                        curr = matched_node;
                        curr_idx += shared;
                    }
                }
            }
        }
    }

    pub fn prefix_match(&self, text: &str) -> (String, String) {
        if text.is_empty() {
            return (String::new(), "empty".to_string());
        }

        // Pre-index text
        let chars: Vec<char> = text.chars().collect();
        let text_count = chars.len();

        let mut curr = Arc::clone(&self.root);
        let mut curr_idx = 0;

        while curr_idx < text_count {
            let first_char = chars[curr_idx];

            // Clone the matched node immediately to release the DashMap borrow
            let matched_node = match curr.children.get(&first_char) {
                Some(entry) => entry.value().clone(),
                None => break, // No match found
            };

            let matched_text = &matched_node.text;
            let matched_count = matched_text.char_count();

            // Count shared prefix (NO LOCK - text is immutable!)
            let mut shared = 0;
            while shared < matched_count && curr_idx + shared < text_count {
                if matched_text.get_char(shared) != Some(chars[curr_idx + shared]) {
                    break;
                }
                shared += 1;
            }

            curr_idx += shared;
            curr = matched_node;

            if shared < matched_count {
                // Partial match, stop
                break;
            }
        }

        // Get first tenant
        let tenant: Option<TenantId> = curr
            .tenant_last_access_time
            .iter()
            .next()
            .map(|kv| Arc::clone(kv.key()));

        // OPTIMIZATION: Probabilistic timestamp update
        if should_update_timestamps() {
            if let Some(ref tenant_id) = tenant {
                let timestamp_ms = get_timestamp_ms();
                let mut current_node = Some(Arc::clone(&curr));
                while let Some(node) = current_node {
                    node.tenant_last_access_time
                        .insert(Arc::clone(tenant_id), timestamp_ms);
                    current_node = node.get_parent();
                }
            }
        }

        let matched_text: String = chars[..curr_idx].iter().collect();
        let tenant_str = tenant
            .map(|t| t.to_string())
            .unwrap_or_else(|| "empty".to_string());

        (matched_text, tenant_str)
    }

    pub fn prefix_match_tenant(&self, text: &str, tenant: &str) -> String {
        if text.is_empty() {
            return String::new();
        }

        let chars: Vec<char> = text.chars().collect();
        let text_count = chars.len();
        let tenant_id: TenantId = Arc::from(tenant);

        let mut curr = Arc::clone(&self.root);
        let mut curr_idx = 0;

        while curr_idx < text_count {
            let first_char = chars[curr_idx];

            // Clone the matched node immediately to release the DashMap borrow
            let matched_node = match curr.children.get(&first_char) {
                Some(entry) => entry.value().clone(),
                None => break,
            };

            // Check tenant ownership
            if !matched_node
                .tenant_last_access_time
                .contains_key(tenant_id.as_ref())
            {
                break;
            }

            let matched_text = &matched_node.text;
            let matched_count = matched_text.char_count();

            let mut shared = 0;
            while shared < matched_count && curr_idx + shared < text_count {
                if matched_text.get_char(shared) != Some(chars[curr_idx + shared]) {
                    break;
                }
                shared += 1;
            }

            curr_idx += shared;
            curr = matched_node;

            if shared < matched_count {
                break;
            }
        }

        // Probabilistic timestamp update
        if should_update_timestamps()
            && curr
                .tenant_last_access_time
                .contains_key(tenant_id.as_ref())
        {
            let timestamp_ms = get_timestamp_ms();
            let mut current_node = Some(curr);
            while let Some(node) = current_node {
                node.tenant_last_access_time
                    .insert(Arc::clone(&tenant_id), timestamp_ms);
                current_node = node.get_parent();
            }
        }

        chars[..curr_idx].iter().collect()
    }

    pub fn get_used_size_per_tenant(&self) -> HashMap<String, usize> {
        self.tenant_char_count
            .iter()
            .map(|entry| (entry.key().to_string(), *entry.value()))
            .collect()
    }

    pub fn remove_tenant(&self, tenant: &str) {
        let tenant_id: TenantId = Arc::from(tenant);

        // BFS to find and remove tenant from all nodes
        let mut stack = vec![Arc::clone(&self.root)];

        while let Some(curr) = stack.pop() {
            curr.tenant_last_access_time.remove(tenant_id.as_ref());

            for child in curr.children.iter() {
                stack.push(child.value().clone());
            }
        }

        self.tenant_char_count.remove(tenant_id.as_ref());
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_basic_insert_match() {
        let tree = OptimizedTree::new();

        tree.insert("hello", "tenant1");
        tree.insert("world", "tenant2");

        let (matched, tenant) = tree.prefix_match("hello");
        assert_eq!(matched, "hello");
        assert_eq!(tenant, "tenant1");

        let (matched, tenant) = tree.prefix_match("world");
        assert_eq!(matched, "world");
        assert_eq!(tenant, "tenant2");
    }

    #[test]
    fn test_prefix_sharing() {
        let tree = OptimizedTree::new();

        tree.insert("apple", "tenant1");
        tree.insert("application", "tenant1");
        tree.insert("apply", "tenant2");

        let (matched, _) = tree.prefix_match("application");
        assert_eq!(matched, "application");

        let (matched, _) = tree.prefix_match("app");
        assert_eq!(matched, "app");
    }

    #[test]
    fn test_concurrent_insert() {
        let tree = Arc::new(OptimizedTree::new());
        let mut handles = vec![];

        for t in 0..4 {
            let tree = Arc::clone(&tree);
            handles.push(thread::spawn(move || {
                let tenant = format!("tenant{}", t);
                for i in 0..1000 {
                    tree.insert(&format!("prefix{}_suffix{}", t, i), &tenant);
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        // Verify all tenants have data
        let sizes = tree.get_used_size_per_tenant();
        for t in 0..4 {
            let tenant = format!("tenant{}", t);
            assert!(sizes.contains_key(&tenant), "Missing tenant {}", tenant);
        }
    }

    #[test]
    fn test_concurrent_read_write() {
        let tree = Arc::new(OptimizedTree::new());

        // Pre-populate
        for i in 0..100 {
            tree.insert(&format!("data_{}", i), "tenant1");
        }

        let mut handles = vec![];

        // Writers
        for t in 0..2 {
            let tree = Arc::clone(&tree);
            handles.push(thread::spawn(move || {
                let tenant = format!("writer{}", t);
                for i in 0..500 {
                    tree.insert(&format!("write_{}_data_{}", t, i), &tenant);
                }
            }));
        }

        // Readers
        for _ in 0..4 {
            let tree = Arc::clone(&tree);
            handles.push(thread::spawn(move || {
                for i in 0..1000 {
                    let _ = tree.prefix_match(&format!("data_{}", i % 100));
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn test_utf8_handling() {
        let tree = OptimizedTree::new();

        tree.insert("你好世界", "tenant1");
        tree.insert("你好嗎", "tenant2");

        let (matched, _) = tree.prefix_match("你好世界");
        assert_eq!(matched, "你好世界");

        let (matched, _) = tree.prefix_match("你好");
        assert_eq!(matched, "你好");
    }
}
