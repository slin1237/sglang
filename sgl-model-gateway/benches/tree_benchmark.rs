//! Benchmarks for the radix tree implementation used in cache-aware routing.
//!
//! This benchmark simulates realistic cache-aware routing scenarios with:
//! - Multiple tenants representing HTTP/gRPC endpoints (10 endpoints)
//! - High-pressure workloads with concurrent operations
//! - Realistic request text patterns (system prompts, user queries, etc.)
//!
//! Run with: cargo bench --bench tree_benchmark
//!
//! For quick validation (CI): cargo bench --bench tree_benchmark -- benchmark_summary --exact

use std::{sync::Arc, thread};

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::{
    distr::{Alphanumeric, SampleString},
    rng as thread_rng, Rng,
};
// Import the tree module
use sgl_model_gateway::policies::tree::Tree;

/// Simulated HTTP/gRPC endpoints representing worker nodes
/// These mirror real-world deployment patterns with 10 tenants
const ENDPOINT_TENANTS: [&str; 10] = [
    "http://worker-0.sglang.svc.cluster.local:8000",
    "http://worker-1.sglang.svc.cluster.local:8000",
    "http://worker-2.sglang.svc.cluster.local:8000",
    "http://worker-3.sglang.svc.cluster.local:8000",
    "http://worker-4.sglang.svc.cluster.local:8000",
    "grpc://worker-5.sglang.svc.cluster.local:50051",
    "grpc://worker-6.sglang.svc.cluster.local:50051",
    "grpc://worker-7.sglang.svc.cluster.local:50051",
    "http://10.0.0.100:8000",
    "http://10.0.0.101:8000",
];

/// Realistic system prompts used in LLM applications
const SYSTEM_PROMPTS: [&str; 5] = [
    "You are a helpful assistant that provides accurate and concise answers.",
    "You are a coding expert. Help the user write clean, efficient code.",
    "You are a creative writer. Generate engaging and original content.",
    "You are a data analyst. Provide insights based on the given data.",
    "You are a customer support agent. Be polite and helpful.",
];

/// Common conversation prefixes that create shared tree paths
const CONVERSATION_PREFIXES: [&str; 6] = [
    "<|system|>\nYou are a helpful assistant.\n<|user|>\n",
    "<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n<|im_start|>user\n",
    "[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\n",
    "Human: ",
    "User: ",
    "### Instruction:\n",
];

/// Generate random ASCII strings of given length
fn random_ascii_string(len: usize) -> String {
    Alphanumeric.sample_string(&mut thread_rng(), len)
}

/// Generate random strings with common prefixes (simulates real request patterns)
fn random_prefixed_strings(prefix: &str, suffix_len: usize, count: usize) -> Vec<String> {
    (0..count)
        .map(|_| format!("{}{}", prefix, random_ascii_string(suffix_len)))
        .collect()
}

/// Generate realistic LLM request texts with system prompts and user queries
fn generate_realistic_requests(count: usize) -> Vec<String> {
    let mut rng = thread_rng();
    (0..count)
        .map(|_| {
            let prefix_idx = rng.random_range(0..CONVERSATION_PREFIXES.len());
            let query_len = rng.random_range(20..200);
            format!(
                "{}{}",
                CONVERSATION_PREFIXES[prefix_idx],
                random_ascii_string(query_len)
            )
        })
        .collect()
}

/// Get a random endpoint tenant
fn random_endpoint() -> &'static str {
    let mut rng = thread_rng();
    ENDPOINT_TENANTS[rng.random_range(0..ENDPOINT_TENANTS.len())]
}

/// Benchmark single-threaded insert throughput with endpoint tenants
fn bench_insert_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert_throughput");

    for text_len in [10, 50, 100, 500].iter() {
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("random_text", text_len),
            text_len,
            |b, &len| {
                let tree = Tree::new();
                let strings: Vec<String> = (0..1000).map(|_| random_ascii_string(len)).collect();
                let mut idx = 0;

                b.iter(|| {
                    let tenant = ENDPOINT_TENANTS[idx % ENDPOINT_TENANTS.len()];
                    tree.insert(black_box(&strings[idx % strings.len()]), tenant);
                    idx += 1;
                });
            },
        );
    }

    // Benchmark with shared prefixes (common cache scenario) - distributed across endpoints
    group.bench_function("shared_prefix_100", |b| {
        let tree = Tree::new();
        let prefixes = ["system:", "user:", "assistant:", "tool:"];
        let strings: Vec<String> = prefixes
            .iter()
            .flat_map(|p| random_prefixed_strings(p, 50, 250))
            .collect();
        let mut idx = 0;

        b.iter(|| {
            let tenant = ENDPOINT_TENANTS[idx % ENDPOINT_TENANTS.len()];
            tree.insert(black_box(&strings[idx % strings.len()]), tenant);
            idx += 1;
        });
    });

    // Benchmark with realistic LLM request patterns
    group.bench_function("realistic_llm_requests", |b| {
        let tree = Tree::new();
        let requests = generate_realistic_requests(2000);
        let mut idx = 0;

        b.iter(|| {
            let tenant = ENDPOINT_TENANTS[idx % ENDPOINT_TENANTS.len()];
            tree.insert(black_box(&requests[idx % requests.len()]), tenant);
            idx += 1;
        });
    });

    group.finish();
}

/// Benchmark prefix_match latency with multi-tenant tree
fn bench_prefix_match_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("prefix_match_latency");

    // Setup: pre-populate tree with data distributed across all endpoints
    let tree = Tree::new();
    let prefixes = ["system:", "user:", "assistant:", "tool:"];
    let strings: Vec<String> = prefixes
        .iter()
        .flat_map(|p| random_prefixed_strings(p, 50, 1000))
        .collect();

    // Distribute entries across all 10 endpoint tenants
    for (i, s) in strings.iter().enumerate() {
        let tenant = ENDPOINT_TENANTS[i % ENDPOINT_TENANTS.len()];
        tree.insert(s, tenant);
    }

    // Benchmark cache hit (exact match)
    group.bench_function("cache_hit", |b| {
        let mut idx = 0;
        b.iter(|| {
            let result = tree.prefix_match(black_box(&strings[idx % strings.len()]));
            idx += 1;
            result
        });
    });

    // Benchmark cache miss (no match)
    let miss_strings: Vec<String> = (0..1000).map(|_| random_ascii_string(50)).collect();
    group.bench_function("cache_miss", |b| {
        let mut idx = 0;
        b.iter(|| {
            let result = tree.prefix_match(black_box(&miss_strings[idx % miss_strings.len()]));
            idx += 1;
            result
        });
    });

    // Benchmark partial match
    group.bench_function("partial_match", |b| {
        let partial_strings: Vec<String> = prefixes
            .iter()
            .map(|p| format!("{}partial_query", p))
            .collect();
        let mut idx = 0;
        b.iter(|| {
            let result =
                tree.prefix_match(black_box(&partial_strings[idx % partial_strings.len()]));
            idx += 1;
            result
        });
    });

    group.finish();
}

/// Benchmark concurrent operations with high pressure (10 endpoint tenants)
fn bench_concurrent_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent");
    group.sample_size(50); // Reduce sample size for concurrent tests

    // Mixed read/write workload with endpoint-style tenants
    for num_threads in [2, 4, 8, 16].iter() {
        group.bench_with_input(
            BenchmarkId::new("mixed_workload", num_threads),
            num_threads,
            |b, &threads| {
                b.iter(|| {
                    let tree = Arc::new(Tree::new());
                    let handles: Vec<_> = (0..threads)
                        .map(|t| {
                            let tree = Arc::clone(&tree);
                            thread::spawn(move || {
                                // Each thread uses a different endpoint tenant
                                let tenant = ENDPOINT_TENANTS[t % ENDPOINT_TENANTS.len()];
                                for i in 0..200 {
                                    let text = format!(
                                        "{}thread{}_request{}",
                                        CONVERSATION_PREFIXES[i % CONVERSATION_PREFIXES.len()],
                                        t,
                                        i
                                    );
                                    if i % 3 == 0 {
                                        tree.prefix_match(&text);
                                    } else {
                                        tree.insert(&text, tenant);
                                    }
                                }
                            })
                        })
                        .collect();

                    for h in handles {
                        h.join().unwrap();
                    }
                });
            },
        );
    }

    // High-contention scenario: all threads sharing same prefixes
    group.bench_function("high_contention_10_tenants", |b| {
        b.iter(|| {
            let tree = Arc::new(Tree::new());
            let handles: Vec<_> = (0..10)
                .map(|t| {
                    let tree = Arc::clone(&tree);
                    thread::spawn(move || {
                        let tenant = ENDPOINT_TENANTS[t];
                        // All threads insert similar prefixes to create contention
                        for i in 0..100 {
                            let text = format!(
                                "<|system|>\nYou are a helpful assistant.\n<|user|>\nQuery {}",
                                i
                            );
                            tree.insert(&text, tenant);
                            tree.prefix_match(&text);
                        }
                    })
                })
                .collect();

            for h in handles {
                h.join().unwrap();
            }
        });
    });

    group.finish();
}

/// Benchmark eviction performance with multi-tenant scenarios
fn bench_eviction(c: &mut Criterion) {
    let mut group = c.benchmark_group("eviction");
    group.sample_size(20); // Eviction is expensive

    for tree_size in [1000, 5000, 10000].iter() {
        group.bench_with_input(
            BenchmarkId::new("evict_to_half_single_tenant", tree_size),
            tree_size,
            |b, &size| {
                b.iter_with_setup(
                    || {
                        // Setup: create tree with many entries for single tenant
                        let tree = Tree::new();
                        let tenant = ENDPOINT_TENANTS[0];
                        for i in 0..size {
                            tree.insert(&format!("entry_{:05}", i), tenant);
                        }
                        tree
                    },
                    |tree| {
                        // Evict to half size
                        tree.evict_tenant_by_size(size / 2);
                    },
                );
            },
        );
    }

    // Multi-tenant eviction: 10 tenants with overlapping data
    for tree_size in [1000, 5000, 10000].iter() {
        group.bench_with_input(
            BenchmarkId::new("evict_multi_tenant_10", tree_size),
            tree_size,
            |b, &size| {
                b.iter_with_setup(
                    || {
                        // Setup: create tree with entries distributed across 10 tenants
                        let tree = Tree::new();
                        for i in 0..size {
                            let tenant = ENDPOINT_TENANTS[i % ENDPOINT_TENANTS.len()];
                            // Use shared prefixes to create overlapping tree structure
                            let prefix = CONVERSATION_PREFIXES[i % CONVERSATION_PREFIXES.len()];
                            tree.insert(&format!("{}entry_{:05}", prefix, i), tenant);
                        }
                        tree
                    },
                    |tree| {
                        // Evict to target size per tenant
                        tree.evict_tenant_by_size(size / 20); // Much smaller target to trigger more eviction
                    },
                );
            },
        );
    }

    group.finish();
}

/// Benchmark UTF-8 handling vs ASCII with multiple endpoint tenants
fn bench_utf8_vs_ascii(c: &mut Criterion) {
    let mut group = c.benchmark_group("encoding");

    let tree_ascii = Tree::new();
    let tree_utf8 = Tree::new();

    // Pre-populate with data distributed across endpoints
    let ascii_strings: Vec<String> = (0..1000).map(|_| random_ascii_string(50)).collect();
    let utf8_strings: Vec<String> = (0..1000).map(|i| format!("你好世界_{}", i)).collect();

    for (i, s) in ascii_strings.iter().enumerate() {
        let tenant = ENDPOINT_TENANTS[i % ENDPOINT_TENANTS.len()];
        tree_ascii.insert(s, tenant);
    }
    for (i, s) in utf8_strings.iter().enumerate() {
        let tenant = ENDPOINT_TENANTS[i % ENDPOINT_TENANTS.len()];
        tree_utf8.insert(s, tenant);
    }

    group.bench_function("ascii_match", |b| {
        let mut idx = 0;
        b.iter(|| {
            let result =
                tree_ascii.prefix_match(black_box(&ascii_strings[idx % ascii_strings.len()]));
            idx += 1;
            result
        });
    });

    group.bench_function("utf8_match", |b| {
        let mut idx = 0;
        b.iter(|| {
            let result = tree_utf8.prefix_match(black_box(&utf8_strings[idx % utf8_strings.len()]));
            idx += 1;
            result
        });
    });

    group.finish();
}

/// Benchmark multi-tenant scenarios with 10 HTTP/gRPC endpoint tenants
fn bench_multi_tenant(c: &mut Criterion) {
    let mut group = c.benchmark_group("multi_tenant");

    let tree = Tree::new();

    // Setup: 10 endpoint tenants with overlapping data patterns
    let prefixes = ["prompt:", "completion:", "context:", "system:", "user:"];

    for tenant in &ENDPOINT_TENANTS {
        for prefix in &prefixes {
            for i in 0..200 {
                tree.insert(&format!("{}data_{}", prefix, i), tenant);
            }
        }
    }

    group.bench_function("shared_prefix_lookup_10_tenants", |b| {
        let queries: Vec<String> = prefixes
            .iter()
            .flat_map(|p| (0..50).map(move |i| format!("{}data_{}", p, i)))
            .collect();
        let mut idx = 0;

        b.iter(|| {
            let result = tree.prefix_match(black_box(&queries[idx % queries.len()]));
            idx += 1;
            result
        });
    });

    group.bench_function("tenant_specific_match_10_tenants", |b| {
        let queries: Vec<(String, &str)> = ENDPOINT_TENANTS
            .iter()
            .flat_map(|&t| (0..20).map(move |i| (format!("prompt:data_{}", i), t)))
            .collect();
        let mut idx = 0;

        b.iter(|| {
            let (query, tenant) = &queries[idx % queries.len()];
            let result = tree.prefix_match_tenant(black_box(query), black_box(tenant));
            idx += 1;
            result
        });
    });

    // Benchmark tenant removal (simulates worker going offline)
    group.bench_function("tenant_removal", |b| {
        b.iter_with_setup(
            || {
                // Setup: create tree with all endpoints
                let tree = Tree::new();
                for tenant in &ENDPOINT_TENANTS {
                    for prefix in &prefixes {
                        for i in 0..100 {
                            tree.insert(&format!("{}data_{}", prefix, i), tenant);
                        }
                    }
                }
                tree
            },
            |tree| {
                // Remove one tenant (simulates worker going offline)
                tree.remove_tenant(ENDPOINT_TENANTS[0]);
            },
        );
    });

    group.finish();
}

/// Benchmark summary for CI - runs a subset of representative benchmarks
fn bench_summary(c: &mut Criterion) {
    let mut group = c.benchmark_group("benchmark_summary");

    // Representative insert benchmark
    group.bench_function("insert_realistic", |b| {
        let tree = Tree::new();
        let requests = generate_realistic_requests(1000);
        let mut idx = 0;

        b.iter(|| {
            let tenant = ENDPOINT_TENANTS[idx % ENDPOINT_TENANTS.len()];
            tree.insert(black_box(&requests[idx % requests.len()]), tenant);
            idx += 1;
        });
    });

    // Representative lookup benchmark
    {
        let tree = Tree::new();
        let requests = generate_realistic_requests(1000);
        for (i, req) in requests.iter().enumerate() {
            let tenant = ENDPOINT_TENANTS[i % ENDPOINT_TENANTS.len()];
            tree.insert(req, tenant);
        }

        group.bench_function("prefix_match_realistic", |b| {
            let mut idx = 0;
            b.iter(|| {
                let result = tree.prefix_match(black_box(&requests[idx % requests.len()]));
                idx += 1;
                result
            });
        });
    }

    // Representative concurrent benchmark
    group.sample_size(30);
    group.bench_function("concurrent_mixed_8_threads", |b| {
        b.iter(|| {
            let tree = Arc::new(Tree::new());
            let handles: Vec<_> = (0..8)
                .map(|t| {
                    let tree = Arc::clone(&tree);
                    thread::spawn(move || {
                        let tenant = ENDPOINT_TENANTS[t % ENDPOINT_TENANTS.len()];
                        for i in 0..100 {
                            let text = format!(
                                "{}thread{}_request{}",
                                CONVERSATION_PREFIXES[i % CONVERSATION_PREFIXES.len()],
                                t,
                                i
                            );
                            if i % 3 == 0 {
                                tree.prefix_match(&text);
                            } else {
                                tree.insert(&text, tenant);
                            }
                        }
                    })
                })
                .collect();

            for h in handles {
                h.join().unwrap();
            }
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_insert_throughput,
    bench_prefix_match_latency,
    bench_concurrent_operations,
    bench_eviction,
    bench_utf8_vs_ascii,
    bench_multi_tenant,
    bench_summary,
);
criterion_main!(benches);
