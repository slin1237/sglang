//! Observability utilities for logging, metrics, and tracing.
//!
//! # Performance Design Principles
//!
//! This module is designed for **minimal CPU and memory overhead** on the hot path:
//!
//! ## Zero-Allocation Hot Paths
//!
//! - **Event emission**: Uses borrowed `&str` references instead of owned `String`
//! - **Logging**: Static format strings, pre-calculated filter capacities
//! - **OTEL checks**: Single `AtomicBool` load with `Relaxed` ordering
//!
//! ## Compile-Time Optimizations
//!
//! - **`#[inline]`** on all frequently-called functions
//! - **`const fn`** where possible for compile-time evaluation
//! - **Zero-sized types** for events with no payload (`RequestReceivedEvent`)
//! - **Static callsite caching** via `callsite_enabled` in OTEL filter
//!
//! ## Memory Efficiency
//!
//! - **Stack allocation** for event structs (16-32 bytes max)
//! - **Static strings** for log format, default targets, allowed OTEL targets
//! - **Pre-allocated vectors** with known capacity
//!
//! # Module Structure
//!
//! - [`events`]: Request lifecycle events with zero-allocation emit
//! - [`logging`]: Tracing subscriber setup with file/console/OTEL layers
//! - [`metrics`]: Prometheus metrics collection (separate performance profile)
//! - [`otel_trace`]: OpenTelemetry span export and context propagation

pub mod events;
pub mod logging;
pub mod metrics;
pub mod otel_trace;
