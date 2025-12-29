//! Request events for observability and monitoring.
//!
//! # Performance Characteristics
//!
//! - **Zero heap allocations** in emit path (uses borrowed strings)
//! - **Single atomic load** per emit to check OTEL status
//! - **Inline functions** for all hot paths
//! - **Static dispatch** via trait - no dynamic dispatch overhead
//!
//! # Usage
//!
//! Events accept borrowed string slices to avoid allocation:
//!
//! ```ignore
//! RequestSentEvent { url: &worker_url }.emit();
//! ```
//!
//! The event structs are stack-allocated and immediately dropped after emit.

use tracing::{debug, event, Level};

use super::otel_trace::is_otel_enabled;

/// Module path used by CustomOtelFilter to identify events for OTEL export.
/// Returns a compile-time constant string.
#[inline]
pub const fn get_module_path() -> &'static str {
    // Note: module_path!() is evaluated at compile time
    "sgl_model_gateway::observability::events"
}

/// Trait for emitting observability events.
///
/// # Performance
///
/// All implementations are `#[inline]` to enable the compiler to
/// optimize away the trait dispatch and potentially the entire
/// event emission when logging is disabled.
pub trait Event {
    /// Emit this event to the configured logging/tracing backends.
    fn emit(&self);
}

/// Event emitted when a prefill-decode request pair is sent.
///
/// # Performance
///
/// Uses borrowed string slices - caller retains ownership, no allocation.
/// Size: 2 pointers (16 bytes on 64-bit) + 2 lengths (16 bytes) = 32 bytes
#[derive(Debug, Clone, Copy)]
pub struct RequestPDSentEvent<'a> {
    pub prefill_url: &'a str,
    pub decode_url: &'a str,
}

impl Event for RequestPDSentEvent<'_> {
    #[inline]
    fn emit(&self) {
        // Single atomic load with Relaxed ordering - minimal CPU overhead
        if is_otel_enabled() {
            event!(
                Level::INFO,
                prefill_url = %self.prefill_url,
                decode_url = %self.decode_url,
                "Sending concurrent requests"
            );
        } else {
            debug!(
                prefill_url = %self.prefill_url,
                decode_url = %self.decode_url,
                "Sending concurrent requests"
            );
        }
    }
}

/// Event emitted when a request is sent to a worker.
///
/// # Performance
///
/// Uses borrowed string slice - zero allocation.
/// Size: 1 pointer (8 bytes) + 1 length (8 bytes) = 16 bytes
#[derive(Debug, Clone, Copy)]
pub struct RequestSentEvent<'a> {
    pub url: &'a str,
}

impl Event for RequestSentEvent<'_> {
    #[inline]
    fn emit(&self) {
        if is_otel_enabled() {
            event!(Level::INFO, url = %self.url, "Sending request");
        } else {
            debug!(url = %self.url, "Sending request");
        }
    }
}

/// Event emitted when concurrent requests are received.
///
/// # Performance
///
/// Zero-sized type - no memory overhead.
/// The compiler can optimize this to a simple function call.
#[derive(Debug, Clone, Copy)]
pub struct RequestReceivedEvent;

impl Event for RequestReceivedEvent {
    #[inline]
    fn emit(&self) {
        if is_otel_enabled() {
            event!(Level::INFO, "Received concurrent requests");
        } else {
            debug!("Received concurrent requests");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::size_of;

    #[test]
    fn test_event_sizes() {
        // Verify our size assumptions
        assert_eq!(
            size_of::<RequestReceivedEvent>(),
            0,
            "RequestReceivedEvent should be ZST"
        );
        assert_eq!(
            size_of::<RequestSentEvent>(),
            16,
            "RequestSentEvent should be 16 bytes"
        );
        assert_eq!(
            size_of::<RequestPDSentEvent>(),
            32,
            "RequestPDSentEvent should be 32 bytes"
        );
    }
}
