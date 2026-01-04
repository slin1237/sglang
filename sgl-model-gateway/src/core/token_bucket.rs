use std::{
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use tokio::sync::Notify;
use tracing::{debug, trace};

/// Fixed-point precision for token storage (1 token = 1_000_000 units)
const TOKEN_PRECISION: f64 = 1_000_000.0;

/// Token bucket for rate limiting using lock-free atomics.
///
/// This implementation provides:
/// - Lock-free token acquisition using atomic CAS operations
/// - Smooth rate limiting with configurable refill rate
/// - Burst capacity handling
/// - Fair queuing for waiting requests via Notify
///
/// # Performance
/// All operations are lock-free, using atomic compare-and-swap for token
/// acquisition. This eliminates mutex contention under high concurrency.
#[derive(Clone)]
pub struct TokenBucket {
    /// Available tokens in fixed-point format (actual_tokens * TOKEN_PRECISION)
    tokens_fp: Arc<AtomicU64>,
    /// Last refill time as nanoseconds since `created_at`
    last_refill_nanos: Arc<AtomicU64>,
    /// Reference instant for time calculations
    created_at: Instant,
    /// Notification for waiters when tokens are returned
    notify: Arc<Notify>,
    /// Maximum capacity in fixed-point format
    capacity_fp: u64,
    /// Refill rate in fixed-point tokens per nanosecond
    refill_per_nano_fp: f64,
    /// Whether refill is enabled (false when refill_rate=0)
    refill_enabled: bool,
}

impl TokenBucket {
    /// Create a new token bucket
    ///
    /// # Arguments
    /// * `capacity` - Maximum number of tokens (burst capacity)
    /// * `refill_rate` - Tokens added per second (0 for pure concurrency limiting)
    pub fn new(capacity: usize, refill_rate: usize) -> Self {
        let capacity_f64 = capacity as f64;
        let refill_rate_f64 = refill_rate as f64;
        let capacity_fp = (capacity_f64 * TOKEN_PRECISION) as u64;

        // Convert refill rate from tokens/sec to fixed-point tokens/nanosecond
        // refill_per_nano = (tokens/sec) * PRECISION / 1_000_000_000
        let refill_per_nano_fp = refill_rate_f64 * TOKEN_PRECISION / 1_000_000_000.0;

        Self {
            tokens_fp: Arc::new(AtomicU64::new(capacity_fp)),
            last_refill_nanos: Arc::new(AtomicU64::new(0)),
            created_at: Instant::now(),
            notify: Arc::new(Notify::new()),
            capacity_fp,
            refill_per_nano_fp,
            refill_enabled: refill_rate > 0,
        }
    }

    /// Get current time as nanoseconds since bucket creation
    #[inline]
    fn now_nanos(&self) -> u64 {
        self.created_at.elapsed().as_nanos() as u64
    }

    /// Perform refill and return new token count (in fixed-point).
    /// Uses CAS to atomically update last_refill_nanos.
    #[inline]
    fn refill_tokens(&self) -> u64 {
        if !self.refill_enabled {
            return self.tokens_fp.load(Ordering::Acquire);
        }

        let now_nanos = self.now_nanos();
        let last_nanos = self.last_refill_nanos.load(Ordering::Acquire);

        if now_nanos <= last_nanos {
            return self.tokens_fp.load(Ordering::Acquire);
        }

        let elapsed_nanos = now_nanos - last_nanos;
        let refill_amount_fp = (elapsed_nanos as f64 * self.refill_per_nano_fp) as u64;

        if refill_amount_fp == 0 {
            return self.tokens_fp.load(Ordering::Acquire);
        }

        // Try to update last_refill_nanos - if we lose the race, another thread did the refill
        if self
            .last_refill_nanos
            .compare_exchange_weak(last_nanos, now_nanos, Ordering::AcqRel, Ordering::Relaxed)
            .is_ok()
        {
            // We won the race, add the refill amount
            let mut current = self.tokens_fp.load(Ordering::Acquire);
            loop {
                let new_tokens = current
                    .saturating_add(refill_amount_fp)
                    .min(self.capacity_fp);
                match self.tokens_fp.compare_exchange_weak(
                    current,
                    new_tokens,
                    Ordering::AcqRel,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => {
                        trace!(
                            "Token bucket: refilled {} tokens, now {} available",
                            refill_amount_fp as f64 / TOKEN_PRECISION,
                            new_tokens as f64 / TOKEN_PRECISION
                        );
                        return new_tokens;
                    }
                    Err(actual) => current = actual,
                }
            }
        }

        // Lost the race, just return current tokens
        self.tokens_fp.load(Ordering::Acquire)
    }

    /// Try to acquire tokens immediately (lock-free).
    ///
    /// Returns `Ok(())` if tokens were acquired, `Err(())` if insufficient tokens.
    pub async fn try_acquire(&self, tokens: f64) -> Result<(), ()> {
        let tokens_fp = (tokens * TOKEN_PRECISION) as u64;

        // First, perform any pending refill
        self.refill_tokens();

        // Try to acquire using CAS
        let mut current = self.tokens_fp.load(Ordering::Acquire);
        loop {
            trace!(
                "Token bucket: {} tokens available, requesting {}",
                current as f64 / TOKEN_PRECISION,
                tokens
            );

            if current < tokens_fp {
                return Err(());
            }

            let new_tokens = current - tokens_fp;
            match self.tokens_fp.compare_exchange_weak(
                current,
                new_tokens,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    debug!(
                        "Token bucket: acquired {} tokens, {} remaining",
                        tokens,
                        new_tokens as f64 / TOKEN_PRECISION
                    );
                    return Ok(());
                }
                Err(actual) => {
                    // Another thread modified tokens, retry with new value
                    current = actual;
                }
            }
        }
    }

    /// Acquire tokens, waiting if necessary.
    ///
    /// When `refill_rate=0`, waits indefinitely for tokens to be returned via `return_tokens()`.
    /// Use `acquire_timeout()` to set an appropriate timeout.
    pub async fn acquire(&self, tokens: f64) -> Result<(), tokio::time::error::Elapsed> {
        if self.try_acquire(tokens).await.is_ok() {
            return Ok(());
        }

        // When refill_rate=0 (pure concurrency limiting), tokens only come back
        // via return_tokens(), so we wait on notify signal only.
        if !self.refill_enabled {
            debug!(
                "Token bucket: waiting indefinitely for {} tokens (refill_rate=0)",
                tokens
            );

            loop {
                // Wait for notify signal from return_tokens()
                self.notify.notified().await;

                if self.try_acquire(tokens).await.is_ok() {
                    return Ok(());
                }
            }
        }

        // Calculate estimated wait time based on current state
        let tokens_fp = (tokens * TOKEN_PRECISION) as u64;
        let current = self.tokens_fp.load(Ordering::Acquire);
        let tokens_needed_fp = tokens_fp.saturating_sub(current);
        let wait_nanos = if self.refill_per_nano_fp > 0.0 {
            (tokens_needed_fp as f64 / self.refill_per_nano_fp) as u64
        } else {
            u64::MAX
        };
        let wait_time = Duration::from_nanos(wait_nanos.min(60_000_000_000)); // Cap at 60s

        debug!(
            "Token bucket: waiting {:?} for {} tokens",
            wait_time, tokens
        );

        tokio::time::timeout(wait_time, async {
            loop {
                if self.try_acquire(tokens).await.is_ok() {
                    return;
                }
                // Wait for notification (from return_tokens) - no polling
                self.notify.notified().await;
            }
        })
        .await?;

        Ok(())
    }

    /// Acquire tokens with custom timeout
    pub async fn acquire_timeout(
        &self,
        tokens: f64,
        timeout: Duration,
    ) -> Result<(), tokio::time::error::Elapsed> {
        tokio::time::timeout(timeout, self.acquire(tokens)).await?
    }

    /// Return tokens to the bucket (lock-free).
    ///
    /// This is safe to call from sync contexts (e.g., Drop handlers).
    pub fn return_tokens_sync(&self, tokens: f64) {
        let tokens_fp = (tokens * TOKEN_PRECISION) as u64;

        let mut current = self.tokens_fp.load(Ordering::Acquire);
        loop {
            let new_tokens = current.saturating_add(tokens_fp).min(self.capacity_fp);
            match self.tokens_fp.compare_exchange_weak(
                current,
                new_tokens,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    debug!(
                        "Token bucket: returned {} tokens, {} available",
                        tokens,
                        new_tokens as f64 / TOKEN_PRECISION
                    );
                    self.notify.notify_waiters();
                    return;
                }
                Err(actual) => current = actual,
            }
        }
    }

    /// Return tokens to the bucket (async version for API compatibility).
    pub async fn return_tokens(&self, tokens: f64) {
        self.return_tokens_sync(tokens);
    }

    /// Get current available tokens (for monitoring).
    ///
    /// This performs a refill calculation, so the returned value reflects
    /// tokens that would be available if acquired now.
    pub async fn available_tokens(&self) -> f64 {
        self.refill_tokens();
        self.tokens_fp.load(Ordering::Acquire) as f64 / TOKEN_PRECISION
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_token_bucket_basic() {
        let bucket = TokenBucket::new(10, 5);

        assert!(bucket.try_acquire(5.0).await.is_ok());
        assert!(bucket.try_acquire(5.0).await.is_ok());

        assert!(bucket.try_acquire(1.0).await.is_err());

        tokio::time::sleep(Duration::from_millis(300)).await;

        assert!(bucket.try_acquire(1.0).await.is_ok());
    }

    #[tokio::test]
    async fn test_token_bucket_refill() {
        let bucket = TokenBucket::new(10, 10);

        assert!(bucket.try_acquire(10.0).await.is_ok());

        tokio::time::sleep(Duration::from_millis(500)).await;

        let available = bucket.available_tokens().await;
        assert!((4.0..=6.0).contains(&available));
    }

    #[tokio::test]
    async fn test_token_bucket_zero_refill_rate() {
        // With refill_rate=0, tokens should only come back via return_tokens()
        let bucket = TokenBucket::new(2, 0);

        // Acquire both tokens
        assert!(bucket.try_acquire(1.0).await.is_ok());
        assert!(bucket.try_acquire(1.0).await.is_ok());

        // No more tokens available
        assert!(bucket.try_acquire(1.0).await.is_err());

        // Wait - should NOT refill automatically
        tokio::time::sleep(Duration::from_millis(500)).await;
        assert!(bucket.try_acquire(1.0).await.is_err());

        // Return a token - now we should be able to acquire
        bucket.return_tokens(1.0).await;
        assert!(bucket.try_acquire(1.0).await.is_ok());

        // No more tokens again
        assert!(bucket.try_acquire(1.0).await.is_err());
    }

    #[tokio::test]
    async fn test_token_bucket_zero_refill_with_notify() {
        // Test that acquire wakes up when tokens are returned
        let bucket = Arc::new(TokenBucket::new(1, 0));

        // Acquire the only token
        assert!(bucket.try_acquire(1.0).await.is_ok());

        let bucket_clone = bucket.clone();

        // Spawn a task that will return the token after a delay
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(100)).await;
            bucket_clone.return_tokens(1.0).await;
        });

        // This should wait and then succeed when token is returned
        let result = bucket.acquire_timeout(1.0, Duration::from_secs(1)).await;
        assert!(result.is_ok());
    }
}
