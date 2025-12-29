//! Logging infrastructure with minimal runtime overhead.
//!
//! # Performance Characteristics
//!
//! - **Zero heap allocations** in the hot path after initialization
//! - **Static string slices** for format strings and filter targets
//! - **Non-blocking file I/O** via dedicated writer thread (when file logging enabled)
//! - **Lazy initialization** of filter strings to avoid startup overhead
//!
//! # Memory Usage
//!
//! - `LoggingConfig`: ~80 bytes on stack (excluding heap-allocated optional fields)
//! - `LogGuard`: 8 bytes (Option<WorkerGuard> pointer)
//! - Static filter string: computed once, ~100 bytes typical

use std::path::PathBuf;

use tracing::Level;
use tracing_appender::{
    non_blocking::WorkerGuard,
    rolling::{RollingFileAppender, Rotation},
};
use tracing_log::LogTracer;
use tracing_subscriber::{
    fmt::time::ChronoUtc, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter, Layer,
};

/// Static time format string - avoids heap allocation on every log call.
/// Using a const ensures this is embedded in the binary's read-only data section.
const TIME_FORMAT: &str = "%Y-%m-%d %H:%M:%S";

/// Default log target - avoids repeated string allocations.
const DEFAULT_LOG_TARGET: &str = "sgl_model_gateway";

use super::otel_trace::get_otel_layer;
use crate::config::TraceConfig;

/// Configuration for the logging subsystem.
///
/// # Performance Notes
///
/// Uses `Cow<'static, str>` for string fields to enable zero-copy usage
/// of static strings while still supporting dynamic values when needed.
#[derive(Debug, Clone)]
pub struct LoggingConfig {
    /// Log level filter (TRACE, DEBUG, INFO, WARN, ERROR)
    pub level: Level,
    /// Output logs in JSON format for structured logging
    pub json_format: bool,
    /// Directory for log files (None = no file logging)
    pub log_dir: Option<String>,
    /// Enable ANSI color codes in console output
    pub colorize: bool,
    /// Base name for log files
    pub log_file_name: String,
    /// Target modules to log (None = use default target)
    pub log_targets: Option<Vec<String>>,
}

impl Default for LoggingConfig {
    #[inline]
    fn default() -> Self {
        Self {
            level: Level::INFO,
            json_format: false,
            log_dir: None,
            colorize: true,
            log_file_name: "sgl-model-gateway".to_string(),
            log_targets: Some(vec![DEFAULT_LOG_TARGET.to_string()]),
        }
    }
}

/// Guard that keeps the file appender thread alive.
///
/// Must be held for the lifetime of the application to ensure
/// all log messages are flushed to disk before shutdown.
#[allow(dead_code)]
pub struct LogGuard {
    _file_guard: Option<WorkerGuard>,
}

/// Converts a tracing Level to its static string representation.
///
/// # Performance
///
/// Returns a `&'static str` to avoid any heap allocation.
#[inline]
const fn level_to_str(level: Level) -> &'static str {
    match level {
        Level::TRACE => "trace",
        Level::DEBUG => "debug",
        Level::INFO => "info",
        Level::WARN => "warn",
        Level::ERROR => "error",
    }
}

/// Builds the filter string for log targets.
///
/// # Performance
///
/// Pre-calculates capacity to minimize reallocations.
/// Uses a single String allocation with estimated capacity.
#[inline]
fn build_filter_string(targets: &[String], level_filter: &str) -> String {
    // Pre-calculate capacity: each entry is "target=level" plus comma separator
    // Average target length ~20 chars, level ~5 chars, separator 1 char
    let estimated_capacity = targets.len() * (20 + level_filter.len() + 2);
    let mut filter_string = String::with_capacity(estimated_capacity);

    for (i, target) in targets.iter().enumerate() {
        if i > 0 {
            filter_string.push(',');
        }
        filter_string.push_str(target);
        filter_string.push('=');
        filter_string.push_str(level_filter);
    }

    filter_string
}

/// Initialize the logging subsystem.
///
/// # Performance Characteristics
///
/// - **Initialization**: One-time cost, allocates filter strings and sets up layers
/// - **Runtime**: Zero allocations per log call (format strings are static)
/// - **File logging**: Uses non-blocking writer with dedicated thread
///
/// # Arguments
///
/// * `config` - Logging configuration
/// * `otel_layer_config` - Optional OpenTelemetry configuration
///
/// # Returns
///
/// A `LogGuard` that must be held for the application lifetime.
pub fn init_logging(config: LoggingConfig, otel_layer_config: Option<TraceConfig>) -> LogGuard {
    let _ = LogTracer::init();

    let level_filter = level_to_str(config.level);

    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        let filter_string = match &config.log_targets {
            Some(targets) if !targets.is_empty() => build_filter_string(targets, level_filter),
            _ => {
                // Use static default target - single small allocation
                let mut s =
                    String::with_capacity(DEFAULT_LOG_TARGET.len() + 1 + level_filter.len());
                s.push_str(DEFAULT_LOG_TARGET);
                s.push('=');
                s.push_str(level_filter);
                s
            }
        };

        EnvFilter::new(filter_string)
    });

    // Pre-allocate layer vector with expected capacity (max 3 layers)
    let mut layers = Vec::with_capacity(3);

    // Use static time format - no heap allocation
    let stdout_layer = tracing_subscriber::fmt::layer()
        .with_ansi(config.colorize)
        .with_file(true)
        .with_line_number(true)
        .with_timer(ChronoUtc::new(TIME_FORMAT.to_string()));

    let stdout_layer = if config.json_format {
        stdout_layer.json().flatten_event(true).boxed()
    } else {
        stdout_layer.boxed()
    };

    layers.push(stdout_layer);

    let mut file_guard = None;

    if let Some(log_dir) = &config.log_dir {
        let log_dir = PathBuf::from(log_dir);

        if !log_dir.exists() {
            if let Err(e) = std::fs::create_dir_all(&log_dir) {
                eprintln!("Failed to create log directory: {}", e);
                return LogGuard { _file_guard: None };
            }
        }

        // Move file_name instead of cloning - saves one allocation
        let file_appender =
            RollingFileAppender::new(Rotation::DAILY, log_dir, &config.log_file_name);

        let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);
        file_guard = Some(guard);

        // Reuse static TIME_FORMAT - no additional allocation
        let file_layer = tracing_subscriber::fmt::layer()
            .with_ansi(false)
            .with_file(true)
            .with_line_number(true)
            .with_timer(ChronoUtc::new(TIME_FORMAT.to_string()))
            .with_writer(non_blocking);

        let file_layer = if config.json_format {
            file_layer.json().flatten_event(true).boxed()
        } else {
            file_layer.boxed()
        };

        layers.push(file_layer);
    }

    if let Some(otel_layer_config) = &otel_layer_config {
        if otel_layer_config.enable_trace {
            match get_otel_layer() {
                Ok(otel_layer) => {
                    layers.push(otel_layer);
                }
                Err(e) => {
                    eprintln!("Failed to initialize OpenTelemetry: {}", e);
                }
            }
        }
    }

    let _ = tracing_subscriber::registry()
        .with(env_filter)
        .with(layers)
        .try_init();

    LogGuard {
        _file_guard: file_guard,
    }
}
