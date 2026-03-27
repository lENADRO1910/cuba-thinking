// src/server/observability.rs
//
// RED Metrics Module — Rate, Errors, Duration
//
// Provides lightweight, lock-free observability for a stdio-based MCP server.
// Since there is no HTTP endpoint, metrics are emitted via structured tracing
// logs on server shutdown and can be queried via the `snapshot()` method.
//
// Design:
// - AtomicU64 for lock-free, thread-safe counters
// - Per-tool breakdown via DashMap-free approach (pre-registered tool set)
// - Relaxed ordering: eventual consistency is acceptable for metrics

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// RED metrics for a single tool (or aggregate).
#[derive(Debug)]
pub struct ToolMetrics {
    /// Total requests processed (Rate).
    pub requests: AtomicU64,
    /// Total errors returned (Errors).
    pub errors: AtomicU64,
    /// Sum of durations in microseconds (Duration).
    pub duration_us: AtomicU64,
    /// Max duration in microseconds (p100).
    pub max_duration_us: AtomicU64,
}

impl ToolMetrics {
    pub const fn new() -> Self {
        Self {
            requests: AtomicU64::new(0),
            errors: AtomicU64::new(0),
            duration_us: AtomicU64::new(0),
            max_duration_us: AtomicU64::new(0),
        }
    }

    pub fn record(&self, duration: Duration, is_error: bool) {
        self.requests.fetch_add(1, Ordering::Relaxed);
        if is_error {
            self.errors.fetch_add(1, Ordering::Relaxed);
        }
        let us = duration.as_micros() as u64;
        self.duration_us.fetch_add(us, Ordering::Relaxed);
        // CAS loop for max
        let mut current_max = self.max_duration_us.load(Ordering::Relaxed);
        while us > current_max {
            match self.max_duration_us.compare_exchange_weak(
                current_max,
                us,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => current_max = actual,
            }
        }
    }

    pub fn snapshot(&self) -> ToolSnapshot {
        let requests = self.requests.load(Ordering::Relaxed);
        let errors = self.errors.load(Ordering::Relaxed);
        let duration_us = self.duration_us.load(Ordering::Relaxed);
        let max_duration_us = self.max_duration_us.load(Ordering::Relaxed);
        let avg_duration_us = if requests > 0 {
            duration_us / requests
        } else {
            0
        };
        ToolSnapshot {
            requests,
            errors,
            error_rate: if requests > 0 {
                errors as f64 / requests as f64
            } else {
                0.0
            },
            avg_duration_ms: avg_duration_us as f64 / 1000.0,
            max_duration_ms: max_duration_us as f64 / 1000.0,
        }
    }
}

/// Immutable snapshot of tool metrics for logging.
#[derive(Debug, Clone)]
pub struct ToolSnapshot {
    pub requests: u64,
    pub errors: u64,
    pub error_rate: f64,
    pub avg_duration_ms: f64,
    pub max_duration_ms: f64,
}

/// Aggregate RED metrics across all tools, plus per-tool breakdown.
pub struct RedMetrics {
    /// Aggregate across all tools.
    pub total: ToolMetrics,
    /// Per-tool metrics (static set: cuba_thinking, verify_code, analyze_reasoning, stress).
    pub cuba_thinking: ToolMetrics,
    pub verify_code: ToolMetrics,
    pub analyze_reasoning: ToolMetrics,
    pub stress_benchmark: ToolMetrics,
    /// Server boot timestamp.
    boot_time: Instant,
}

impl RedMetrics {
    pub fn new() -> Self {
        Self {
            total: ToolMetrics::new(),
            cuba_thinking: ToolMetrics::new(),
            verify_code: ToolMetrics::new(),
            analyze_reasoning: ToolMetrics::new(),
            stress_benchmark: ToolMetrics::new(),
            boot_time: Instant::now(),
        }
    }

    /// Record a tool call's outcome.
    pub fn record_call(&self, tool_name: &str, duration: Duration, is_error: bool) {
        self.total.record(duration, is_error);
        match tool_name {
            "cuba_thinking" => self.cuba_thinking.record(duration, is_error),
            "verify_code" => self.verify_code.record(duration, is_error),
            "analyze_reasoning" => self.analyze_reasoning.record(duration, is_error),
            "run_stress_benchmark" => self.stress_benchmark.record(duration, is_error),
            _ => {} // Unknown tools still counted in total
        }
    }

    /// Emit RED summary via structured tracing (stderr).
    pub fn emit_summary(&self) {
        let uptime = self.boot_time.elapsed();
        let total = self.total.snapshot();
        let rps = if uptime.as_secs() > 0 {
            total.requests as f64 / uptime.as_secs() as f64
        } else {
            total.requests as f64
        };

        tracing::info!(
            target: "red_metrics",
            uptime_secs = uptime.as_secs(),
            requests_total = total.requests,
            errors_total = total.errors,
            error_rate = format!("{:.4}", total.error_rate),
            avg_duration_ms = format!("{:.2}", total.avg_duration_ms),
            max_duration_ms = format!("{:.2}", total.max_duration_ms),
            requests_per_sec = format!("{:.2}", rps),
            "RED metrics summary"
        );

        // Per-tool breakdown (only if tool was called)
        for (name, metrics) in [
            ("cuba_thinking", &self.cuba_thinking),
            ("verify_code", &self.verify_code),
            ("analyze_reasoning", &self.analyze_reasoning),
            ("run_stress_benchmark", &self.stress_benchmark),
        ] {
            let snap = metrics.snapshot();
            if snap.requests > 0 {
                tracing::info!(
                    target: "red_metrics",
                    tool = name,
                    requests = snap.requests,
                    errors = snap.errors,
                    error_rate = format!("{:.4}", snap.error_rate),
                    avg_ms = format!("{:.2}", snap.avg_duration_ms),
                    max_ms = format!("{:.2}", snap.max_duration_ms),
                    "RED tool breakdown"
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_metrics_basic() {
        let m = ToolMetrics::new();
        m.record(Duration::from_millis(100), false);
        m.record(Duration::from_millis(200), false);
        m.record(Duration::from_millis(50), true);
        let snap = m.snapshot();
        assert_eq!(snap.requests, 3);
        assert_eq!(snap.errors, 1);
        assert!((snap.error_rate - 1.0 / 3.0).abs() < 0.01);
        // avg should be ~116ms
        assert!(snap.avg_duration_ms > 100.0 && snap.avg_duration_ms < 130.0);
        // max should be ~200ms
        assert!(snap.max_duration_ms > 195.0 && snap.max_duration_ms < 205.0);
    }

    #[test]
    fn test_tool_metrics_empty() {
        let m = ToolMetrics::new();
        let snap = m.snapshot();
        assert_eq!(snap.requests, 0);
        assert_eq!(snap.errors, 0);
        assert_eq!(snap.error_rate, 0.0);
        assert_eq!(snap.avg_duration_ms, 0.0);
    }

    #[test]
    fn test_red_metrics_per_tool() {
        let r = RedMetrics::new();
        r.record_call("cuba_thinking", Duration::from_millis(50), false);
        r.record_call("verify_code", Duration::from_millis(10), true);
        r.record_call("cuba_thinking", Duration::from_millis(100), false);
        r.record_call("unknown_tool", Duration::from_millis(5), false);

        let total = r.total.snapshot();
        assert_eq!(total.requests, 4);
        assert_eq!(total.errors, 1);

        let ct = r.cuba_thinking.snapshot();
        assert_eq!(ct.requests, 2);
        assert_eq!(ct.errors, 0);

        let vc = r.verify_code.snapshot();
        assert_eq!(vc.requests, 1);
        assert_eq!(vc.errors, 1);

        // Unknown tool not counted per-tool, but IS in total
        let ar = r.analyze_reasoning.snapshot();
        assert_eq!(ar.requests, 0);
    }

    #[test]
    fn test_tool_metrics_zero_duration() {
        let m = ToolMetrics::new();
        m.record(Duration::ZERO, false);
        let snap = m.snapshot();
        assert_eq!(snap.requests, 1);
        assert_eq!(snap.errors, 0);
        assert_eq!(snap.avg_duration_ms, 0.0);
        assert_eq!(snap.max_duration_ms, 0.0);
    }

    #[test]
    fn test_analyze_reasoning_zero_duration() {
        let r = RedMetrics::new();
        // Missing/empty context scenario often results in near-zero duration
        r.record_call("analyze_reasoning", Duration::ZERO, false);

        let ar = r.analyze_reasoning.snapshot();
        assert_eq!(ar.requests, 1);
        assert_eq!(ar.avg_duration_ms, 0.0);

        let total = r.total.snapshot();
        assert_eq!(total.requests, 1);
    }
}
