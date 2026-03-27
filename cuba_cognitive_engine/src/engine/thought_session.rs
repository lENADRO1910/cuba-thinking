// src/engine/thought_session.rs
//
// Phase 5A: Persistent Thought Sessions (G1, G6, G10, G11)
//
// Maintains cognitive state across multiple tool calls in a reasoning chain.
// Before this module, EwmaTracker, NoveltyTracker, and ThoughtGraph were
// re-created from scratch on every call — making cross-step analysis meaningless.
//
// Now, the same session persists across calls sharing the same hypothesis,
// enabling:
// - G1: Accumulated EWMA quality tracking
// - G6: Cross-call thought accumulation (previous thoughts available for coherence)
// - G10: Quality trend visualization (↗️ improving, →️ stable, ↘️ declining)
// - G11: Hypothesis drift detection (semantic distance from original hypothesis)
//
// Sessions auto-expire after TTL_SECONDS to prevent memory leaks.

use crate::engine::budget::BudgetMode;
use crate::engine::ewma_reward::EwmaTracker;
use crate::engine::novelty_tracker::NoveltyTracker;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Session TTL: 10 minutes.
const TTL_SECONDS: u64 = 600;

/// Quality trend indicator based on EWMA history.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub enum TrendIndicator {
    /// ↗️ Quality is improving (last 2 steps ascending)
    Improving,
    /// →️ Quality is stable (less than 3% change)
    Stable,
    /// ↘️ Quality is declining (last 2 steps descending)
    Declining,
    /// 🔄 Not enough data (fewer than 2 steps)
    Insufficient,
}

impl TrendIndicator {
    /// Compute trend from EWMA reward history.
    pub fn from_ewma(ewma: &EwmaTracker) -> Self {
        if ewma.reward_history.len() < 2 {
            return TrendIndicator::Insufficient;
        }

        let history = &ewma.reward_history;
        let last = history[history.len() - 1];
        let prev = history[history.len() - 2];
        let delta = last - prev;

        if delta > 0.03 {
            TrendIndicator::Improving
        } else if delta < -0.03 {
            TrendIndicator::Declining
        } else {
            TrendIndicator::Stable
        }
    }

    /// English label for display.
    pub fn label(&self) -> &'static str {
        match self {
            TrendIndicator::Improving => "Improving",
            TrendIndicator::Stable => "Stable",
            TrendIndicator::Declining => "Declining",
            TrendIndicator::Insufficient => "Insufficient data",
        }
    }
}

/// Maximum number of thoughts to retain per session (ring buffer).
const MAX_THOUGHTS: usize = 20;

/// Persistent cognitive state for a reasoning chain.
pub struct ThoughtSession {
    /// EWMA quality tracker — accumulated across calls.
    pub ewma: EwmaTracker,
    /// Novelty tracker — vocabulary grows across calls.
    pub novelty: NoveltyTracker,
    /// Recent thought texts — capped at MAX_THOUGHTS (ring buffer).
    pub thoughts: VecDeque<Arc<str>>,
    /// Total thoughts recorded (monotonic, not capped).
    total_thoughts: usize,
    /// The original hypothesis text — for drift detection (G11).
    pub original_hypothesis: Arc<str>,
    /// Hash of the hypothesis — session key.
    _hypothesis_hash: [u8; 32],
    /// When this session was created.
    created_at: Instant,
    /// When this session was last accessed.
    last_accessed: Instant,
    /// Current quality trend.
    pub trend: TrendIndicator,
    /// NEW-1: Confidence history for oscillation detection.
    pub confidence_history: VecDeque<f64>,
    /// Vector A: First thought text for root-anchoring.
    pub first_thought: Option<Arc<str>>,
}

impl ThoughtSession {
    /// Create a new session for a hypothesis.
    pub fn new(hypothesis: &str, budget: BudgetMode) -> Self {
        Self {
            ewma: EwmaTracker::new(budget),
            novelty: NoveltyTracker::new(),
            thoughts: VecDeque::with_capacity(MAX_THOUGHTS),
            total_thoughts: 0,
            original_hypothesis: hypothesis.into(),
            _hypothesis_hash: compute_hypothesis_hash(hypothesis),
            created_at: Instant::now(),
            last_accessed: Instant::now(),
            trend: TrendIndicator::Insufficient,
            confidence_history: VecDeque::with_capacity(MAX_THOUGHTS),
            first_thought: None,
        }
    }

    /// Record a new thought in this session.
    /// Returns the total thought index (0-based, monotonic).
    pub fn record_thought(&mut self, thought: &str) -> usize {
        let idx = self.total_thoughts;

        // Ring buffer: evict oldest if at capacity
        if self.thoughts.len() >= MAX_THOUGHTS {
            self.thoughts.pop_front();
        }

        // Zero-Copy PRM Guard: Allocate once and copy pointers O(1)
        let arc_thought: Arc<str> = thought.into();
        self.thoughts.push_back(arc_thought.clone());
        self.total_thoughts += 1;

        // Vector A: Capture first thought for root-anchoring
        if self.first_thought.is_none() {
            self.first_thought = Some(arc_thought.clone());
        }

        // Update trend
        self.trend = TrendIndicator::from_ewma(&self.ewma);

        // Touch last accessed
        self.last_accessed = Instant::now();

        idx
    }

    /// Get previous thought texts for coherence analysis.
    /// Returns the last N thoughts (or all if fewer than N).
    pub fn previous_thoughts(&self, n: usize) -> Vec<&str> {
        self.thoughts
            .iter()
            .rev()
            .take(n)
            .map(|s| &**s)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect()
    }

    /// Detect hypothesis drift (G11).
    /// Returns drift score in [0.0, 1.0]:
    /// - 0.0 = no drift (hypothesis unchanged)
    /// - 1.0 = complete drift (completely different topic)
    pub fn hypothesis_drift(&self, current_hypothesis: &str) -> f64 {
        use crate::engine::semantic_similarity;
        let coherence = semantic_similarity::compute_coherence(
            current_hypothesis,
            Some(&self.original_hypothesis),
        );
        // Drift is inverse of coherence
        (1.0 - coherence).clamp(0.0, 1.0)
    }

    /// Vector A: Combined drift — max of hypothesis drift and root-thought drift.
    ///
    /// Catches two failure modes:
    /// 1. Current hypothesis diverges from original hypothesis (existing G11)
    /// 2. Current thought diverges from the first thought in the chain
    ///    (root-anchoring — even if hypothesis text stays the same)
    pub fn combined_drift(&self, current_hypothesis: &str, current_thought: &str) -> f64 {
        let hyp_drift = self.hypothesis_drift(current_hypothesis);

        let root_drift = if let Some(ref first) = self.first_thought {
            use crate::engine::semantic_similarity;
            let coh = semantic_similarity::compute_coherence(current_thought, Some(first));
            (1.0 - coh).clamp(0.0, 1.0)
        } else {
            0.0
        };

        hyp_drift.max(root_drift)
    }

    /// Check if this session has expired.
    pub fn is_expired(&self) -> bool {
        self.last_accessed.elapsed().as_secs() > TTL_SECONDS
    }

    /// Number of thoughts retained in ring buffer.
    pub fn thought_count(&self) -> usize {
        self.thoughts.len()
    }

    /// NEW-1: Record declared confidence for this step.
    pub fn record_confidence(&mut self, confidence: f64) {
        if self.confidence_history.len() >= MAX_THOUGHTS {
            self.confidence_history.pop_front();
        }
        self.confidence_history.push_back(confidence);
    }

    /// NEW-1: Detect confidence oscillation — sign changes in Δconfidence.
    /// 3+ sign changes in last 5 readings indicates model confusion.
    /// Based on derivative sign analysis of the confidence time series.
    pub fn is_confidence_oscillating(&self) -> bool {
        if self.confidence_history.len() < 4 {
            return false;
        }
        let slice: Vec<f64> = self
            .confidence_history
            .iter()
            .rev()
            .take(5)
            .copied()
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        let deltas: Vec<f64> = slice.windows(2).map(|w| w[1] - w[0]).collect();
        let sign_changes = deltas
            .windows(2)
            .filter(|w| w[0].signum() != w[1].signum() && w[0].abs() > 0.05)
            .count();
        sign_changes >= 2
    }
}

/// Thread-safe session store with TTL-based cleanup.
///
/// # Concurrency Note (F3)
///
/// Uses `std::sync::Mutex` (not `tokio::sync::Mutex`) because:
/// 1. All closures passed to `with_session` are sync and O(μs) —
///    no I/O, no awaits, just in-memory data manipulation.
/// 2. Current transport is STDIO (single client, zero contention).
/// 3. `tokio::sync::Mutex` would require `async FnOnce` closures
///    (not yet stable in Rust) or a complete API redesign.
///
/// **IMPORTANT**: If a concurrent transport (REST/WebSocket) is added,
/// this MUST be migrated to `tokio::sync::Mutex` or `DashMap` to prevent
/// blocking the tokio worker thread pool under contention.
///
/// Maximum number of active cognitive sessions.
/// Reduced from 1000 to 50: stdio transport = 1 client, 50 is more than sufficient.
const MAX_ACTIVE_SESSIONS: usize = 50;
pub struct SessionStore {
    sessions: Mutex<HashMap<[u8; 32], ThoughtSession>>,
    /// Atomic counter for periodic TTL cleanup (every 10 calls instead of every lock).
    access_counter: AtomicUsize,
}

impl SessionStore {
    pub fn new() -> Self {
        Self {
            sessions: Mutex::new(HashMap::new()),
            access_counter: AtomicUsize::new(0),
        }
    }

    /// Get or create a session for the given hypothesis.
    /// Returns a closure-based access pattern to avoid holding the lock.
    pub fn with_session<F, R>(&self, hypothesis: &str, budget: BudgetMode, f: F) -> R
    where
        F: FnOnce(&mut ThoughtSession) -> R,
    {
        let hash = compute_hypothesis_hash(hypothesis);
        let mut sessions = self.sessions.lock().unwrap_or_else(|e| {
            tracing::warn!("SessionStore mutex was poisoned in with_session(), recovering");
            e.into_inner()
        });

        // Cleanup expired sessions every 10 calls (not every lock acquisition)
        let count = self.access_counter.fetch_add(1, Ordering::Relaxed);
        if count % 10 == 0 {
            sessions.retain(|_, session| !session.is_expired());
        }

        // Enforce capacity limit to prevent unbounded memory growth (DoS protection)
        if !sessions.contains_key(&hash) && sessions.len() >= MAX_ACTIVE_SESSIONS {
            // Find and remove the oldest session
            if let Some(oldest_key) = sessions
                .iter()
                .min_by_key(|(_, session)| session.last_accessed)
                .map(|(k, _)| *k)
            {
                sessions.remove(&oldest_key);
            }
        }

        // Get or create
        let session = sessions
            .entry(hash)
            .or_insert_with(|| ThoughtSession::new(hypothesis, budget));

        // Touch last accessed
        session.last_accessed = Instant::now();

        f(session)
    }

    /// Number of active sessions.
    pub fn active_count(&self) -> usize {
        let sessions = self.sessions.lock().unwrap_or_else(|e| {
            tracing::warn!("SessionStore mutex was poisoned in active_count(), recovering");
            e.into_inner()
        });
        sessions.values().filter(|s| !s.is_expired()).count()
    }
}

/// Compute a deterministic hash for a hypothesis string.
/// Used as session key — same hypothesis = same session.
fn compute_hypothesis_hash(hypothesis: &str) -> [u8; 32] {
    let normalized = hypothesis.trim().to_lowercase();
    let mut hasher = Sha256::new();
    hasher.update(normalized.as_bytes());
    hasher.finalize().into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_creation() {
        let session = ThoughtSession::new("Test hypothesis", BudgetMode::Balanced);
        assert_eq!(session.thought_count(), 0);
        assert_eq!(session.trend, TrendIndicator::Insufficient);
        assert!(!session.is_expired());
    }

    #[test]
    fn test_record_thought() {
        let mut session = ThoughtSession::new("Test hypothesis", BudgetMode::Balanced);
        let idx = session.record_thought("First thought about database");
        assert_eq!(idx, 0);
        assert_eq!(session.thought_count(), 1);

        let idx = session.record_thought("Second thought about caching");
        assert_eq!(idx, 1);
        assert_eq!(session.thought_count(), 2);
    }

    #[test]
    fn test_previous_thoughts() {
        let mut session = ThoughtSession::new("Hypothesis", BudgetMode::Balanced);
        session.record_thought("Alpha");
        session.record_thought("Beta");
        session.record_thought("Gamma");

        let prev = session.previous_thoughts(2);
        assert_eq!(prev.len(), 2);
        assert_eq!(prev[0], "Beta");
        assert_eq!(prev[1], "Gamma");
    }

    #[test]
    fn test_hypothesis_drift_no_change() {
        let session = ThoughtSession::new("Database migration strategy", BudgetMode::Balanced);
        let drift = session.hypothesis_drift("Database migration strategy");
        assert!(
            drift < 0.2,
            "Same hypothesis should have low drift: {:.3}",
            drift
        );
    }

    #[test]
    fn test_hypothesis_drift_significant_change() {
        let session = ThoughtSession::new(
            "Database migration with PostgreSQL and zero downtime",
            BudgetMode::Balanced,
        );
        let drift =
            session.hypothesis_drift("Frontend animation performance optimization using WebGL");
        assert!(
            drift > 0.3,
            "Completely different hypothesis should have high drift: {:.3}",
            drift
        );
    }

    #[test]
    fn test_session_store_get_or_create() {
        let store = SessionStore::new();
        let result = store.with_session("Test hypothesis", BudgetMode::Balanced, |session| {
            session.record_thought("First thought");
            session.thought_count()
        });
        assert_eq!(result, 1);

        // Same hypothesis should return same session
        let result = store.with_session("Test hypothesis", BudgetMode::Balanced, |session| {
            session.record_thought("Second thought");
            session.thought_count()
        });
        assert_eq!(result, 2);
    }

    #[test]
    fn test_session_store_different_hypotheses() {
        let store = SessionStore::new();
        store.with_session("Hypothesis A", BudgetMode::Balanced, |session| {
            session.record_thought("Thought for A");
        });
        store.with_session("Hypothesis B", BudgetMode::Balanced, |session| {
            session.record_thought("Thought for B");
        });
        assert_eq!(store.active_count(), 2);
    }

    #[test]
    fn test_hypothesis_hash_deterministic() {
        let h1 = compute_hypothesis_hash("Test hypothesis");
        let h2 = compute_hypothesis_hash("Test hypothesis");
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_hypothesis_hash_case_insensitive() {
        let h1 = compute_hypothesis_hash("Test Hypothesis");
        let h2 = compute_hypothesis_hash("test hypothesis");
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_trend_insufficient() {
        let ewma = EwmaTracker::new(BudgetMode::Balanced);
        let trend = TrendIndicator::from_ewma(&ewma);
        assert_eq!(trend, TrendIndicator::Insufficient);
    }

    #[test]
    fn test_trend_improving() {
        let mut ewma = EwmaTracker::new(BudgetMode::Balanced);
        use crate::engine::ewma_reward::RewardSignals;
        ewma.update(&RewardSignals {
            quality: 0.3,
            ..Default::default()
        });
        ewma.update(&RewardSignals {
            quality: 0.9,
            ..Default::default()
        });
        let trend = TrendIndicator::from_ewma(&ewma);
        assert_eq!(trend, TrendIndicator::Improving);
    }

    #[test]
    fn test_trend_declining() {
        let mut ewma = EwmaTracker::new(BudgetMode::Balanced);
        use crate::engine::ewma_reward::RewardSignals;
        ewma.update(&RewardSignals {
            quality: 0.9,
            ..Default::default()
        });
        ewma.update(&RewardSignals {
            quality: 0.2,
            ..Default::default()
        });
        let trend = TrendIndicator::from_ewma(&ewma);
        assert_eq!(trend, TrendIndicator::Declining);
    }

    // ─── NEW-1: Confidence Oscillation Tests ──────────

    #[test]
    fn test_confidence_oscillation_detection() {
        let mut session = ThoughtSession::new("hypothesis", BudgetMode::Balanced);
        // Rapidly alternating confidence: model is confused
        session.record_confidence(0.9);
        session.record_confidence(0.3);
        session.record_confidence(0.8);
        session.record_confidence(0.2);
        session.record_confidence(0.7);
        assert!(
            session.is_confidence_oscillating(),
            "Should detect oscillation in rapidly alternating confidence"
        );
    }

    #[test]
    fn test_no_oscillation_monotonic() {
        let mut session = ThoughtSession::new("hypothesis", BudgetMode::Balanced);
        // Steadily increasing confidence: not oscillating
        session.record_confidence(0.3);
        session.record_confidence(0.5);
        session.record_confidence(0.7);
        session.record_confidence(0.9);
        assert!(
            !session.is_confidence_oscillating(),
            "Monotonic confidence should not trigger oscillation"
        );
    }

    #[test]
    fn test_no_oscillation_too_few_readings() {
        let mut session = ThoughtSession::new("hypothesis", BudgetMode::Balanced);
        session.record_confidence(0.9);
        session.record_confidence(0.2);
        assert!(
            !session.is_confidence_oscillating(),
            "Too few readings should not trigger oscillation"
        )
    }

    // ─── Vector A: Root-Anchoring Tests ───────────────

    #[test]
    fn test_first_thought_captured() {
        let mut session = ThoughtSession::new("hypothesis", BudgetMode::Balanced);
        session.record_thought("My first thought about databases");
        session.record_thought("My second thought about caching");
        assert_eq!(
            session.first_thought.as_deref(),
            Some("My first thought about databases")
        );
    }

    #[test]
    fn test_combined_drift_uses_max() {
        let mut session =
            ThoughtSession::new("Database migration zero downtime", BudgetMode::Balanced);
        session.record_thought("Database schema migration plan with rollback");
        // current_thought completely off-topic from first-thought
        let drift = session.combined_drift(
            "Database migration zero downtime", // hypothesis unchanged
            "Frontend CSS animation performance tuning with WebGL", // drifted
        );
        assert!(
            drift > 0.3,
            "Combined drift should detect topic change: {:.3}",
            drift
        );
    }

}
