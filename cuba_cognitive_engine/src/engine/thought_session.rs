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
use crate::engine::thought_graph::ThoughtGraph;
use serde::Serialize;
use sha2::{Sha256, Digest};
use std::collections::{HashMap, VecDeque};
use std::sync::{Mutex, Arc};
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

    /// Emoji representation for compact output.
    #[allow(dead_code)]
    pub fn emoji(&self) -> &'static str {
        match self {
            TrendIndicator::Improving => "↗️",
            TrendIndicator::Stable => "→",
            TrendIndicator::Declining => "↘️",
            TrendIndicator::Insufficient => "🔄",
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
    /// Graph-of-Thought — builds DAG across calls.
    pub graph: ThoughtGraph,
    /// Recent thought texts — capped at MAX_THOUGHTS (ring buffer).
    pub thoughts: VecDeque<Arc<str>>,
    /// Total thoughts recorded (monotonic, not capped).
    total_thoughts: usize,
    /// The original hypothesis text — for drift detection (G11).
    pub original_hypothesis: Arc<str>,
    /// Hash of the hypothesis — session key.
    #[allow(dead_code)]
    hypothesis_hash: [u8; 32],
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
    /// V5-1: Epistemological snapshots — maps thought_number -> valid thoughts length.
    /// Used to rollback state when MCTS rejects a thought, preventing state poisoning.
    thought_snapshots: HashMap<usize, usize>,
    /// V6-3: Depth score history — tracks quality.depth per thought for degradation detection.
    /// When depth drops >50% vs baseline (first 3 thoughts), attention is collapsing.
    pub depth_history: VecDeque<f64>,
    /// V7-2: Failed thought texts for mode collapse detection.
    /// When MCTS rejects a thought, its text is stored here.
    /// New thoughts are compared against these via Jaccard similarity
    /// to detect paraphrasing of rejected ideas (mode collapse).
    failed_thoughts: Vec<Arc<str>>,
}

impl ThoughtSession {
    /// Create a new session for a hypothesis.
    pub fn new(hypothesis: &str, budget: BudgetMode) -> Self {
        Self {
            ewma: EwmaTracker::new(budget),
            novelty: NoveltyTracker::new(),
            graph: ThoughtGraph::new(),
            thoughts: VecDeque::with_capacity(MAX_THOUGHTS),
            total_thoughts: 0,
            original_hypothesis: hypothesis.into(),
            hypothesis_hash: compute_hypothesis_hash(hypothesis),
            created_at: Instant::now(),
            last_accessed: Instant::now(),
            trend: TrendIndicator::Insufficient,
            confidence_history: VecDeque::with_capacity(MAX_THOUGHTS),
            first_thought: None,
            thought_snapshots: HashMap::new(),
            depth_history: VecDeque::with_capacity(MAX_THOUGHTS),
            failed_thoughts: Vec::with_capacity(5),
        }
    }

    /// Record a new thought in this session.
    /// Returns the total thought index (0-based, monotonic).
    pub fn record_thought(&mut self, thought: &str) -> usize {
        let idx = self.total_thoughts;

        // V5-1: Snapshot BEFORE mutation — allows rollback if MCTS rejects this thought
        self.thought_snapshots.insert(idx, self.thoughts.len());

        // F4: Prune stale snapshots — retain only entries for thoughts still in
        // the ring buffer window, preventing unbounded HashMap growth.
        // At most MAX_THOUGHTS snapshots survive (one per ring buffer slot).
        if self.thought_snapshots.len() > MAX_THOUGHTS * 2 {
            let min_thought = idx.saturating_sub(MAX_THOUGHTS);
            self.thought_snapshots.retain(|&k, _| k >= min_thought);
        }

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

        // Update graph
        self.graph.add_node(idx);
        if idx > 0 {
            self.graph.add_edge(idx - 1, idx);
        }

        // Update trend
        self.trend = TrendIndicator::from_ewma(&self.ewma);

        // Touch last accessed
        self.last_accessed = Instant::now();

        idx
    }

    /// V5-1: Epistemological rollback — rewind session to state before `target_thought`.
    ///
    /// When MCTS rejects a thought (low EWMA), the thought's content already
    /// pollutes the session's `thoughts` buffer, `confidence_history`, and
    /// `graph`. This method physically removes the dead branch's state,
    /// preventing hallucinated premises from poisoning future reasoning.
    ///
    /// Complexity: O(1) for truncate + O(n) for stage_history retain.
    #[allow(dead_code)]
    pub fn rollback_to_thought(&mut self, target_thought: usize) {
        if let Some(&valid_len) = self.thought_snapshots.get(&target_thought) {
            // Truncate thoughts buffer to pre-mutation length
            while self.thoughts.len() > valid_len {
                self.thoughts.pop_back();
            }
            self.total_thoughts = target_thought;

            // Trim confidence history to match
            while self.confidence_history.len() > target_thought {
                self.confidence_history.pop_back();
            }

            // Prune graph nodes from dead branches
            self.graph.prune_after(target_thought);

            // V6-3: Trim depth history to match
            while self.depth_history.len() > target_thought {
                self.depth_history.pop_back();
            }

            // Update trend from current state
            self.trend = TrendIndicator::from_ewma(&self.ewma);

            // Clean up stale snapshots
            self.thought_snapshots.retain(|&k, _| k <= target_thought);

            // V7-2: Clear failed thoughts on rollback — fresh start for new branch
            self.failed_thoughts.clear();
        }
    }

    /// V6-3: Record a depth score for the current thought.
    /// Called by the engine after computing quality metrics.
    #[allow(dead_code)]
    pub fn record_depth_score(&mut self, depth: f64) {
        if self.depth_history.len() >= MAX_THOUGHTS {
            self.depth_history.pop_front();
        }
        self.depth_history.push_back(depth);
    }

    /// V6-3: Detect attention collapse via depth degradation.
    ///
    /// Computes the baseline from the first 3 depth scores,
    /// then checks if the current depth has fallen >50% below it.
    ///
    /// Returns Some(degradation_ratio) when degradation detected, None otherwise.
    /// degradation_ratio is in (0.0, 1.0): 0.0 = no drop, 1.0 = total collapse.
    ///
    /// Based on: Press et al. (2022) "Train Short, Test Long" —
    /// LLMs lose syntactic depth as KV cache saturates.
    #[allow(dead_code)]
    pub fn depth_degradation(&self) -> Option<f64> {
        // Need at least 4 scores: 3 for baseline + 1 current
        if self.depth_history.len() < 4 {
            return None;
        }

        // Baseline: mean of first 3 depth scores
        let baseline: f64 = self.depth_history.iter()
            .take(3)
            .sum::<f64>() / 3.0;

        // Guard: if baseline is near-zero, no meaningful comparison
        if baseline < 0.05 {
            return None;
        }

        // Current: last depth score
        let current = *self.depth_history.back().unwrap();

        // Degradation ratio: how much depth has fallen vs baseline
        let ratio = 1.0 - (current / baseline);

        // Only report if degradation exceeds 50%
        if ratio > 0.50 {
            Some(ratio.clamp(0.0, 1.0))
        } else {
            None
        }
    }

    /// V7-2: Register a thought that was rejected by MCTS.
    ///
    /// Stores the text for future mode collapse detection.
    /// Capped at 5 entries to bound memory.
    #[allow(dead_code)]
    pub fn register_failed_thought(&mut self, text: &str) {
        if text.len() > 100_000 {
            return; // V21: Prevent memory exhaustion from giant strings
        }
        const MAX_FAILED: usize = 5;
        if self.failed_thoughts.len() >= MAX_FAILED {
            self.failed_thoughts.remove(0);
        }
        self.failed_thoughts.push(text.into());
    }

    /// V7-2: Detect mode collapse — LLM paraphrasing rejected thoughts.
    ///
    /// Computes Jaccard similarity between `new_thought` and each stored
    /// failed thought. If any similarity exceeds 0.6 (60% shared vocabulary),
    /// the LLM is likely rewriting the same rejected idea with synonyms.
    ///
    /// Returns `Some(max_similarity)` when mode collapse detected, `None` otherwise.
    ///
    /// Uses shared_utils::stopwords for consistent tokenization.
    /// Complexity: O(n·m) where n=failed thoughts, m=terms per thought.
    #[allow(dead_code)]
    pub fn is_mode_collapse(&self, new_thought: &str) -> Option<f64> {
        if self.failed_thoughts.is_empty() || new_thought.len() > 100_000 {
            return None; // V21: Prevent CPU exhaustion from huge inputs
        }

        let stopwords = crate::engine::shared_utils::stopwords();
        let new_terms: std::collections::HashSet<String> = new_thought
            .split_whitespace()
            .map(|w| w.chars().filter(|c| c.is_alphanumeric()).collect::<String>().to_lowercase())
            .filter(|w| w.len() > 2 && !stopwords.contains(w.as_str()))
            .collect();

        if new_terms.is_empty() {
            return None;
        }

        let mut max_similarity = 0.0_f64;

        for failed_text in &self.failed_thoughts {
            let failed_terms: std::collections::HashSet<String> = failed_text
                .split_whitespace()
                .map(|w| w.chars().filter(|c| c.is_alphanumeric()).collect::<String>().to_lowercase())
                .filter(|w| w.len() > 2 && !stopwords.contains(w.as_str()))
                .collect();

            if failed_terms.is_empty() {
                continue;
            }

            // Jaccard similarity: |A ∩ B| / |A ∪ B|
            let intersection = new_terms.intersection(&failed_terms).count() as f64;
            let union = new_terms.union(&failed_terms).count() as f64;

            if union > 0.0 {
                let sim = intersection / union;
                max_similarity = max_similarity.max(sim);
            }
        }

        // Threshold: >0.6 similarity = mode collapse (paraphrasing)
        if max_similarity > 0.6 {
            Some(max_similarity)
        } else {
            None
        }
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

    /// Total thoughts ever recorded (monotonic counter).
    #[allow(dead_code)]
    pub fn total_thought_count(&self) -> usize {
        self.total_thoughts
    }

    /// Session age in seconds.
    #[allow(dead_code)]
    pub fn age_seconds(&self) -> u64 {
        self.created_at.elapsed().as_secs()
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
        let slice: Vec<f64> = self.confidence_history.iter()
            .rev().take(5).copied().collect::<Vec<_>>().into_iter().rev().collect();
        let deltas: Vec<f64> = slice.windows(2).map(|w| w[1] - w[0]).collect();
        let sign_changes = deltas.windows(2)
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
pub struct SessionStore {
    sessions: Mutex<HashMap<[u8; 32], ThoughtSession>>,
}

impl SessionStore {
    pub fn new() -> Self {
        Self {
            sessions: Mutex::new(HashMap::new()),
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

        // Cleanup expired sessions (opportunistic)
        sessions.retain(|_, session| !session.is_expired());

        // Get or create
        let session = sessions
            .entry(hash)
            .or_insert_with(|| ThoughtSession::new(hypothesis, budget));

        // Touch last accessed
        session.last_accessed = Instant::now();

        f(session)
    }

    /// Number of active sessions.
    #[allow(dead_code)]
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
        assert!(drift < 0.2, "Same hypothesis should have low drift: {:.3}", drift);
    }

    #[test]
    fn test_hypothesis_drift_significant_change() {
        let session = ThoughtSession::new(
            "Database migration with PostgreSQL and zero downtime",
            BudgetMode::Balanced,
        );
        let drift = session.hypothesis_drift(
            "Frontend animation performance optimization using WebGL"
        );
        assert!(drift > 0.3, "Completely different hypothesis should have high drift: {:.3}", drift);
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
        ewma.update(&RewardSignals { quality: 0.3, ..Default::default() });
        ewma.update(&RewardSignals { quality: 0.9, ..Default::default() });
        let trend = TrendIndicator::from_ewma(&ewma);
        assert_eq!(trend, TrendIndicator::Improving);
    }

    #[test]
    fn test_trend_declining() {
        let mut ewma = EwmaTracker::new(BudgetMode::Balanced);
        use crate::engine::ewma_reward::RewardSignals;
        ewma.update(&RewardSignals { quality: 0.9, ..Default::default() });
        ewma.update(&RewardSignals { quality: 0.2, ..Default::default() });
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
        assert!(session.is_confidence_oscillating(),
            "Should detect oscillation in rapidly alternating confidence");
    }

    #[test]
    fn test_no_oscillation_monotonic() {
        let mut session = ThoughtSession::new("hypothesis", BudgetMode::Balanced);
        // Steadily increasing confidence: not oscillating
        session.record_confidence(0.3);
        session.record_confidence(0.5);
        session.record_confidence(0.7);
        session.record_confidence(0.9);
        assert!(!session.is_confidence_oscillating(),
            "Monotonic confidence should not trigger oscillation");
    }

    #[test]
    fn test_no_oscillation_too_few_readings() {
        let mut session = ThoughtSession::new("hypothesis", BudgetMode::Balanced);
        session.record_confidence(0.9);
        session.record_confidence(0.2);
        assert!(!session.is_confidence_oscillating(),
            "Too few readings should not trigger oscillation")
    }

    // ─── Vector A: Root-Anchoring Tests ───────────────

    #[test]
    fn test_first_thought_captured() {
        let mut session = ThoughtSession::new("hypothesis", BudgetMode::Balanced);
        session.record_thought("My first thought about databases");
        session.record_thought("My second thought about caching");
        assert_eq!(session.first_thought.as_deref(), Some("My first thought about databases"));
    }

    #[test]
    fn test_combined_drift_uses_max() {
        let mut session = ThoughtSession::new(
            "Database migration zero downtime",
            BudgetMode::Balanced,
        );
        session.record_thought("Database schema migration plan with rollback");
        // current_thought completely off-topic from first-thought
        let drift = session.combined_drift(
            "Database migration zero downtime",  // hypothesis unchanged
            "Frontend CSS animation performance tuning with WebGL",  // drifted
        );
        assert!(drift > 0.3, "Combined drift should detect topic change: {:.3}", drift);
    }

    // ─── V6-3: Depth Degradation Tests ────────────────

    #[test]
    fn test_depth_degradation_detected() {
        let mut session = ThoughtSession::new("hypothesis", BudgetMode::Balanced);
        // Baseline: 3 thoughts with good depth
        session.record_depth_score(0.80);
        session.record_depth_score(0.75);
        session.record_depth_score(0.85);
        // Thought 4: depth collapses (attention saturated)
        session.record_depth_score(0.20);
        let degradation = session.depth_degradation();
        assert!(degradation.is_some(), "Should detect depth degradation");
        let ratio = degradation.unwrap();
        assert!(ratio > 0.50, "Degradation ratio should exceed 50%: {:.3}", ratio);
    }

    #[test]
    fn test_no_degradation_stable_depth() {
        let mut session = ThoughtSession::new("hypothesis", BudgetMode::Balanced);
        // Stable depth across all thoughts
        session.record_depth_score(0.70);
        session.record_depth_score(0.65);
        session.record_depth_score(0.75);
        session.record_depth_score(0.68);
        assert!(session.depth_degradation().is_none(),
            "Stable depth should not trigger degradation");
    }

    #[test]
    fn test_no_degradation_insufficient_data() {
        let mut session = ThoughtSession::new("hypothesis", BudgetMode::Balanced);
        session.record_depth_score(0.80);
        session.record_depth_score(0.10); // Looks like collapse but too few samples
        assert!(session.depth_degradation().is_none(),
            "Need at least 4 depth scores for degradation detection");
    }

    #[test]
    fn test_depth_degradation_near_zero_baseline() {
        let mut session = ThoughtSession::new("hypothesis", BudgetMode::Balanced);
        // Code-heavy thoughts with near-zero natural language depth
        session.record_depth_score(0.02);
        session.record_depth_score(0.01);
        session.record_depth_score(0.03);
        session.record_depth_score(0.01);
        assert!(session.depth_degradation().is_none(),
            "Near-zero baseline should not trigger false positive");
    }

    // ─── V7-2: Mode Collapse Detection Tests ──────────

    #[test]
    fn test_no_collapse_without_failures() {
        let session = ThoughtSession::new("hypothesis", BudgetMode::Balanced);
        // No failed thoughts registered → always orthogonal
        assert!(session.is_mode_collapse("any new thought here").is_none(),
            "Should not detect collapse when no failures registered");
    }

    #[test]
    fn test_collapse_detected_exact_clone() {
        let mut session = ThoughtSession::new("hypothesis", BudgetMode::Balanced);
        session.register_failed_thought("implement database migration with zero downtime using postgresql");
        // Exact same text → Jaccard = 1.0 → mode collapse
        let result = session.is_mode_collapse("implement database migration with zero downtime using postgresql");
        assert!(result.is_some(), "Should detect mode collapse on exact clone");
        assert!((result.unwrap() - 1.0).abs() < 0.01, "Exact clone should have ~1.0 similarity");
    }

    #[test]
    fn test_collapse_detected_paraphrase() {
        let mut session = ThoughtSession::new("hypothesis", BudgetMode::Balanced);
        session.register_failed_thought("implement database migration with zero downtime using postgresql");
        // Paraphrase with mostly same vocabulary → mode collapse
        let result = session.is_mode_collapse("database migration implementation for postgresql zero downtime");
        assert!(result.is_some(), "Should detect mode collapse on paraphrase: {:?}", result);
    }

    #[test]
    fn test_orthogonal_thought_passes() {
        let mut session = ThoughtSession::new("hypothesis", BudgetMode::Balanced);
        session.register_failed_thought("implement database migration with zero downtime using postgresql");
        // Completely different topic → orthogonal
        let result = session.is_mode_collapse("frontend css animation performance tuning with webgl shaders");
        assert!(result.is_none(),
            "Orthogonal thought should not trigger collapse: {:?}", result);
    }

    #[test]
    fn test_rollback_clears_failures() {
        let mut session = ThoughtSession::new("hypothesis", BudgetMode::Balanced);
        session.record_thought("thought zero");
        session.register_failed_thought("a failed thought");
        assert!(!session.failed_thoughts.is_empty());
        // Rollback clears failures for fresh branch
        session.rollback_to_thought(0);
        assert!(session.failed_thoughts.is_empty(),
            "Rollback should clear failed thoughts for fresh exploration");
    }
}
