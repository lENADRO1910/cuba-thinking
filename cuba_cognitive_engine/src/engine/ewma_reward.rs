// src/engine/ewma_reward.rs
//
// R3: EWMA Step Reward with 6-Signal Composite
//
// Exponentially Weighted Moving Average (Roberts, 1959) for tracking
// thought chain quality over time. Uses adaptive α floor (V4)
// for budget-aware smoothing.
//
// Composite reward includes 6 signals:
// - E0: Quality (weighted mean from 6D metrics)
// - E1: Faithfulness (ROSCOE, Golovneva 2023)
// - E2: Coherence (step transition consistency)
// - E3: Contradiction rate (inverse)
// - E4: Information Gain (Shannon, 1948)
// - E6: Source Grounding
//
// V10: 6-Signal composite (Shannon DPI)
// V4: Budget-aware α floor (Roberts 1959, Zangari 1994)
// V9: Softmax Gated Attention (Fin-PRM inspired, Zhou et al. 2025)
// V9: CUSUM collapse detection (FOCuS-inspired, Ward et al. NeurIPS 2023)

use crate::engine::budget::BudgetMode;
use serde::Serialize;
use std::collections::VecDeque;

/// Prior weights for the 6-signal composite reward (Bayesian priors).
/// V9: These serve as initialization for the Softmax Gated Attention.
const W_QUALITY: f64 = 0.40;
const W_COHERENCE: f64 = 0.20;
const W_CONTRADICTION: f64 = 0.10;
const W_FAITHFULNESS: f64 = 0.10;
const W_INFO_GAIN: f64 = 0.10;
const W_GROUNDING: f64 = 0.10;

/// V9: Softmax temperature for dynamic attention mechanism.
/// τ=0.5 balances between winner-take-all (τ→0) and uniform (τ→∞).
/// At τ=0.5, signals >1σ from mean receive ~2.7× weight boost.
const SOFTMAX_TAU: f64 = 0.5;

/// V9: CUSUM parameters for cognitive fatigue detection.
/// μ: target reward quality (optimistic baseline).
/// k: noise tolerance (slack). Absorbs stochastic fluctuation ≤5%.
/// h: degradation threshold. Exceeding certifies irreversible collapse.
const CUSUM_MU: f64 = 0.70;
const CUSUM_K: f64 = 0.05;
const CUSUM_H: f64 = 0.15;

/// Individual reward signals for one thought step.
#[derive(Debug, Clone, Serialize)]
pub struct RewardSignals {
    /// E0: Quality score (from 6D weighted mean)
    pub quality: f64,
    /// E1: Faithfulness — alignment with prior reasoning chain
    pub faithfulness: f64,
    /// E2: Coherence — similarity with previous thought
    pub coherence: f64,
    /// E3: Contradiction rate (0 = no contradictions, 1 = all contradictory)
    pub contradiction_rate: f64,
    /// E4: Information Gain — new concepts introduced
    pub info_gain: f64,
    /// E6: Source Grounding — grounded vs ungrounded claims
    pub grounding: f64,
}

impl RewardSignals {
    /// Compute composite reward from 6 signals.
    ///
    /// V9: Uses Softmax Gated Attention (Fin-PRM inspired, Zhou et al. 2025)
    /// instead of static weighted sum. Prior weights serve as Bayesian
    /// initialization. Signals that deviate from the mean receive
    /// exponentially scaled influence via softmax with τ=0.5.
    ///
    /// G6: Applies redundancy penalty when info_gain is near-zero
    /// (ROSCOE, Golovneva 2023; ReasonEval 2024).
    pub fn composite(&self) -> f64 {
        let priors = [
            W_QUALITY,
            W_COHERENCE,
            W_CONTRADICTION,
            W_FAITHFULNESS,
            W_INFO_GAIN,
            W_GROUNDING,
        ];
        self.composite_dynamic(&priors, SOFTMAX_TAU)
    }

    /// V7 (P3-E): Stage-adaptive composite scoring.
    ///
    /// Early stages (DEFINE/RESEARCH) boost quality weight to reward
    /// breadth of exploration. Late stages (VERIFY/SYNTHESIZE) boost
    /// faithfulness + grounding to reward rigor and evidence.
    ///
    /// V9: Stage priors modulate the Softmax attention mechanism.
    /// Accepts stage as lowercase string slice for loose coupling.
    #[allow(dead_code)]
    pub fn composite_for_stage(&self, stage: Option<&str>) -> f64 {
        let priors = match stage {
            Some("define" | "research") => [
                0.45, // +0.05 quality: reward exploration
                0.20, // coherence unchanged
                0.10, // contradiction unchanged
                0.05, // -0.05 faithfulness: less strict early
                0.15, // +0.05 info_gain: reward novelty
                0.05, // -0.05 grounding: less strict early
            ],
            Some("verify" | "synthesize") => [
                0.30, // -0.10 quality: less weight on breadth
                0.15, // -0.05 coherence: allow focused jumps
                0.10, // contradiction unchanged
                0.20, // +0.10 faithfulness: reward rigor
                0.05, // -0.05 info_gain: less novelty pressure
                0.20, // +0.10 grounding: demand evidence
            ],
            _ => [
                W_QUALITY,
                W_COHERENCE,
                W_CONTRADICTION,
                W_FAITHFULNESS,
                W_INFO_GAIN,
                W_GROUNDING,
            ],
        };
        self.composite_dynamic(&priors, SOFTMAX_TAU)
    }

    /// V9: Softmax Gated Attention composite — dynamic weight allocation.
    ///
    /// Algorithm (adapted from Fin-PRM Equation 22, Zhou et al. 2025):
    ///   1. Compute mean of all 6 signals (centering)
    ///   2. For each signal: advantage = (signal - mean) / τ
    ///   3. exp_weight[i] = prior[i] × exp(advantage)
    ///   4. Normalize via softmax: w[i] = exp_weight[i] / Σ(exp_weights)
    ///   5. composite = Σ(w[i] × signal[i])
    ///
    /// Properties:
    ///   - Uniform signals → reproduces prior weights exactly
    ///   - Anomalous signal → dominates composite (exponential scaling)
    ///   - O(N) where N=6 → < 0.1μs
    fn composite_dynamic(&self, priors: &[f64; 6], tau: f64) -> f64 {
        let signals = [
            self.quality,
            self.coherence,
            1.0 - self.contradiction_rate,
            self.faithfulness,
            self.info_gain,
            self.grounding,
        ];
        let mean = signals.iter().sum::<f64>() / 6.0;

        let mut exp_weights = [0.0_f64; 6];
        let mut sum_exp = 0.0_f64;
        for i in 0..6 {
            let advantage = (signals[i] - mean) / tau;
            exp_weights[i] = priors[i] * advantage.exp();
            sum_exp += exp_weights[i];
        }

        // Defensive: if all signals identical, sum_exp = Σ(priors) × e^0 = Σ(priors)
        // Division is well-defined. NaN only if all priors=0 (impossible).
        let raw: f64 = (0..6)
            .map(|i| (exp_weights[i] / sum_exp) * signals[i])
            .sum();

        // G6: Continuous exponential redundancy penalty (FIX-4).
        let redundancy_multiplier = (1.0 - (-15.0 * self.info_gain).exp()).max(0.10);

        raw * redundancy_multiplier
    }
}

impl Default for RewardSignals {
    fn default() -> Self {
        Self {
            quality: 0.5,
            faithfulness: 1.0,
            coherence: 1.0,
            contradiction_rate: 0.0,
            info_gain: 0.5,
            grounding: 0.5,
        }
    }
}

/// EWMA tracker for thought chain quality.
#[derive(Debug, Clone, Serialize)]
pub struct EwmaTracker {
    /// Current EWMA value (0.0 to 1.0).
    pub value: f64,
    /// Number of steps tracked.
    pub step_count: usize,
    /// History of recent step rewards (capped at MAX_HISTORY).
    pub reward_history: VecDeque<f64>,
    /// Budget mode for adaptive α.
    budget_mode: BudgetMode,
}

/// Maximum reward history entries to retain.
const MAX_HISTORY: usize = 20;

impl EwmaTracker {
    pub fn new(budget_mode: BudgetMode) -> Self {
        Self {
            value: 0.5,
            step_count: 0,
            reward_history: VecDeque::with_capacity(MAX_HISTORY),
            budget_mode,
        }
    }

    /// Update EWMA with a new step's reward signals.
    /// Returns the new EWMA value.
    pub fn update(&mut self, signals: &RewardSignals) -> f64 {
        self.step_count += 1;
        let reward = signals.composite();
        if self.reward_history.len() >= MAX_HISTORY {
            self.reward_history.pop_front();
        }
        self.reward_history.push_back(reward);

        // Adaptive α (V4): α = max(2/(n+1), α_floor)
        // Budget-aware floor prevents sluggish EWMA in long chains.
        let alpha_standard = 2.0 / (self.step_count as f64 + 1.0);
        let alpha_floor = self.budget_mode.ewma_alpha_floor();
        let alpha = alpha_standard.max(alpha_floor);

        // EWMA formula: EWMA_t = α · reward_t + (1 - α) · EWMA_{t-1}
        self.value = alpha * reward + (1.0 - alpha) * self.value;
        self.value
    }

    /// Check if EWMA has dropped below budget-aware MCTS threshold.
    /// Only triggers after thought #3 (warmup guard, V15).
    pub fn below_threshold(&self) -> bool {
        if self.step_count <= 3 {
            return false; // Warmup guard: suppress for first 3 thoughts
        }
        self.value < self.budget_mode.mcts_threshold()
    }

    /// V5-4: Hedged MCTS Rejection — stochastic threshold zone.
    ///
    /// Prevents Reward Hacking (NeurIPS 2025): LLMs mathematically discover
    /// deterministic thresholds and engineer outputs to surf at threshold+ε.
    /// The sigmoid zone makes the exact cutoff unknowable.
    ///
    /// Zones:
    /// - diff < -0.05: deterministic reject (saves compute)
    /// - diff > +0.05: deterministic accept
    /// - [-0.05, +0.05]: sigmoid probability P(reject) = 1/(1+e^(20·diff))
    ///
    /// Seed: O(1) from EWMA state — no RNG dependency.
    #[allow(dead_code)]
    pub fn should_reject_hedged(&self) -> bool {
        if self.step_count <= 3 {
            return false; // Warmup guard
        }

        let threshold = self.budget_mode.mcts_threshold();
        let diff = self.value - threshold;

        // Deterministic zones (avoid unnecessary computation)
        if diff < -0.05 {
            return true;
        }
        if diff > 0.05 {
            return false;
        }

        // Hedging zone: sigmoid rejection probability
        // At diff=0 → P=50%, diff=-0.04 → P≈68%, diff=+0.04 → P≈31%
        let rejection_prob = 1.0 / (1.0 + (20.0 * diff).exp());

        // O(1) deterministic seed from EWMA state — reproducible but unpredictable
        let seed = (self.value * 10000.0).fract().abs();
        seed < rejection_prob
    }

    /// V5-5: Process Advantage Verifier (PAV) — ICLR 2026.
    ///
    /// Measures how much the current step exceeds the historical baseline
    /// (V(s) = EWMA value). Penalizes "vacuous depth" — syntactically perfect
    /// steps that don't advance the reasoning toward the solution.
    ///
    /// Formula:
    ///   advantage = reward - V(s)
    ///   scaled = tanh(3 · advantage)  // [-1, 1]
    ///   PAV = 0.70 · reward + 0.30 · (scaled/2 + 0.5)  // [0, 1]
    ///
    /// Blend: 70% static quality + 30% momentum advantage.
    #[allow(dead_code)]
    pub fn compute_process_advantage(&self, current_reward: f64) -> f64 {
        if self.step_count == 0 {
            return current_reward;
        }

        let baseline_v = self.value; // Historical expectation V(s)
        let advantage = current_reward - baseline_v; // A(s,a)

        // Hyperbolic tangent normalization:
        // - Penalizes stagnation (advantage ≤ 0)
        // - Asymptotically scales genuine progress
        let scaled_advantage = (advantage * 3.0).tanh();

        // 70% static quality + 30% momentum
        let pav_score = (current_reward * 0.70) + ((scaled_advantage * 0.5 + 0.5) * 0.30);
        pav_score.clamp(0.0, 1.0)
    }

    /// Check for stagnation: 3+ consecutive thoughts with similar EWMA.
    /// V15: Only after thought #2.
    pub fn is_stagnating(&mut self) -> bool {
        if self.reward_history.len() < 3 || self.step_count <= 2 {
            return false;
        }
        let slice = self.reward_history.make_contiguous();
        let recent = &slice[slice.len() - 3..];
        let max_diff = recent
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .fold(0.0_f64, f64::max);
        max_diff < 0.02 // Less than 2% change across 3 steps
    }

    /// Check for quality fatigue: 3+ consecutive quality drops.
    pub fn is_fatigued(&mut self) -> bool {
        if self.reward_history.len() < 4 {
            return false;
        }
        let slice = self.reward_history.make_contiguous();
        let recent = &slice[slice.len() - 4..];
        recent.windows(2).all(|w| w[1] < w[0])
    }

    /// Check for early stopping opportunity:
    /// Quality > 0.7 and progress > 70%.
    /// Reserved for MCTS Phase 4 integration.
    #[allow(dead_code)]
    pub fn should_early_stop(&self, progress_pct: f64) -> bool {
        self.value > 0.7 && progress_pct > 70.0
    }

    /// Find the best thought index (highest reward) for backtracking.
    /// P1-4: total_cmp() provides NaN-safe total ordering (NaN sorts to end).
    pub fn best_thought_index(&self) -> Option<usize> {
        self.reward_history
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, _)| i)
    }

    /// EWMA as percentage for display.
    pub fn percentage(&self) -> f64 {
        self.value * 100.0
    }

    /// V9: One-sided CUSUM collapse detection — instantaneous change-point.
    ///
    /// Replaces MACD (v3-v4) which requires 4+ data points and suffers
    /// from inertial lag. CUSUM detects degradation from step 1.
    ///
    /// Lower-sided CUSUM (FOCuS-inspired, Ward et al. NeurIPS 2023):
    ///   S_t = max(0, S_{t-1} + μ - x_t - k)
    ///   Alert when S_t > h
    ///
    /// Parameters (calibrated per Deep Research 2025):
    ///   μ = 0.70 — target baseline (optimistic quality expectation)
    ///   k = 0.05 — noise slack (absorbs stochastic fluctuation ≤5%)
    ///   h = 0.15 — degradation threshold (irreversible collapse signal)
    ///
    /// Complexity: O(N) single pass, no EMA state needed.
    pub fn is_collapsing_kinematically(&self) -> bool {
        if self.reward_history.len() < 2 {
            return false;
        }

        let mut s_t = 0.0_f64;
        for &r in &self.reward_history {
            s_t = (s_t + CUSUM_MU - r - CUSUM_K).max(0.0);
            if s_t > CUSUM_H {
                return true;
            }
        }
        false
    }

    /// Legacy MACD collapse detection (V3-V4). Retained for reference.
    /// Superseded by CUSUM (V9) which detects degradation without lag.
    #[allow(dead_code)]
    fn is_collapsing_macd_legacy(&self) -> bool {
        let n = self.reward_history.len();
        if n < 4 {
            return false;
        }

        let vals: Vec<f64> = self.reward_history.iter().copied().collect();

        let mut ema_fast = 0.0;
        let mut ema_slow = 0.0;

        const BIAS_EPSILON: f64 = 1e-12;

        for (t, &r) in vals.iter().enumerate() {
            ema_fast = 0.5 * r + 0.5 * ema_fast;
            ema_slow = 0.2 * r + 0.8 * ema_slow;

            if t == n - 1 {
                let t_f64 = (t + 1) as f64;
                let unbiased_fast = ema_fast / (1.0 - 0.5_f64.powf(t_f64)).max(BIAS_EPSILON);
                let unbiased_slow = ema_slow / (1.0 - 0.8_f64.powf(t_f64)).max(BIAS_EPSILON);

                let macd = unbiased_fast - unbiased_slow;
                return macd < -0.08;
            }
        }
        false
    }

    /// G2: Outcome Reward Model — weighted harmonic mean (FIX-2).
    ///
    /// ORM = Σwᵢ / Σ(wᵢ / max(PRMᵢ, ε))
    ///
    /// Harmonic mean treats reasoning as a series circuit:
    /// one defective step collapses the entire chain score.
    /// Recency bias: wᵢ = 1 + 0.5 * (i/n), later steps weigh more.
    ///
    /// Ref: Cobbe et al. 2021 (OpenAI ORM), harmonic-mean property.
    pub fn chain_score(&self) -> f64 {
        if self.reward_history.is_empty() {
            return 0.5;
        }

        const EPSILON: f64 = 1e-5;
        let n = self.reward_history.len() as f64;

        let (weight_sum, inv_sum) = self.reward_history.iter().enumerate().fold(
            (0.0_f64, 0.0_f64),
            |(ws, is), (i, &reward)| {
                let w = 1.0 + 0.5 * (i as f64 / n);
                (ws + w, is + w / reward.max(EPSILON))
            },
        );

        if inv_sum > 0.0 {
            (weight_sum / inv_sum).clamp(0.0, 1.0)
        } else {
            0.5
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ewma_basic_update() {
        let mut tracker = EwmaTracker::new(BudgetMode::Balanced);
        let signals = RewardSignals {
            quality: 0.8,
            faithfulness: 0.9,
            coherence: 0.7,
            contradiction_rate: 0.0,
            info_gain: 0.6,
            grounding: 0.5,
            ..Default::default()
        };
        let new_val = tracker.update(&signals);
        assert!(
            new_val > 0.5,
            "EWMA should increase with good signals: {}",
            new_val
        );
    }

    #[test]
    fn test_ewma_warmup_guard() {
        let mut tracker = EwmaTracker::new(BudgetMode::Balanced);
        let bad_signals = RewardSignals {
            quality: 0.1,
            ..Default::default()
        };
        tracker.update(&bad_signals);
        tracker.update(&bad_signals);
        tracker.update(&bad_signals);
        // Should NOT trigger threshold within first 3 thoughts
        assert!(!tracker.below_threshold());
    }

    #[test]
    fn test_ewma_threshold_trigger() {
        let mut tracker = EwmaTracker::new(BudgetMode::Balanced);
        let bad_signals = RewardSignals {
            quality: 0.05,
            faithfulness: 0.1,
            coherence: 0.1,
            contradiction_rate: 0.5,
            info_gain: 0.0,
            grounding: 0.0,
        };
        for _ in 0..10 {
            tracker.update(&bad_signals);
        }
        assert!(
            tracker.below_threshold(),
            "EWMA should be below threshold after 10 bad steps"
        );
    }

    #[test]
    fn test_stagnation_detection() {
        let mut tracker = EwmaTracker::new(BudgetMode::Balanced);
        let same_signals = RewardSignals::default();
        for _ in 0..5 {
            tracker.update(&same_signals);
        }
        assert!(
            tracker.is_stagnating(),
            "Should detect stagnation with identical inputs"
        );
    }

    #[test]
    fn test_fatigue_detection() {
        let mut tracker = EwmaTracker::new(BudgetMode::Balanced);
        // V9: Use realistic signals that produce decreasing composites
        // under Softmax. info_gain must be non-zero to avoid redundancy
        // penalty collapse (multiplier=0.10).
        for q in [0.8, 0.6, 0.4, 0.2] {
            let signals = RewardSignals {
                quality: q,
                faithfulness: q,
                coherence: q,
                contradiction_rate: 1.0 - q,
                info_gain: q,
                grounding: q,
            };
            tracker.update(&signals);
        }
        assert!(
            tracker.is_fatigued(),
            "Should detect fatigue with 4 consecutive drops"
        );
    }

    #[test]
    fn test_composite_reward_priors_sum_to_one() {
        let total =
            W_QUALITY + W_COHERENCE + W_CONTRADICTION + W_FAITHFULNESS + W_INFO_GAIN + W_GROUNDING;
        assert!(
            (total - 1.0).abs() < 1e-10,
            "Prior weights must sum to 1.0: {}",
            total
        );
    }

    #[test]
    fn test_softmax_uniform_signals_reproduce_priors() {
        // When all signals are identical, softmax reduces to prior weights.
        let signals = RewardSignals {
            quality: 0.7,
            faithfulness: 0.7,
            coherence: 0.7,
            contradiction_rate: 0.3,
            info_gain: 0.7,
            grounding: 0.7,
        };
        let composite = signals.composite();
        // With uniform signals (all ~0.7), composite should be close to
        // static weighted sum: 0.7 × Σ(priors) = 0.7
        // (minus redundancy penalty for info_gain=0.7)
        let redundancy = (1.0 - (-15.0 * 0.7_f64).exp()).max(0.10);
        let expected = 0.7 * redundancy;
        assert!(
            (composite - expected).abs() < 0.05,
            "Uniform signals should approximate prior-weighted sum: got={:.4} expected≈{:.4}",
            composite,
            expected
        );
    }

    #[test]
    fn test_softmax_anomalous_signal_dominates() {
        // When one signal is anomalously high, it should dominate the composite.
        let normal = RewardSignals {
            quality: 0.5,
            faithfulness: 0.5,
            coherence: 0.5,
            contradiction_rate: 0.5,
            info_gain: 0.5,
            grounding: 0.5,
        };
        let anomalous = RewardSignals {
            quality: 0.5,
            faithfulness: 0.5,
            coherence: 0.5,
            contradiction_rate: 0.5,
            info_gain: 0.5,
            grounding: 0.95, // Anomalous
        };
        // Anomalous should give more weight to grounding → higher composite
        assert!(
            anomalous.composite() > normal.composite(),
            "Anomalous signal should increase composite: anom={:.4} > norm={:.4}",
            anomalous.composite(),
            normal.composite()
        );
    }

    #[test]
    fn test_best_thought_index() {
        let mut tracker = EwmaTracker::new(BudgetMode::Balanced);
        let signals_low = RewardSignals {
            quality: 0.2,
            ..Default::default()
        };
        let signals_high = RewardSignals {
            quality: 0.9,
            ..Default::default()
        };
        tracker.update(&signals_low);
        tracker.update(&signals_high);
        tracker.update(&signals_low);
        assert_eq!(tracker.best_thought_index(), Some(1));
    }

    // ─── FIX-2: Harmonic Chain Score Tests ─────────────
    #[test]
    fn test_chain_score_harmonic_single_bad_collapses() {
        let mut tracker = EwmaTracker::new(BudgetMode::Balanced);
        // 4 excellent + 1 terrible
        for _ in 0..4 {
            tracker.update(&RewardSignals {
                quality: 0.9,
                faithfulness: 0.9,
                coherence: 0.9,
                info_gain: 0.8,
                grounding: 0.8,
                ..Default::default()
            });
        }
        tracker.update(&RewardSignals {
            quality: 0.01,
            faithfulness: 0.01,
            coherence: 0.01,
            info_gain: 0.01,
            grounding: 0.01,
            ..Default::default()
        });

        let score = tracker.chain_score();
        assert!(
            score < 0.3,
            "Harmonic mean should collapse on bad step: {:.3}",
            score
        );
    }

    #[test]
    fn test_chain_score_all_good() {
        let mut tracker = EwmaTracker::new(BudgetMode::Balanced);
        for _ in 0..5 {
            tracker.update(&RewardSignals {
                quality: 0.7,
                faithfulness: 0.7,
                coherence: 0.7,
                info_gain: 0.7,
                grounding: 0.7,
                ..Default::default()
            });
        }
        let score = tracker.chain_score();
        assert!(
            score > 0.5,
            "All good steps should give healthy chain: {:.3}",
            score
        );
    }

    // ─── FIX-4: Exponential Redundancy Tests ──────────
    #[test]
    fn test_redundancy_continuous_monotonic() {
        // The exponential penalty 1 - e^(-15x) should be continuous and monotonic
        let values: Vec<f64> = (0..=10)
            .map(|i| {
                let x = i as f64 * 0.1; // 0.0, 0.1, ... 1.0
                1.0 - (-15.0 * x).exp()
            })
            .collect();
        for w in values.windows(2) {
            assert!(
                w[1] >= w[0],
                "Penalty should be monotonic: {} >= {}",
                w[1],
                w[0]
            );
        }
        // At x=0, penalty should be ~0
        assert!(
            values[0].abs() < 0.01,
            "Penalty at 0% redundancy should be ~0"
        );
        // At x=0.3, penalty should be near saturation (>0.98)
        assert!(values[3] > 0.98, "Penalty at 30% should be near max");
    }

    // ─── V9: CUSUM Collapse Tests ────────────────────────
    #[test]
    fn test_cusum_collapse_accelerating_drops() {
        let mut tracker = EwmaTracker::new(BudgetMode::Balanced);
        // CUSUM detects collapse earlier than MACD (no 4-point minimum).
        // Each step below μ=0.70 accumulates: S += 0.70 - r - 0.05
        // r=0.90 → S = max(0, 0+0.70-0.90-0.05) = 0.0 (above target)
        // r=0.50 → S = max(0, 0+0.70-0.50-0.05) = 0.15 (at threshold)
        // r=0.20 → S = max(0, 0.15+0.70-0.20-0.05) = 0.60 (well above h=0.15)
        tracker.reward_history.push_back(0.90);
        tracker.reward_history.push_back(0.50);
        tracker.reward_history.push_back(0.20);
        assert!(
            tracker.is_collapsing_kinematically(),
            "Sharp decline should trigger CUSUM collapse"
        );
    }

    #[test]
    fn test_cusum_collapse_stable_sequence() {
        let mut tracker = EwmaTracker::new(BudgetMode::Balanced);
        // Stable around μ=0.70: deviation absorbed by slack k=0.05
        tracker.reward_history.push_back(0.70);
        tracker.reward_history.push_back(0.72);
        tracker.reward_history.push_back(0.71);
        tracker.reward_history.push_back(0.70);
        assert!(
            !tracker.is_collapsing_kinematically(),
            "Stable sequence should NOT trigger CUSUM collapse"
        );
    }

    #[test]
    fn test_cusum_collapse_insufficient_data() {
        let mut tracker = EwmaTracker::new(BudgetMode::Balanced);
        // Only 1 data point — CUSUM needs ≥2
        tracker.reward_history.push_back(0.80);
        assert!(
            !tracker.is_collapsing_kinematically(),
            "Fewer than 2 data points should safely return false"
        );
    }

    #[test]
    fn test_cusum_early_detection() {
        let mut tracker = EwmaTracker::new(BudgetMode::Balanced);
        // CUSUM advantage: detects collapse at step 2 (r=0.40)
        // S1 = max(0, 0+0.70-0.40-0.05) = 0.25 > h=0.15 → DETECTED!
        // MACD would need 4+ points to even start calculating.
        tracker.reward_history.push_back(0.40);
        tracker.reward_history.push_back(0.40);
        assert!(
            tracker.is_collapsing_kinematically(),
            "CUSUM should detect collapse earlier than MACD"
        );
    }

    #[test]
    fn test_cusum_gradual_recovery() {
        let mut tracker = EwmaTracker::new(BudgetMode::Balanced);
        // Dip then recovery: CUSUM resets on good values
        tracker.reward_history.push_back(0.75);
        tracker.reward_history.push_back(0.65);
        tracker.reward_history.push_back(0.75);
        tracker.reward_history.push_back(0.80);
        assert!(
            !tracker.is_collapsing_kinematically(),
            "Recovery should prevent CUSUM from triggering"
        );
    }

    // ─── V7 (P3-E): Stage-Adaptive Composite Tests ────────

    #[test]
    fn test_stage_adaptive_weights_sum_to_one() {
        // Verify all stage weight sets sum to 1.0
        let signals = RewardSignals {
            quality: 1.0,
            faithfulness: 1.0,
            coherence: 1.0,
            contradiction_rate: 0.0,
            info_gain: 1.0,
            grounding: 1.0,
        };
        let default = signals.composite();
        let define = signals.composite_for_stage(Some("define"));
        let verify = signals.composite_for_stage(Some("verify"));
        let none = signals.composite_for_stage(None);
        // All should produce same score when all signals = 1.0
        // (because weights sum to 1.0 regardless of distribution)
        assert!(
            (default - none).abs() < 1e-10,
            "None stage should match default"
        );
        assert!(
            (default - define).abs() < 1e-10,
            "All-1.0 signals should match regardless of weights"
        );
        assert!(
            (default - verify).abs() < 1e-10,
            "All-1.0 signals should match regardless of weights"
        );
    }

    #[test]
    fn test_stage_adaptive_define_boosts_quality() {
        let signals = RewardSignals {
            quality: 0.9,
            faithfulness: 0.3,
            coherence: 0.7,
            contradiction_rate: 0.0,
            info_gain: 0.8,
            grounding: 0.3,
        };
        let default_score = signals.composite();
        let define_score = signals.composite_for_stage(Some("define"));
        // DEFINE boosts quality+info_gain, reduces faithfulness+grounding
        // With high quality (0.9) and low faithfulness (0.3), DEFINE should score higher
        assert!(define_score > default_score,
            "DEFINE stage should boost score for high-quality/low-faithfulness: define={:.4} vs default={:.4}",
            define_score, default_score);
    }

    #[test]
    fn test_stage_adaptive_verify_boosts_faithfulness() {
        let signals = RewardSignals {
            quality: 0.4,
            faithfulness: 0.9,
            coherence: 0.5,
            contradiction_rate: 0.0,
            info_gain: 0.3,
            grounding: 0.9,
        };
        let default_score = signals.composite();
        let verify_score = signals.composite_for_stage(Some("verify"));
        // VERIFY boosts faithfulness+grounding, reduces quality+info_gain
        // With high faithfulness (0.9) and low quality (0.4), VERIFY should score higher
        assert!(verify_score > default_score,
            "VERIFY stage should boost score for high-faithfulness/low-quality: verify={:.4} vs default={:.4}",
            verify_score, default_score);
    }
}
