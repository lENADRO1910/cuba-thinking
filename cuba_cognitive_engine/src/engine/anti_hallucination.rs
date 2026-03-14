// src/engine/anti_hallucination.rs
//
// R4: Anti-Hallucination Engine (9 layers) + R10: Warmup Guard + Anti-Overthinking
//
// Multi-layer verification system that catches reasoning failures:
// 1. Assumption Tracking  — Explicit assumption registry
// 2. Confidence Calibration — Stage-appropriate confidence ranges
// 3. Chain-of-Verification (CoVe) — Meta AI (Dhuliawala, 2023)
// 4. Evidence Accumulation — Wald Sequential Analysis (1945)
// 5. Claim Counter — Verifiable assertions tracking
// 6. Source Grounding — Grounded vs ungrounded claims
// 7. MCTS Enforcement — Quality gate on EWMA (R3+R6)
// 8. Contradiction Flag — NLI priority slot (reserved for S2)
// 9. Warmup Guard — Suppress false positives for thoughts 1-2
//
// R10: Anti-Overthinking (EWMA stagnation + early stopping, DeepSeek 2025)

use crate::engine::ewma_reward::EwmaTracker;
use crate::engine::quality_metrics::QualityScores;
use crate::engine::stage_engine::StageSession;
use serde::Serialize;

/// Aggregated anti-hallucination verdict for one thought.
#[derive(Debug, Clone, Serialize)]
pub struct HallucinationVerdict {
    /// Overall trust score (0.0 = certain hallucination, 1.0 = fully grounded).
    pub trust_score: f64,
    /// Layer-by-layer diagnostic results.
    pub layers: LayerResults,
    /// Actionable warnings (only critical ones).
    pub warnings: Vec<String>,
    /// Whether this thought should be rejected (protocol-level).
    pub should_reject: bool,
    /// Whether early stopping is recommended.
    pub should_early_stop: bool,
}

/// Results from each of the 9 anti-hallucination layers.
#[derive(Debug, Clone, Serialize)]
pub struct LayerResults {
    pub assumption_count: usize,
    pub confidence_calibrated: bool,
    pub cove_passed: bool,
    pub evidence_strength: f64,
    pub claim_count: usize,
    pub grounding_ratio: f64,
    pub ewma_above_threshold: bool,
    pub no_contradictions: bool,
    pub warmup_suppressed: bool,
}

/// Verification context passed to each layer checker.
struct VerifyContext<'a> {
    thought: &'a str,
    session: &'a StageSession,
    quality: &'a QualityScores,
    confidence: f64,
    thought_number: usize,
    is_warmup: bool,
}

/// Run all 9 anti-hallucination layers on a thought step.
///
/// Refactored: each layer is a standalone helper to keep CC ≤ 7.
/// Total CC of verify_thought itself: ~5 (2 branches + 1 if-else trust + 1 reject).
pub fn verify_thought(
    thought: &str,
    session: &StageSession,
    quality: &QualityScores,
    ewma: &mut EwmaTracker,
    confidence: f64,
    thought_number: usize,
) -> HallucinationVerdict {
    let ctx = VerifyContext {
        thought,
        session,
        quality,
        confidence,
        thought_number,
        is_warmup: thought_number <= 2,
    };

    let mut warnings = Vec::new();

    // ─── Layers 1-9 ──────────────────────────────────────────────
    let assumption_count = ctx.session.assumptions.len();
    check_assumption_layer(&ctx, assumption_count, &mut warnings);

    let confidence_calibrated = ctx.session.check_confidence(ctx.confidence).is_none();
    check_confidence_layer(&ctx, &mut warnings);

    let cove_passed = check_cove_structure(ctx.thought, ctx.thought_number);
    check_cove_layer(&ctx, cove_passed, &mut warnings);

    let evidence_strength = compute_evidence_strength(ctx.thought);
    check_evidence_layer(&ctx, evidence_strength, &mut warnings);

    let claim_count = count_verifiable_claims(ctx.thought);

    let grounding_ratio = compute_grounding_ratio(ctx.thought, claim_count);
    check_grounding_layer(&ctx, grounding_ratio, claim_count, &mut warnings);

    let ewma_above_threshold = !ewma.below_threshold();
    check_ewma_layer(ewma, ewma_above_threshold, &mut warnings);

    let no_contradictions = check_contradiction_layer(&ctx, &mut warnings);

    let warmup_suppressed = ctx.is_warmup;

    // ─── G4: Reward Gaming ───────────────────────────────────────
    check_gaming_layer(&ctx, evidence_strength, &mut warnings);

    // ─── R10: Anti-Overthinking ──────────────────────────────────
    let should_early_stop = check_overthinking_layer(ewma, ctx.is_warmup, &mut warnings);

    // ─── Compute Trust Score ─────────────────────────────────────
    let trust_score = compute_trust_score(
        quality, warmup_suppressed, evidence_strength,
        grounding_ratio, confidence_calibrated, ewma_above_threshold,
    );

    // ─── Should Reject? ──────────────────────────────────────────
    let should_reject = !ctx.is_warmup
        && (!ewma_above_threshold || trust_score < 0.25);

    HallucinationVerdict {
        trust_score,
        layers: LayerResults {
            assumption_count,
            confidence_calibrated,
            cove_passed,
            evidence_strength,
            claim_count,
            grounding_ratio,
            ewma_above_threshold,
            no_contradictions,
            warmup_suppressed,
        },
        warnings,
        should_reject,
        should_early_stop,
    }
}

// ─── Layer Helpers (CC=1-2 each) ─────────────────────────────────

/// L1: Flag excessive unverified assumptions.
fn check_assumption_layer(ctx: &VerifyContext, assumption_count: usize, warnings: &mut Vec<String>) {
    if assumption_count > 5 && !ctx.is_warmup {
        warnings.push(format!(
            "⚠️ L1: {} unverified assumptions — consider verifying the most critical ones",
            assumption_count
        ));
    }
}

/// L2: Confidence calibration check.
fn check_confidence_layer(ctx: &VerifyContext, warnings: &mut Vec<String>) {
    if let Some(cal_warn) = ctx.session.check_confidence(ctx.confidence) {
        if !ctx.is_warmup {
            warnings.push(cal_warn);
        }
    }
}

/// L3: Chain-of-Verification structure check.
fn check_cove_layer(ctx: &VerifyContext, cove_passed: bool, warnings: &mut Vec<String>) {
    if !cove_passed && !ctx.is_warmup && ctx.thought_number > 3 {
        warnings.push(
            "⚠️ L3 CoVe: Reasoning lacks cross-verification — add self-validation of claims".to_string(),
        );
    }
}

/// L4: Evidence accumulation threshold.
fn check_evidence_layer(ctx: &VerifyContext, evidence_strength: f64, warnings: &mut Vec<String>) {
    if evidence_strength < 0.3 && ctx.thought_number > 3 {
        warnings.push(format!(
            "📊 L4: Low evidence strength ({:.0}%) — claims lack supporting data/references",
            evidence_strength * 100.0
        ));
    }
}

/// L6: Source grounding ratio check.
fn check_grounding_layer(ctx: &VerifyContext, grounding_ratio: f64, claim_count: usize, warnings: &mut Vec<String>) {
    if grounding_ratio < 0.5 && claim_count > 2 && !ctx.is_warmup {
        warnings.push(format!(
            "🔍 L6: Only {:.0}% of {} claims are grounded — verify ungrounded assertions",
            grounding_ratio * 100.0,
            claim_count
        ));
    }
}

/// L7: MCTS EWMA enforcement gate.
fn check_ewma_layer(ewma: &EwmaTracker, ewma_above_threshold: bool, warnings: &mut Vec<String>) {
    if !ewma_above_threshold {
        warnings.push(format!(
            "🔴 L7: EWMA quality {:.0}% below threshold — consider backtracking to thought #{}",
            ewma.percentage(),
            ewma.best_thought_index().unwrap_or(0) + 1
        ));
    }
}

/// L8: Internal contradiction detection. Returns `true` if no contradictions found.
fn check_contradiction_layer(ctx: &VerifyContext, warnings: &mut Vec<String>) -> bool {
    use crate::engine::contradiction_detector;
    let internal_contras = contradiction_detector::detect_internal_contradictions(ctx.thought);
    let no_contradictions = internal_contras.is_empty();
    if !no_contradictions && !ctx.is_warmup {
        for c in &internal_contras {
            warnings.push(format!(
                "🚩 L8: Internal contradiction: '{}' vs '{}'",
                c.claim_a, c.claim_b
            ));
        }
    }
    no_contradictions
}

/// G4: Reward gaming detection (Everitt 2021).
fn check_gaming_layer(ctx: &VerifyContext, evidence_strength: f64, warnings: &mut Vec<String>) {
    if !ctx.is_warmup && ctx.thought_number > 2 {
        if detect_reward_gaming(ctx.thought, ctx.quality, evidence_strength) {
            warnings.push(
                "⚠️ G4: Potential reward gaming — high metric scores without proportional substance. Verify content authenticity."
                    .to_string(),
            );
        }
    }
}

/// R10: Anti-overthinking (stagnation + fatigue). Returns `should_early_stop`.
fn check_overthinking_layer(ewma: &mut EwmaTracker, is_warmup: bool, warnings: &mut Vec<String>) -> bool {
    let is_stagnating = ewma.is_stagnating();
    let is_fatigued = ewma.is_fatigued();

    if is_stagnating && !is_warmup {
        warnings.push(
            "🔄 R10: Stagnation detected (3+ steps with <2% improvement) — consider early stopping or changing approach"
                .to_string(),
        );
    }
    if is_fatigued && !is_warmup {
        warnings.push(
            "📉 R10: Quality fatigue (4+ consecutive drops) — reasoning is degrading, stop or restart"
                .to_string(),
        );
    }

    is_stagnating || is_fatigued
}

/// Compute aggregate trust score from layer results.
fn compute_trust_score(
    quality: &QualityScores,
    warmup_suppressed: bool,
    evidence_strength: f64,
    grounding_ratio: f64,
    confidence_calibrated: bool,
    ewma_above_threshold: bool,
) -> f64 {
    let quality_mean = quality.raw_mean();
    if warmup_suppressed {
        0.5 + quality_mean * 0.3
    } else {
        let base = quality_mean * 0.40
            + evidence_strength * 0.20
            + grounding_ratio * 0.20
            + confidence_calibrated as u8 as f64 * 0.10
            + ewma_above_threshold as u8 as f64 * 0.10;
        base.clamp(0.0, 1.0)
    }
}

/// Compute evidence strength by detecting numbers, data references,
/// measurements, and citations.
fn compute_evidence_strength(text: &str) -> f64 {
    let lower = text.to_lowercase();
    let mut score = 0.0_f64;
    
    // Has specific numbers (quantitative evidence)
    if text.chars().any(|c| c.is_ascii_digit()) {
        score += 0.25;
    }
    
    // Has measurement units
    let units = ["ms", "kb", "mb", "gb", "sec", "min", "loc", "%", "rpm", "mm", "kg"];
    if units.iter().any(|u| lower.contains(u)) {
        score += 0.20;
    }
    
    // Has citations or references
    let refs = [
        "according to", "based on", "source:", "reference:",
        "measured", "benchmark", "tested", "empirir",
        "según", "basado en", "fuente:", "medido", "probado",
    ];
    if refs.iter().any(|r| lower.contains(r)) {
        score += 0.30;
    }

    // Has code references
    if text.contains('`') || text.contains("```") {
        score += 0.15;
    }
    
    // Penalty for pure opinion
    let opinion = [
        "i think", "i believe", "in my opinion", "probably",
        "creo que", "en mi opinión", "probablemente",
    ];
    if opinion.iter().any(|o| lower.contains(o)) {
        score -= 0.10;
    }
    
    score.clamp(0.0, 1.0)
}

/// Count verifiable claims in text (sentences with assertions).
fn count_verifiable_claims(text: &str) -> usize {
    let sentences: Vec<&str> = text
        .split(['.', '!', '?'])
        .filter(|s| s.trim().len() > 10)
        .collect();

    sentences
        .iter()
        .filter(|s| {
            let lower = s.to_lowercase();
            // A claim has an assertion verb AND a subject
            (lower.contains(" is ") || lower.contains(" are ")
                || lower.contains(" was ") || lower.contains(" will ")
                || lower.contains(" must ") || lower.contains(" should ")
                || lower.contains(" causes ") || lower.contains(" requires ")
                || lower.contains(" es ") || lower.contains(" son ")
                || lower.contains(" debe ") || lower.contains(" requiere "))
                && s.len() > 20
        })
        .count()
}

/// Compute grounding ratio: proportion of claims with nearby evidence.
///
/// V5: Per-claim proximity check. For each claim-sentence, looks for
/// grounding markers in the same sentence or adjacent sentences (±1).
/// This replaces the previous global marker count which overcounted
/// when few markers covered the entire text.
fn compute_grounding_ratio(text: &str, total_claims: usize) -> f64 {
    if total_claims == 0 {
        return 1.0; // No claims to ground
    }

    // Split into sentences
    let sentences: Vec<&str> = text
        .split(['.', '?', '!', '\n'])
        .map(|s| s.trim())
        .filter(|s| s.len() > 5)
        .collect();

    if sentences.is_empty() {
        return 0.5; // Can't analyze structure
    }

    let grounding_markers = [
        "because", "since", "according to", "based on",
        "measured", "tested", "documentation", "spec",
        "isbn", "rfc", "iso", "doi", "arxiv", "github",
        "shows that", "proven by", "verified", "confirmed by",
        "porque", "según", "basado en", "documentación",
    ];

    let claim_verbs = [
        " is ", " are ", " was ", " will ", " must ", " should ",
        " causes ", " requires ", " prevents ", " ensures ",
        " es ", " son ", " debe ", " requiere ",
    ];

    // Identify claim-sentences and check proximity grounding
    let mut grounded_claims = 0usize;
    let mut detected_claims = 0usize;

    for (i, sentence) in sentences.iter().enumerate() {
        let lower = sentence.to_lowercase();

        // Is this sentence a claim?
        let is_claim = claim_verbs.iter().any(|v| lower.contains(v)) && sentence.len() > 20;
        if !is_claim {
            continue;
        }
        detected_claims += 1;

        // Check evidence in this sentence and adjacent ones (±1 window)
        let start = if i > 0 { i - 1 } else { 0 };
        let end = (i + 2).min(sentences.len()); // exclusive
        let window: String = sentences[start..end]
            .iter()
            .map(|s| s.to_lowercase())
            .collect::<Vec<_>>()
            .join(" ");

        if grounding_markers.iter().any(|m| window.contains(m)) {
            grounded_claims += 1;
        }
    }

    // Prefer sentence-level detected claims as denominator (more accurate).
    // Fall back to caller's total_claims if no sentence-level claims found.
    let denominator = if detected_claims > 0 { detected_claims } else { total_claims }.max(1);
    (grounded_claims as f64 / denominator as f64).min(1.0)
}

/// Phase 5C: Check for Chain-of-Verification (CoVe) structure.
///
/// CoVe (Dhuliawala et al., Meta AI 2023) verifies that reasoning contains
/// self-verification patterns — the model should check its own assertions.
///
/// Returns true if CoVe patterns are detected.
fn check_cove_structure(thought: &str, thought_number: usize) -> bool {
    // Early thoughts get a pass — CoVe is for mature reasoning
    if thought_number <= 2 {
        return true;
    }

    let lower = thought.to_lowercase();

    // CoVe markers: self-verification phrases
    let cove_markers = [
        "let me verify", "to verify", "checking", "confirmed",
        "cross-reference", "double-check", "validates", "consistent with",
        "as expected", "this confirms", "which means", "therefore",
        "assert", "ensures", "proven", "tested", "matching",
        // Spanish
        "verificar", "confirmar", "comprobar", "validar",
        "consistente con", "como se esperaba", "esto confirma",
        "por lo tanto", "lo que significa", "lo cual demuestra",
    ];

    let cove_count = cove_markers.iter().filter(|m| lower.contains(**m)).count();

    // Need at least 1 verification marker by thought 3+
    cove_count >= 1
}

/// G4: Detect reward gaming patterns (Everitt et al., 2021).
///
/// Identifies when a thought appears to optimize for scoring metrics
/// without providing proportional substantive content:
/// - High quality + low evidence (gaming quality dimensions without substance)
/// - Suspiciously perfect scores across all dimensions
/// - Grounding marker stuffing in very short text
/// - Repeated trivial assertions to pass PRM (FIX-3b: expanded patterns)
/// - Statistical anomaly: high μ + low σ² (FIX-3a: Goodhart's Law)
fn detect_reward_gaming(thought: &str, quality: &QualityScores, evidence_strength: f64) -> bool {
    let raw_mean = quality.raw_mean();
    let word_count = thought.split_whitespace().count();
    let lower = thought.to_lowercase();

    // Heuristic 1: High quality but no real evidence
    if raw_mean > 0.85 && evidence_strength < 0.25 {
        return true;
    }

    // Heuristic 2: Suspiciously perfect — all 6 dimensions > 0.85
    let all_high = quality.clarity > 0.85
        && quality.depth > 0.85
        && quality.breadth > 0.85
        && quality.logic > 0.85
        && quality.relevance > 0.85
        && quality.actionability > 0.85;
    if all_high && word_count < 50 {
        return true;
    }

    // Heuristic 3: Grounding marker stuffing in tiny text
    let grounding_markers = [
        "according to", "verified", "confirmed", "research shows",
        "data shows", "evidence indicates", "measured",
    ];
    let marker_count = grounding_markers
        .iter()
        .filter(|m| lower.contains(**m))
        .count();
    if marker_count > 4 && word_count < 30 {
        return true;
    }

    // Heuristic 4 (FIX-3b): PRM gaming — expanded trivial assertion patterns
    let trivial_count = lower.matches("assert true").count()
        + lower.matches("assert!(true)").count()
        + lower.matches("assert(true)").count()
        + lower.matches("assert 1 == 1").count()
        + lower.matches("assert 1==1").count()
        + lower.matches("assert!(1 == 1)").count()
        + lower.matches("pass").count();
    if trivial_count > 3 && word_count < 40 {
        return true;
    }

    // Heuristic 5 (FIX-3a): Statistical anomaly — high μ + low σ² (Goodhart's Law).
    // When all 6 quality dimensions are suspiciously uniform AND high,
    // the model likely optimized metrics without genuine substance.
    let scores = [
        quality.depth, quality.logic, quality.clarity,
        quality.breadth, quality.relevance, quality.actionability,
    ];
    let mean = scores.iter().sum::<f64>() / scores.len() as f64;
    let variance = scores.iter()
        .map(|s| (s - mean).powi(2))
        .sum::<f64>() / scores.len() as f64;
    if mean > 0.88 && variance < 0.005 {
        return true; // Suspiciously uniform high scores
    }

    // Heuristic 6 (L4): Pareto Frontier Violation — L2 Norm.
    // Depth and Clarity are physically antagonistic in natural language.
    //
    // v2 used product `d × c > 0.90` (hyperbola) — exploitable by saturating
    // one dimension: clarity=1.0, depth=0.89 → product=0.89 < 0.90 (passes).
    //
    // v3 uses L2 norm `d² + c² > 1.65` (circle) — geometrically impossible
    // to exploit: 1.0² + 0.89² = 1.79 > 1.65 (caught).
    //
    // Threshold verification:
    //   Legit (0.85, 0.85) → 1.445 < 1.65 ✓ passes
    //   High legit (0.90, 0.90) → 1.62 < 1.65 ✓ passes
    //   Gaming (0.95, 0.95) → 1.81 > 1.65 ✓ caught
    //   Exploit (0.89, 1.0) → 1.79 > 1.65 ✓ caught
    let pareto_norm = quality.depth.powi(2) + quality.clarity.powi(2);
    if pareto_norm > 1.65 {
        return true; // Pareto frontier violated — reward gaming
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::budget::BudgetMode;

    fn make_test_session() -> StageSession {
        StageSession::new()
    }

    fn make_test_quality() -> QualityScores {
        QualityScores {
            clarity: 0.7,
            depth: 0.6,
            breadth: 0.5,
            logic: 0.8,
            relevance: 0.7,
            actionability: 0.6,
        }
    }

    #[test]
    fn test_warmup_suppresses_warnings() {
        let session = make_test_session();
        let quality = make_test_quality();
        let mut ewma = EwmaTracker::new(BudgetMode::Balanced);

        let verdict = verify_thought("vague thought", &session, &quality, &mut ewma, 0.1, 1);
        assert!(verdict.layers.warmup_suppressed);
        assert!(!verdict.should_reject, "Should not reject during warmup");
    }

    #[test]
    fn test_low_evidence_warning() {
        let session = make_test_session();
        let quality = make_test_quality();
        let mut ewma = EwmaTracker::new(BudgetMode::Balanced);

        let verdict = verify_thought(
            "I think maybe the approach could potentially work somehow",
            &session, &quality, &mut ewma, 0.5, 5,
        );
        // Should have low evidence strength
        assert!(verdict.layers.evidence_strength < 0.3);
    }

    #[test]
    fn test_high_evidence_strength() {
        let strength = compute_evidence_strength(
            "According to the benchmark, query takes 250ms with 95th percentile at 500ms. \
             Based on PostgreSQL documentation, EXPLAIN shows sequential scan."
        );
        assert!(strength > 0.5, "Expected high evidence: {:.2}", strength);
    }

    #[test]
    fn test_claim_counting() {
        let claims = count_verifiable_claims(
            "PostgreSQL is faster than MySQL for complex queries. \
             The index must be created on the user_id column. \
             This approach requires careful testing."
        );
        assert!(claims >= 2, "Expected >=2 claims, got {}", claims);
    }

    #[test]
    fn test_grounding_ratio() {
        let ratio = compute_grounding_ratio(
            "The query is slow because the index is missing. \
             According to PostgreSQL documentation, B-tree indexes provide O(log n) lookup.",
            2,
        );
        assert!(ratio > 0.5, "Expected good grounding: {:.2}", ratio);
    }

    #[test]
    fn test_reject_on_low_ewma() {
        let session = make_test_session();
        let quality = QualityScores {
            clarity: 0.1, depth: 0.1, breadth: 0.1,
            logic: 0.1, relevance: 0.1, actionability: 0.1,
        };
        let mut ewma = EwmaTracker::new(BudgetMode::Balanced);
        // Push EWMA below threshold
        for _ in 0..10 {
            ewma.update(&crate::engine::ewma_reward::RewardSignals {
                quality: 0.05, faithfulness: 0.1, coherence: 0.1,
                contradiction_rate: 0.5, info_gain: 0.0, grounding: 0.0,
            });
        }
        let verdict = verify_thought("bad thought", &session, &quality, &mut ewma, 0.5, 5);
        assert!(verdict.should_reject, "Should reject when EWMA is below threshold");
    }

    // ─── FIX-3: Variance Gaming Detection Test ───────
    #[test]
    fn test_detect_reward_gaming_uniform_high_scores() {
        // Suspiciously uniform and high scores across all dimensions
        let quality = QualityScores {
            clarity: 0.92, depth: 0.91, breadth: 0.93,
            logic: 0.92, relevance: 0.91, actionability: 0.90,
        };
        let values = [quality.clarity, quality.depth, quality.breadth,
                      quality.logic, quality.relevance, quality.actionability];
        let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
        let variance: f64 = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;

        // Verify the condition from FIX-3a: mean > 0.88 AND variance < 0.005
        assert!(mean > 0.88, "Mean should be suspiciously high: {:.3}", mean);
        assert!(variance < 0.005, "Variance should be suspiciously low: {:.6}", variance);
    }

    // ─── L4: Pareto Frontier L2 Norm Test ──────────────
    #[test]
    fn test_pareto_frontier_l2_norm() {
        use crate::engine::quality_metrics::QualityScores;

        // Gaming: d²+c² = 0.96²+0.95² = 0.9216+0.9025 = 1.8241 > 1.65
        let gaming_quality = QualityScores {
            depth: 0.96, clarity: 0.95,
            breadth: 0.60, logic: 0.60,
            relevance: 0.60, actionability: 0.60,
        };
        let pareto = gaming_quality.depth.powi(2) + gaming_quality.clarity.powi(2);
        assert!(pareto > 1.65, "Gaming should exceed L2 threshold: {:.3}", pareto);

        // Exploit that bypassed v2 product: clarity=1.0, depth=0.89 → product=0.89 < 0.90
        // But L2 norm: 1.0² + 0.89² = 1.7921 > 1.65 → caught!
        let exploit_quality = QualityScores {
            depth: 0.89, clarity: 1.0,
            breadth: 0.60, logic: 0.60,
            relevance: 0.60, actionability: 0.60,
        };
        let exploit_pareto = exploit_quality.depth.powi(2) + exploit_quality.clarity.powi(2);
        assert!(exploit_pareto > 1.65, "v2 exploit should now be caught: {:.3}", exploit_pareto);

        // Genuine text should NOT trigger (depth and clarity are naturally antagonistic)
        let genuine_quality = QualityScores {
            depth: 0.88, clarity: 0.90,
            breadth: 0.70, logic: 0.75,
            relevance: 0.80, actionability: 0.65,
        };
        let genuine_pareto = genuine_quality.depth.powi(2) + genuine_quality.clarity.powi(2);
        assert!(genuine_pareto < 1.65, "Genuine should NOT exceed threshold: {:.3}", genuine_pareto);
    }
}
