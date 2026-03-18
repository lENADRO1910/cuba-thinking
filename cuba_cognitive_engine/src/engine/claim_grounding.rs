// src/engine/claim_grounding.rs
//
// S5: ROSCOE Faithfulness + Source Grounding Analysis
//
// Measures how well-grounded claims are in evidence and data.
// Based on ROSCOE (Golovneva et al., 2023, ICLR) metrics
// for evaluating reasoning chain quality.
//
// Replaces hardcoded `faithfulness: 1.0` and `grounding: heuristic`
// in EWMA signals.
//
// Signals:
// - Faithfulness: ratio of claims with supporting evidence
// - Grounding: ratio of claims anchored in concrete data
// - Hedging ratio: proportion of uncertain language (penalty)
// - Evidence density: specific facts per claim

use serde::Serialize;

/// Result of claim grounding analysis.
#[derive(Debug, Clone, Serialize)]
pub struct GroundingResult {
    /// Faithfulness score (0.0 to 1.0).
    /// How well does the thought stay faithful to prior evidence?
    pub faithfulness: f64,
    /// Grounding score (0.0 to 1.0).
    /// How well-grounded are claims in concrete data?
    pub grounding: f64,
    /// Number of claims detected.
    pub claim_count: usize,
    /// Number of evidence markers found.
    pub evidence_count: usize,
    /// Hedging ratio (0.0 to 1.0).
    pub hedging_ratio: f64,
}

/// Analyze grounding quality of a thought.
///
/// Returns `GroundingResult` with faithfulness and grounding scores
/// based on claim density, evidence markers, and hedging analysis.
pub fn analyze_grounding(thought: &str) -> GroundingResult {
    if thought.is_empty() {
        return GroundingResult {
            faithfulness: 0.5,
            grounding: 0.5,
            claim_count: 0,
            evidence_count: 0,
            hedging_ratio: 0.0,
        };
    }

    let lower = thought.to_lowercase();

    // ── Claim detection ──────────────────────────────────────
    let claim_count = count_claims(&lower);

    // ── Evidence markers ─────────────────────────────────────
    let evidence_count = count_evidence_markers(&lower, thought);

    // ── Hedging detection ────────────────────────────────────
    let hedging_ratio = compute_hedging_ratio(&lower);

    // ── Faithfulness: evidence support per claim ─────────────
    // High faithfulness = many evidence markers relative to claims
    let faithfulness = if claim_count == 0 {
        0.7 // Neutral: no claims made → mildly faithful
    } else {
        let support_ratio = (evidence_count as f64 / claim_count as f64).min(1.0);
        // Penalize heavy hedging
        let hedge_penalty = hedging_ratio * 0.3;
        (support_ratio * 0.7 + 0.3 - hedge_penalty).clamp(0.0, 1.0)
    };

    // ── Grounding: concrete data anchoring ───────────────────
    // High grounding = specific numbers, code, references
    let specificity = compute_specificity(thought);
    let grounding = if claim_count == 0 {
        specificity // No claims → grounding is just specificity
    } else {
        let evidence_density = (evidence_count as f64 * 0.15).min(0.5);
        (specificity * 0.5 + evidence_density + 0.1 - hedging_ratio * 0.2).clamp(0.0, 1.0)
    };

    GroundingResult {
        faithfulness,
        grounding,
        claim_count,
        evidence_count,
        hedging_ratio,
    }
}

/// Count claim-like statements in text.
/// Claims are declarative statements making assertions.
fn count_claims(lower: &str) -> usize {
    let claim_markers = [
        "should ",
        "must ",
        "will ",
        "need to ",
        "require",
        "is better",
        "is worse",
        "performs",
        "results in",
        "causes ",
        "leads to",
        "prevents ",
        "ensures ",
        "guarantees ",
        "eliminates ",
        "reduces ",
        "increases ",
        "the best",
        "the worst",
        "optimal",
        "recommended",
        "therefore",
        "consequently",
        "in conclusion",
        // Spanish
        "debe ",
        "necesita ",
        "resulta en",
        "causa ",
        "garantiza ",
        "elimina ",
        "reduce ",
        "aumenta ",
        "el mejor",
        "el peor",
        "óptimo",
        "recomendado",
        "por lo tanto",
        "en conclusión",
    ];

    claim_markers
        .iter()
        .map(|m| lower.matches(m).count())
        .sum::<usize>()
        .max(
            // Also count sentences ending with period (basic claim detection)
            lower.matches(". ").count() / 2,
        )
}

/// Count evidence markers — specific facts, data, references.
fn count_evidence_markers(lower: &str, original: &str) -> usize {
    let mut count = 0;

    // Numbers (quantities, measurements, percentages) — count tokens, not digits
    count += original
        .split_whitespace()
        .filter(|w| w.chars().any(|c| c.is_ascii_digit()))
        .count()
        .min(5); // Cap at 5 to avoid over-counting

    // Code references (backticks, file paths)
    count += lower.matches('`').count() / 2; // Pairs of backticks
    count += lower.matches("```").count();

    // Measurement units
    let units = [
        "ms", "seconds", "minutes", "bytes", "kb", "mb", "gb", "percent", "%", "rpm", "mm", "cm",
        "hz", "mhz", "ghz",
    ];
    count += units.iter().filter(|u| lower.contains(**u)).count();

    // Reference markers
    let references = [
        "according to",
        "based on",
        "as described",
        "per ",
        "see ",
        "ref:",
        "source:",
        "paper:",
        "iso ",
        "rfc ",
        "pep ",
        "owasp",
        "según ",
        "basado en",
        "como describe",
    ];
    count += references.iter().filter(|r| lower.contains(**r)).count();

    // Technical specificity (concrete terms)
    let specific_terms = [
        "function",
        "method",
        "class",
        "struct",
        "endpoint",
        "port",
        "config",
        "parameter",
        "variable",
        "column",
        "table",
        "index",
        "constraint",
        "migration",
        "función",
        "método",
        "tabla",
        "columna",
        "migración",
    ];
    count += specific_terms
        .iter()
        .filter(|t| lower.contains(**t))
        .count()
        / 2; // Halve to avoid over-counting

    count
}

/// Compute hedging ratio — proportion of uncertain language.
fn compute_hedging_ratio(lower: &str) -> f64 {
    let hedging_markers = [
        "maybe",
        "perhaps",
        "possibly",
        "probably",
        "might",
        "could be",
        "seems like",
        "appears to",
        "kind of",
        "sort of",
        "somewhat",
        "roughly",
        "approximately",
        "around",
        "about ",
        "unclear",
        "uncertain",
        "debatable",
        "arguable",
        // Spanish
        "quizás",
        "tal vez",
        "posiblemente",
        "probablemente",
        "podría ser",
        "parece que",
        "algo así",
        "más o menos",
        "aproximadamente",
        "alrededor de",
        "incierto",
    ];

    let word_count = lower.split_whitespace().count().max(1);
    let hedge_count: usize = hedging_markers
        .iter()
        .map(|m| lower.matches(m).count())
        .sum();

    (hedge_count as f64 / word_count as f64 * 10.0).min(1.0)
}

/// Compute specificity score — how concrete and specific is the text?
fn compute_specificity(text: &str) -> f64 {
    let mut score = 0.0;

    // Numbers present
    if text.chars().any(|c| c.is_ascii_digit()) {
        score += 0.2;
    }

    // Code present
    if text.contains('`') || text.contains("```") || text.contains("::") {
        score += 0.2;
    }

    // File paths
    if text.contains('/') || text.contains(".rs") || text.contains(".py") || text.contains(".ts") {
        score += 0.15;
    }

    // Technical terms density
    let lower = text.to_lowercase();
    let technical = [
        "api", "sql", "http", "json", "yaml", "docker", "redis", "postgres", "nginx", "async",
        "thread",
    ];
    let tech_count = technical.iter().filter(|t| lower.contains(**t)).count();
    score += (tech_count as f64 * 0.05).min(0.25);

    // Length bonus (longer = more specific, up to a point)
    let words = text.split_whitespace().count();
    if words > 20 {
        score += 0.1;
    }
    if words > 50 {
        score += 0.1;
    }

    score.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_well_grounded_thought() {
        let thought = "Based on ISO 286, we should set the tolerance to 0.05mm. \
                        The `create_cylinder` function handles this with diameter=50mm. \
                        According to the PostgreSQL docs, the migration takes ~300ms.";
        let result = analyze_grounding(thought);
        assert!(
            result.grounding > 0.5,
            "Well-grounded thought should score high: {:.3}",
            result.grounding
        );
        assert!(
            result.faithfulness > 0.5,
            "Evidence-backed claims should be faithful: {:.3}",
            result.faithfulness
        );
        assert!(
            result.evidence_count > 3,
            "Should find multiple evidence markers"
        );
    }

    #[test]
    fn test_hedging_heavy_thought() {
        let thought = "Maybe we should probably consider possibly implementing \
                        something that could perhaps work, roughly speaking.";
        let result = analyze_grounding(thought);
        assert!(
            result.hedging_ratio > 0.2,
            "Heavy hedging should be detected: {:.3}",
            result.hedging_ratio
        );
        assert!(
            result.faithfulness < 0.7,
            "Hedging should reduce faithfulness: {:.3}",
            result.faithfulness
        );
    }

    #[test]
    fn test_empty_thought() {
        let result = analyze_grounding("");
        assert_eq!(result.faithfulness, 0.5);
        assert_eq!(result.grounding, 0.5);
        assert_eq!(result.claim_count, 0);
    }

    #[test]
    fn test_code_heavy_thought() {
        let thought = "The function `validate_input()` at line 42 returns `Result<(), Error>`. \
                        It processes 1000 requests/second with 99.9% uptime.";
        let result = analyze_grounding(thought);
        assert!(
            result.grounding > 0.4,
            "Code-heavy thought should have high grounding: {:.3}",
            result.grounding
        );
    }

    #[test]
    fn test_vague_thought() {
        let thought = "Things should work better if we change stuff around somehow.";
        let result = analyze_grounding(thought);
        assert!(
            result.grounding < 0.5,
            "Vague thought should have low grounding: {:.3}",
            result.grounding
        );
    }
}
