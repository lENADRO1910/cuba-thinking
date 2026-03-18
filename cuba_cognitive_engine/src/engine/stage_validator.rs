// src/engine/stage_validator.rs
//
// Phase 5B: Stage-Content Alignment Validator (G8)
//
// Validates that a thought's content actually matches its declared cognitive stage.
// For example, a thought declared as VERIFY should contain verification patterns
// (testing, assertion, comparison), not definition patterns (conceptual, taxonomy).
//
// This prevents the agent from misusing stages to game the quality weights.

use crate::engine::stage_engine::CognitiveStage;
use serde::Serialize;

/// Result of stage-content alignment validation.
#[derive(Debug, Clone, Serialize)]
pub struct StageAlignmentResult {
    /// Declared stage by the agent.
    pub declared: CognitiveStage,
    /// Detected stage from content analysis.
    pub detected: CognitiveStage,
    /// Alignment score (0.0 = misaligned, 1.0 = perfectly aligned).
    pub alignment: f64,
    /// Warning if misaligned.
    pub warning: Option<String>,
}

/// Stage-specific keyword patterns for content detection.
const DEFINE_PATTERNS: &[&str] = &[
    "define",
    "what is",
    "concept",
    "definition",
    "taxonomy",
    "classify",
    "scope",
    "problem statement",
    "requirements",
    "context",
    "definir",
    "qué es",
    "concepto",
    "definición",
    "taxonomía",
    "alcance",
    "requisitos",
    "contexto",
];

const RESEARCH_PATTERNS: &[&str] = &[
    "research",
    "investigate",
    "literature",
    "survey",
    "compare",
    "benchmark",
    "study",
    "existing",
    "prior work",
    "state of the art",
    "investigar",
    "literatura",
    "comparar",
    "estudio",
    "existente",
];

const ANALYZE_PATTERNS: &[&str] = &[
    "analyze",
    "breakdown",
    "decompose",
    "inspect",
    "examine",
    "root cause",
    "trace",
    "debug",
    "dissect",
    "pattern",
    "analizar",
    "descomponer",
    "inspeccionar",
    "examinar",
    "causa raíz",
    "trazar",
    "depurar",
    "patrón",
];

const HYPOTHESIZE_PATTERNS: &[&str] = &[
    "hypothesis",
    "assume",
    "propose",
    "predict",
    "if",
    "then",
    "expect",
    "conjecture",
    "theory",
    "suppose",
    "hipótesis",
    "asumir",
    "proponer",
    "predecir",
    "si",
    "entonces",
    "conjetura",
    "teoría",
    "suponer",
];

const VERIFY_PATTERNS: &[&str] = &[
    "verify",
    "test",
    "assert",
    "validate",
    "check",
    "confirm",
    "prove",
    "disprove",
    "evidence",
    "measure",
    "experiment",
    "verificar",
    "probar",
    "validar",
    "comprobar",
    "confirmar",
    "demostrar",
    "evidencia",
    "medir",
    "experimento",
];

const SYNTHESIZE_PATTERNS: &[&str] = &[
    "synthesize",
    "conclude",
    "summary",
    "therefore",
    "recommend",
    "final",
    "decision",
    "integrate",
    "consolidate",
    "lesson",
    "sintetizar",
    "concluir",
    "resumen",
    "por lo tanto",
    "recomendar",
    "final",
    "decisión",
    "integrar",
    "consolidar",
    "lección",
];

/// Validate that thought content matches its declared stage.
pub fn validate_stage_alignment(
    thought: &str,
    declared_stage: CognitiveStage,
) -> StageAlignmentResult {
    let lower = thought.to_lowercase();

    // Count matches for each stage
    let scores = [
        (
            CognitiveStage::Define,
            count_matches(&lower, DEFINE_PATTERNS),
        ),
        (
            CognitiveStage::Research,
            count_matches(&lower, RESEARCH_PATTERNS),
        ),
        (
            CognitiveStage::Analyze,
            count_matches(&lower, ANALYZE_PATTERNS),
        ),
        (
            CognitiveStage::Hypothesize,
            count_matches(&lower, HYPOTHESIZE_PATTERNS),
        ),
        (
            CognitiveStage::Verify,
            count_matches(&lower, VERIFY_PATTERNS),
        ),
        (
            CognitiveStage::Synthesize,
            count_matches(&lower, SYNTHESIZE_PATTERNS),
        ),
    ];

    // Find the stage with most matches
    let total_matches: usize = scores.iter().map(|(_, c)| c).sum();
    if total_matches == 0 {
        // No detectable patterns — give benefit of doubt
        return StageAlignmentResult {
            declared: declared_stage,
            detected: declared_stage,
            alignment: 0.5,
            warning: None,
        };
    }

    let (detected_stage, detected_count) = scores.iter().max_by_key(|(_, c)| *c).unwrap();

    // Declared stage match count
    let declared_count = scores
        .iter()
        .find(|(s, _)| *s == declared_stage)
        .map(|(_, c)| *c)
        .unwrap_or(0);

    // Alignment: ratio of declared stage matches to detected stage matches
    let alignment = if *detected_count > 0 {
        declared_count as f64 / *detected_count as f64
    } else {
        1.0
    };
    let alignment = alignment.clamp(0.0, 1.0);

    // Generate warning if significant misalignment
    let warning = if alignment < 0.5 && *detected_count > 2 {
        Some(format!(
            "⚠️ Stage mismatch: declaraste {:?} pero el contenido sugiere {:?} ({} vs {} patterns). Ajusta el stage o el contenido.",
            declared_stage, detected_stage, declared_count, detected_count
        ))
    } else {
        None
    };

    StageAlignmentResult {
        declared: declared_stage,
        detected: *detected_stage,
        alignment,
        warning,
    }
}

/// Count how many patterns from the list appear in the text.
fn count_matches(text: &str, patterns: &[&str]) -> usize {
    patterns.iter().filter(|p| text.contains(**p)).count()
}

/// G7: Logical Step Validity Score (ReasonEval 2024, RECEVAL 2024).
///
/// Measures whether a reasoning step follows logically from prior context.
/// Returns a validity score (0.0-1.0) and optional warning.
///
/// 4 dimensions:
/// - Premise reference (30%): cites prior reasoning with "since", "because"
/// - Conclusion support (30%): draws conclusions with "therefore", "thus"
/// - Backward reference (20%): explicitly references prior thought number
/// - No logic gaps (20%): absence of hand-waving indicators
pub fn validate_logical_validity(thought: &str, thought_number: usize) -> (f64, Option<String>) {
    // First 2 thoughts get a pass — no prior context to validate against
    if thought_number <= 2 {
        return (0.5, None);
    }

    let lower = thought.to_lowercase();
    let mut validity = 0.0;

    // D1: Premise reference (30%)
    let premise_markers = [
        "since",
        "because",
        "given that",
        "from",
        "as shown",
        "based on",
        "according to",
        "following from",
        "considering",
        "building on",
        "extending",
        "as established",
    ];
    if premise_markers.iter().any(|m| lower.contains(m)) {
        validity += 0.30;
    }

    // D2: Conclusion support (30%)
    let conclusion_markers = [
        "therefore",
        "thus",
        "this means",
        "which implies",
        "so ",
        "hence",
        "consequently",
        "we can conclude",
        "it follows",
        "this confirms",
        "this shows",
        "this demonstrates",
    ];
    if conclusion_markers.iter().any(|m| lower.contains(m)) {
        validity += 0.30;
    }

    // D3: Backward reference (20%) — references prior thought numbers
    let has_back_ref = (1..thought_number).any(|n| {
        lower.contains(&format!("thought {}", n))
            || lower.contains(&format!("step {}", n))
            || lower.contains(&format!("#{}", n))
            || lower.contains(&format!("thought #{}", n))
    });
    if has_back_ref || lower.contains("previous") || lower.contains("earlier") {
        validity += 0.20;
    }

    // D4: No logic gaps (20%) — absence of hand-waving
    let gap_indicators = [
        "somehow",
        "magically",
        "just works",
        "obviously",
        "clearly",
        "trivially",
        "simply put",
    ];
    if !gap_indicators.iter().any(|m| lower.contains(m)) {
        validity += 0.20;
    }

    let warning = if validity < 0.30 {
        Some(format!(
            "⚠️ G7: Low logical validity ({:.0}%) in thought #{} — add premise-conclusion structure (because X → therefore Y)",
            validity * 100.0, thought_number
        ))
    } else {
        None
    };

    (validity, warning)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aligned_define_stage() {
        let result = validate_stage_alignment(
            "Let me define the problem scope and context requirements for this system",
            CognitiveStage::Define,
        );
        assert!(
            result.alignment >= 0.5,
            "Define content should align with Define stage: {:.2}",
            result.alignment
        );
        assert!(result.warning.is_none());
    }

    #[test]
    fn test_aligned_verify_stage() {
        let result = validate_stage_alignment(
            "I will verify this hypothesis by testing the assert statements and checking the evidence",
            CognitiveStage::Verify,
        );
        assert!(
            result.alignment >= 0.5,
            "Verify content should align with Verify stage: {:.2}",
            result.alignment
        );
    }

    #[test]
    fn test_misaligned_define_vs_verify() {
        let result = validate_stage_alignment(
            "Let me verify and test each assertion. Check the evidence and confirm the results.",
            CognitiveStage::Define,
        );
        assert!(
            result.alignment < 0.8,
            "Verify content should NOT align with Define stage: {:.2}",
            result.alignment
        );
    }

    #[test]
    fn test_no_patterns_neutral() {
        let result = validate_stage_alignment("x = 42; y = x * 2", CognitiveStage::Analyze);
        assert_eq!(
            result.alignment, 0.5,
            "No patterns should give neutral alignment"
        );
    }

    #[test]
    fn test_synthesize_patterns() {
        let result = validate_stage_alignment(
            "In conclusion, let me synthesize the findings. Therefore my recommendation is to consolidate lessons learned.",
            CognitiveStage::Synthesize,
        );
        assert!(result.alignment >= 0.5);
        assert_eq!(result.detected, CognitiveStage::Synthesize);
    }

    #[test]
    fn test_spanish_patterns() {
        let result = validate_stage_alignment(
            "Vamos a verificar y validar esta hipótesis. Comprobar la evidencia es necesario.",
            CognitiveStage::Verify,
        );
        assert!(
            result.alignment >= 0.5,
            "Spanish patterns should match: {:.2}",
            result.alignment
        );
    }
}
