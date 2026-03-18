// src/engine/corrective_directives.rs
//
// Phase 5B: Corrective Feedback Directives (G2, G7, G9, G12)
//
// Transforms passive metric reporting into active, prescriptive corrections.
// Instead of "Lógica 25%", outputs "⚡ Estructura con conectivos causales".
//
// Based on SEAL (Intel/UT Austin) and FS-C (Filter Supervisor-SelfCorrection)
// frameworks for improving CoT reasoning quality.
//
// Severity levels:
// - INFO: Suggestion for improvement (quality 30-40%)
// - WARNING: Strong recommendation (quality 20-30%)
// - CORRECTION: Mandatory fix (quality < 20% or critical failure)

use crate::engine::anti_hallucination::HallucinationVerdict;
use crate::engine::metacognition::MetacognitiveReport;
use crate::engine::quality_metrics::QualityScores;
use serde::Serialize;

/// Threshold below which a dimension triggers a corrective directive.
const DIRECTIVE_THRESHOLD: f64 = 0.40;
/// Threshold below which a directive becomes mandatory.
const CRITICAL_THRESHOLD: f64 = 0.20;

/// Severity of a corrective directive.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub enum Severity {
    Info,
    Warning,
    Correction,
}

impl Severity {
    pub fn emoji(&self) -> &'static str {
        match self {
            Severity::Info => "💡",
            Severity::Warning => "⚠️",
            Severity::Correction => "⚡",
        }
    }
}

/// A single corrective directive for the agent.
#[derive(Debug, Clone, Serialize)]
pub struct Directive {
    pub severity: Severity,
    pub dimension: &'static str,
    pub instruction: String,
}

impl Directive {
    pub fn display(&self) -> String {
        format!(
            "{} **{}**: {}",
            self.severity.emoji(),
            self.dimension,
            self.instruction
        )
    }
}

/// Generate all applicable corrective directives based on analysis results.
///
/// Analyzes quality scores, hallucination verdict, and metacognitive report
/// to produce specific, actionable instructions for the agent.
/// F18: When `is_code` is true, NL-specific directives (Depth, Logic, Clarity,
/// Filler, Dialectics) are suppressed — they produce noise for code inputs.
pub fn generate_directives(
    quality: &QualityScores,
    verdict: &HallucinationVerdict,
    metacog: &MetacognitiveReport,
    claim_count: usize,
    is_code: bool,
) -> Vec<Directive> {
    let mut directives = Vec::new();

    // ─── Quality Dimension Directives ────────────────────────────
    // F18: Depth/Logic/Clarity are NL-specific; skip for code inputs
    if !is_code && quality.depth < DIRECTIVE_THRESHOLD {
        directives.push(Directive {
            severity: severity_for(quality.depth),
            dimension: "Depth",
            instruction:
                "Add causal chains: 'because X → therefore Y'. Include second-order reasoning."
                    .to_string(),
        });
    }

    if !is_code && quality.logic < DIRECTIVE_THRESHOLD {
        directives.push(Directive {
            severity: severity_for(quality.logic),
            dimension: "Logic",
            instruction: "Structure with connectives: 'first...then...however...therefore'. Include at least 3 connective types.".to_string(),
        });
    }

    if quality.actionability < DIRECTIVE_THRESHOLD {
        directives.push(Directive {
            severity: severity_for(quality.actionability),
            dimension: "Actionability",
            instruction:
                "Include specific data: numbers, file paths, function names, concrete measurements."
                    .to_string(),
        });
    }

    if !is_code && quality.clarity < DIRECTIVE_THRESHOLD {
        directives.push(Directive {
            severity: severity_for(quality.clarity),
            dimension: "Clarity",
            instruction: "Reduce vocabulary repetition. Vary sentence opening words. Use synonyms."
                .to_string(),
        });
    }

    if quality.breadth < DIRECTIVE_THRESHOLD {
        directives.push(Directive {
            severity: severity_for(quality.breadth),
            dimension: "Breadth",
            instruction: "Expand analysis to more problem dimensions. Consider technical, operational and business aspects.".to_string(),
        });
    }

    if quality.relevance < DIRECTIVE_THRESHOLD {
        directives.push(Directive {
            severity: severity_for(quality.relevance),
            dimension: "Relevance",
            instruction: "Your reasoning is diverging from the main topic. Refocus on the original hypothesis using its key terms.".to_string(),
        });
    }

    // ─── Grounding Directives ────────────────────────────────────
    if verdict.layers.grounding_ratio < 0.3 && verdict.layers.claim_count > 0 {
        let ungrounded = verdict.layers.claim_count
            - (verdict.layers.grounding_ratio * verdict.layers.claim_count as f64) as usize;
        directives.push(Directive {
            severity: Severity::Warning,
            dimension: "Grounding",
            instruction: format!(
                "{} of {} claims are unsupported. Add evidence: 'according to X', 'verified in Y', quantitative data.",
                ungrounded, verdict.layers.claim_count
            ),
        });
    }

    // ─── Metacognitive Directives ────────────────────────────────
    // F18: Filler detection not meaningful for code
    if !is_code && metacog.filler_ratio > 0.15 {
        directives.push(Directive {
            severity: Severity::Warning,
            dimension: "Filler",
            instruction: "Too much filler text. Remove: 'in other words', 'basically', 'to be clear'. Be direct.".to_string(),
        });
    }

    // F18: Dialectics not meaningful for code
    if !is_code && !metacog.has_dialectical {
        directives.push(Directive {
            severity: Severity::Info,
            dimension: "Dialectics",
            instruction: "Consider counter-arguments before concluding. What evidence contradicts your position?".to_string(),
        });
    }

    if !metacog.fallacies.is_empty() {
        for fallacy in &metacog.fallacies {
            directives.push(Directive {
                severity: Severity::Warning,
                dimension: "Fallacy",
                instruction: format!(
                    "Fallacy detected: {} — '{}'. Review the logic of that argument.",
                    fallacy.fallacy_type, fallacy.evidence
                ),
            });
        }
    }

    // ─── Complexity / Decomposition Directive (G9) ───────────────
    if claim_count > 5 {
        directives.push(Directive {
            severity: Severity::Info,
            dimension: "Decomposition",
            instruction: format!(
                "Complex problem detected ({} claims). Decompose into independent sub-problems before solving.",
                claim_count
            ),
        });
    }

    // ─── Confidence Integrity Directive ──────────────────────────
    if !verdict.layers.confidence_calibrated {
        directives.push(Directive {
            severity: Severity::Warning,
            dimension: "Confidence",
            instruction: "Your declared confidence does not match the evidence. Adjust to a level that reflects real uncertainty.".to_string(),
        });
    }

    directives
}

/// G5: Reflexion-Style Self-Evaluation Directive (Shinn et al., 2023).
///
/// Injects an explicit self-evaluation prompt when trust is low,
/// forcing the agent to question its assumptions before continuing.
/// Only activates after thought #3 with trust < 50% to avoid overhead.
pub fn generate_reflexion_directive(trust_score: f64, thought_number: usize) -> Option<Directive> {
    if trust_score >= 0.50 || thought_number <= 3 {
        return None;
    }

    Some(Directive {
        severity: Severity::Warning,
        dimension: "Reflexion",
        instruction: "⟳ REFLECT: Before continuing, evaluate: \
            (1) What assumption am I most uncertain about? \
            (2) What evidence would change my conclusion? \
            (3) Am I anchoring on my first hypothesis?"
            .to_string(),
    })
}

/// NEW-1 + Vector 5: Confidence Oscillation — Abductive Shift Directive.
///
/// Injected when is_confidence_oscillating() returns true — the model's
/// confidence is rapidly alternating between high and low values,
/// indicating it's trapped in a local minimum between competing hypotheses.
///
/// Uses abductive logic (inference to the best explanation) instead of
/// forcing arbitrary choice — which would cause confirmation bias in
/// a 50/50 decision. The model must synthesize an orthogonal H3.
pub fn generate_oscillation_directive() -> Directive {
    Directive {
        severity: Severity::Correction,
        dimension: "Oscillation",
        instruction: "🔄 ATTRACTOR CONFLICT: Your confidence is oscillating rapidly. \
            You are trapped in a local minimum between competing models. \
            DO NOT arbitrarily pick one. Instead: \
            (1) Synthesize an ORTHOGONAL hypothesis (H3) that explains WHY \
            your previous hypotheses contradict each other. \
            (2) What evidence would distinguish H3 from both alternatives? \
            (3) Test that evidence before continuing."
            .to_string(),
    }
}

/// Determine severity based on score value.
fn severity_for(score: f64) -> Severity {
    if score < CRITICAL_THRESHOLD {
        Severity::Correction
    } else if score < 0.30 {
        Severity::Warning
    } else {
        Severity::Info
    }
}

/// Check if any directive is mandatory (Correction severity).
pub fn has_mandatory_corrections(directives: &[Directive]) -> bool {
    directives
        .iter()
        .any(|d| d.severity == Severity::Correction)
}

/// Format all directives for display.
pub fn format_directives(directives: &[Directive]) -> String {
    if directives.is_empty() {
        return String::new();
    }

    let mut output = String::from("📋 **Corrective Directives**:\n");
    for directive in directives {
        output.push_str(&format!("  {}\n", directive.display()));
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::anti_hallucination::LayerResults;

    fn default_verdict() -> HallucinationVerdict {
        HallucinationVerdict {
            trust_score: 0.75,
            layers: LayerResults {
                assumption_count: 0,
                confidence_calibrated: true,
                cove_passed: true,
                evidence_strength: 0.7,
                claim_count: 3,
                grounding_ratio: 0.8,
                ewma_above_threshold: true,
                no_contradictions: true,
                warmup_suppressed: false,
            },
            warnings: vec![],
            should_reject: false,
            should_early_stop: false,
        }
    }

    fn default_metacog() -> MetacognitiveReport {
        MetacognitiveReport {
            filler_ratio: 0.05,
            content_word_ratio: 0.7,
            claim_density: 0.5,
            fallacies: vec![],
            has_dialectical: true,
            warnings: vec![],
        }
    }

    #[test]
    fn test_no_directives_when_quality_high() {
        let quality = QualityScores {
            clarity: 0.8,
            depth: 0.7,
            breadth: 0.6,
            logic: 0.9,
            relevance: 0.8,
            actionability: 0.7,
        };
        let directives =
            generate_directives(&quality, &default_verdict(), &default_metacog(), 3, false);
        assert!(
            directives.is_empty(),
            "Should have no directives for high quality"
        );
    }

    #[test]
    fn test_depth_directive_when_low() {
        let quality = QualityScores {
            clarity: 0.8,
            depth: 0.1,
            breadth: 0.8,
            logic: 0.9,
            relevance: 0.8,
            actionability: 0.7,
        };
        let directives =
            generate_directives(&quality, &default_verdict(), &default_metacog(), 3, false);
        assert!(!directives.is_empty());
        assert!(directives.iter().any(|d| d.dimension == "Depth"));
        assert!(directives
            .iter()
            .any(|d| d.severity == Severity::Correction));
    }

    #[test]
    fn test_multiple_weak_dimensions() {
        let quality = QualityScores {
            clarity: 0.2,
            depth: 0.15,
            breadth: 0.1,
            logic: 0.25,
            relevance: 0.3,
            actionability: 0.1,
        };
        let directives =
            generate_directives(&quality, &default_verdict(), &default_metacog(), 3, false);
        assert!(
            directives.len() >= 5,
            "Should flag multiple weak dims: {}",
            directives.len()
        );
    }

    #[test]
    fn test_grounding_directive() {
        let quality = QualityScores {
            clarity: 0.8,
            depth: 0.8,
            breadth: 0.8,
            logic: 0.8,
            relevance: 0.8,
            actionability: 0.8,
        };
        let mut verdict = default_verdict();
        verdict.layers.grounding_ratio = 0.1;
        verdict.layers.claim_count = 5;

        let directives = generate_directives(&quality, &verdict, &default_metacog(), 5, false);
        assert!(directives.iter().any(|d| d.dimension == "Grounding"));
    }

    #[test]
    fn test_decomposition_directive_for_complex() {
        let quality = QualityScores {
            clarity: 0.8,
            depth: 0.8,
            breadth: 0.8,
            logic: 0.8,
            relevance: 0.8,
            actionability: 0.8,
        };
        let directives =
            generate_directives(&quality, &default_verdict(), &default_metacog(), 8, false);
        assert!(directives.iter().any(|d| d.dimension == "Decomposition"));
    }

    #[test]
    fn test_filler_directive() {
        let quality = QualityScores {
            clarity: 0.8,
            depth: 0.8,
            breadth: 0.8,
            logic: 0.8,
            relevance: 0.8,
            actionability: 0.8,
        };
        let mut metacog = default_metacog();
        metacog.filler_ratio = 0.25;

        let directives = generate_directives(&quality, &default_verdict(), &metacog, 3, false);
        assert!(directives.iter().any(|d| d.dimension == "Filler"));
    }

    #[test]
    fn test_dialectical_directive() {
        let quality = QualityScores {
            clarity: 0.8,
            depth: 0.8,
            breadth: 0.8,
            logic: 0.8,
            relevance: 0.8,
            actionability: 0.8,
        };
        let mut metacog = default_metacog();
        metacog.has_dialectical = false;

        let directives = generate_directives(&quality, &default_verdict(), &metacog, 3, false);
        assert!(directives.iter().any(|d| d.dimension == "Dialectics"));
    }

    #[test]
    fn test_has_mandatory_corrections() {
        let directives = vec![
            Directive {
                severity: Severity::Info,
                dimension: "Test",
                instruction: "test".to_string(),
            },
            Directive {
                severity: Severity::Correction,
                dimension: "Test2",
                instruction: "test2".to_string(),
            },
        ];
        assert!(has_mandatory_corrections(&directives));
    }

    #[test]
    fn test_no_mandatory_corrections() {
        let directives = vec![Directive {
            severity: Severity::Info,
            dimension: "Test",
            instruction: "test".to_string(),
        }];
        assert!(!has_mandatory_corrections(&directives));
    }

    #[test]
    fn test_severity_levels() {
        assert_eq!(severity_for(0.1), Severity::Correction);
        assert_eq!(severity_for(0.25), Severity::Warning);
        assert_eq!(severity_for(0.35), Severity::Info);
    }
}
