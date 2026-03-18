// src/engine/bias_detector.rs
//
// R7: Cognitive Bias Detection (5 types)
//
// Identifies common cognitive biases in reasoning based on
// Kahneman & Tversky (1974), "Judgment under Uncertainty: Heuristics and Biases"
//
// Detected biases:
// 1. Anchoring — Over-reliance on first information encountered
// 2. Confirmation — Favoring information that confirms existing beliefs
// 3. Availability — Overweighting easily recalled examples
// 4. Sunk Cost — Continuing due to past investment rather than future value
// 5. Bandwagon — Following popular opinion without independent analysis

use serde::Serialize;

/// Detected bias with explanation and suggestion.
#[derive(Debug, Clone, Serialize)]
pub struct DetectedBias {
    pub bias_type: BiasType,
    pub confidence: f64,
    pub explanation: &'static str,
    pub suggestion: &'static str,
}

/// Cognitive bias categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BiasType {
    Anchoring,
    Confirmation,
    Availability,
    SunkCost,
    Bandwagon,
}

impl BiasType {
    pub fn label(self) -> &'static str {
        match self {
            BiasType::Anchoring => "🔗 Anchoring Bias",
            BiasType::Confirmation => "✅ Confirmation Bias",
            BiasType::Availability => "📰 Availability Bias",
            BiasType::SunkCost => "💸 Sunk Cost Bias",
            BiasType::Bandwagon => "🐑 Bandwagon Bias",
        }
    }
}

/// Analyze thought text for cognitive biases.
/// Returns detected biases with confidence scores and actionable suggestions.
///
/// Refactored: each bias type is a standalone checker (CC=1-2).
/// detect_biases CC: ~4 (agent-reported match + extend).
pub fn detect_biases(
    thought: &str,
    thought_number: usize,
    previous_thoughts: &[&str],
    agent_reported_bias: Option<&str>,
) -> Vec<DetectedBias> {
    let lower = thought.to_lowercase();

    // Collect all detected biases from individual checkers
    let mut biases: Vec<DetectedBias> = [
        check_anchoring_bias(&lower, thought_number),
        check_confirmation_bias(&lower, thought_number),
        check_availability_bias(&lower),
        check_sunk_cost_bias(&lower),
        check_bandwagon_bias(&lower),
        check_repetition_bias(&lower, previous_thoughts),
    ]
    .into_iter()
    .flatten()
    .collect();

    // ─── Agent-Reported Bias ─────────────────────────────────────
    if let Some(reported) = agent_reported_bias {
        let bias_type = match reported.to_lowercase().as_str() {
            "anchoring" => Some(BiasType::Anchoring),
            "confirmation" => Some(BiasType::Confirmation),
            "availability" => Some(BiasType::Availability),
            "sunk_cost" | "sunkcost" => Some(BiasType::SunkCost),
            "bandwagon" => Some(BiasType::Bandwagon),
            _ => None,
        };
        if let Some(bt) = bias_type {
            if !biases.iter().any(|b| b.bias_type == bt) {
                biases.push(DetectedBias {
                    bias_type: bt,
                    confidence: 0.9,
                    explanation: "Self-reported by the agent during reasoning",
                    suggestion: "Agent acknowledged this bias — take corrective action",
                });
            }
        }
    }

    biases
}

// ─── Bias Checker Helpers (CC=1-2 each) ──────────────────────────

/// Anchoring: early fixation on a single solution without exploring alternatives.
fn check_anchoring_bias(lower: &str, thought_number: usize) -> Option<DetectedBias> {
    if thought_number > 2 {
        return None;
    }

    let solution_markers = [
        "the answer is",
        "we should",
        "the solution is",
        "obviously",
        "clearly",
        "la respuesta es",
        "debemos",
        "la solución es",
        "obviamente",
        "claramente",
    ];
    let is_premature = solution_markers.iter().any(|m| lower.contains(m));
    let no_alternatives = !lower.contains("alternative")
        && !lower.contains("option")
        && !lower.contains("alternativa")
        && !lower.contains("opción");

    if is_premature && no_alternatives {
        Some(DetectedBias {
            bias_type: BiasType::Anchoring,
            confidence: 0.7,
            explanation: "Proposing a solution in early thoughts without exploring alternatives",
            suggestion: "Consider at least 2-3 alternative approaches before committing",
        })
    } else {
        None
    }
}

/// Confirmation: only supporting evidence, no counter-arguments.
fn check_confirmation_bias(lower: &str, thought_number: usize) -> Option<DetectedBias> {
    let support_markers = [
        "confirms",
        "supports",
        "proves",
        "validates",
        "agrees with",
        "confirma",
        "soporta",
        "prueba",
        "valida",
        "coincide",
    ];
    let counter_markers = [
        "however",
        "but",
        "although",
        "counter",
        "disagree",
        "challenge",
        "risk",
        "downside",
        "limitation",
        "weakness",
        "sin embargo",
        "pero",
        "aunque",
        "riesgo",
        "desventaja",
        "limitación",
        "debilidad",
    ];

    let has_support = support_markers.iter().any(|m| lower.contains(m));
    let has_counter = counter_markers.iter().any(|m| lower.contains(m));

    if has_support && !has_counter && thought_number > 3 {
        Some(DetectedBias {
            bias_type: BiasType::Confirmation,
            confidence: 0.6,
            explanation: "Only supporting evidence found, no counter-arguments considered",
            suggestion: "Actively seek disconfirming evidence or opposing viewpoints",
        })
    } else {
        None
    }
}

/// Availability: over-reliance on anecdotal/recalled examples.
fn check_availability_bias(lower: &str) -> Option<DetectedBias> {
    let anecdotal_markers = [
        "i've seen",
        "in my experience",
        "usually",
        "typically",
        "everyone knows",
        "common knowledge",
        "obviously",
        "he visto",
        "en mi experiencia",
        "usualmente",
        "típicamente",
        "todos saben",
        "conocimiento común",
    ];
    let anecdotal_count = anecdotal_markers
        .iter()
        .filter(|m| lower.contains(**m))
        .count();

    if anecdotal_count >= 2 {
        Some(DetectedBias {
            bias_type: BiasType::Availability,
            confidence: 0.5,
            explanation: "Reasoning relies heavily on anecdotal/recalled examples rather than data",
            suggestion: "Ground claims with specific data, measurements, or verified sources",
        })
    } else {
        None
    }
}

/// Sunk cost: continuing with a failing approach due to prior investment.
fn check_sunk_cost_bias(lower: &str) -> Option<DetectedBias> {
    let sunk_cost_markers = [
        "already invested",
        "too far to stop",
        "we've spent",
        "can't abandon",
        "committed to",
        "already built",
        "ya invertimos",
        "muy tarde para",
        "ya gastamos",
        "no podemos abandonar",
        "comprometidos con",
    ];
    if sunk_cost_markers.iter().any(|m| lower.contains(m)) {
        Some(DetectedBias {
            bias_type: BiasType::SunkCost,
            confidence: 0.8,
            explanation: "Decision influenced by prior investment rather than future value",
            suggestion: "Evaluate the decision based solely on future costs and benefits",
        })
    } else {
        None
    }
}

/// Bandwagon: citing popularity without justification.
fn check_bandwagon_bias(lower: &str) -> Option<DetectedBias> {
    let bandwagon_markers = [
        "everyone uses",
        "industry standard",
        "most popular",
        "best practice",
        "widely adopted",
        "trending",
        "todos usan",
        "estándar de la industria",
        "más popular",
        "mejor práctica",
        "ampliamente adoptado",
    ];
    let no_justification = !lower.contains("because")
        && !lower.contains("since")
        && !lower.contains("due to")
        && !lower.contains("porque")
        && !lower.contains("debido a");

    if bandwagon_markers.iter().any(|m| lower.contains(m)) && no_justification {
        Some(DetectedBias {
            bias_type: BiasType::Bandwagon,
            confidence: 0.5,
            explanation: "Citing popularity without justifying why it fits this specific case",
            suggestion: "Explain WHY the popular choice is appropriate for THIS specific context",
        })
    } else {
        None
    }
}

/// Repetition anchoring: structural repetition across previous thoughts.
///
/// V7 (P3-D): Uses Jaccard similarity over full word bags instead of
/// first-10-words sequential matching. Sequential `.zip()` only caught
/// thoughts starting identically; Jaccard catches semantic repetition
/// even when word order differs (reasoning loops with varied openings).
fn check_repetition_bias(lower: &str, previous_thoughts: &[&str]) -> Option<DetectedBias> {
    if previous_thoughts.len() < 3 {
        return None;
    }

    // Build word set for current thought (stopwords excluded for signal)
    let stopwords = [
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "to", "of", "in", "for", "on",
        "with", "at", "by", "from", "it", "this", "that", "and", "or", "but", "not", "as", "el",
        "la", "los", "las", "un", "una", "de", "en", "con", "por", "para", "es", "son", "no",
        "que", "se", "del",
    ];

    let current_words: std::collections::HashSet<&str> = lower
        .split_whitespace()
        .filter(|w| w.len() > 2 && !stopwords.contains(w))
        .collect();

    if current_words.len() < 3 {
        return None; // Too few content words to judge
    }

    let similar_count = previous_thoughts
        .iter()
        .filter(|prev| {
            let prev_lower = prev.to_lowercase();
            let prev_words: std::collections::HashSet<&str> = prev_lower
                .split_whitespace()
                .filter(|w| w.len() > 2 && !stopwords.contains(w))
                .collect();

            if prev_words.is_empty() {
                return false;
            }

            // Jaccard coefficient: |A ∩ B| / |A ∪ B|
            let intersection = current_words.intersection(&prev_words).count();
            let union = current_words.union(&prev_words).count();
            let jaccard = intersection as f64 / union.max(1) as f64;

            jaccard > 0.50 // Stricter than sequential (0.6) since Jaccard is already tighter
        })
        .count();

    if similar_count >= 2 {
        Some(DetectedBias {
            bias_type: BiasType::Anchoring,
            confidence: 0.6,
            explanation: "Thought pattern is repeating — may be stuck in a reasoning loop",
            suggestion: "Try a fundamentally different approach or perspective",
        })
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anchoring_early_solution() {
        let biases = detect_biases(
            "Obviously the answer is to use PostgreSQL for everything",
            1,
            &[],
            None,
        );
        assert!(biases.iter().any(|b| b.bias_type == BiasType::Anchoring));
    }

    #[test]
    fn test_confirmation_no_counter() {
        let biases = detect_biases(
            "This confirms our hypothesis and validates the approach",
            5,
            &[],
            None,
        );
        assert!(biases.iter().any(|b| b.bias_type == BiasType::Confirmation));
    }

    #[test]
    fn test_sunk_cost() {
        let biases = detect_biases(
            "We've already invested too much time, we can't abandon this approach",
            4,
            &[],
            None,
        );
        assert!(biases.iter().any(|b| b.bias_type == BiasType::SunkCost));
    }

    #[test]
    fn test_no_bias_balanced_thought() {
        let biases = detect_biases(
            "However, there are risks with this approach. Although it works, we should consider alternatives because the scalability might be limited.",
            5,
            &[],
            None,
        );
        // Should not detect confirmation bias (has counter-arguments)
        assert!(!biases.iter().any(|b| b.bias_type == BiasType::Confirmation));
    }

    #[test]
    fn test_agent_reported_bias() {
        let biases = detect_biases("Let me explore this approach", 3, &[], Some("confirmation"));
        assert!(biases
            .iter()
            .any(|b| b.bias_type == BiasType::Confirmation && b.confidence == 0.9));
    }
}
