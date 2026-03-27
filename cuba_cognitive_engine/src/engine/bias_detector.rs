// src/engine/bias_detector.rs
//
// R7: Cognitive Bias Detection — Agent-Reported Only
//
// Simplified: keyword-based heuristic detection removed due to high
// false-positive rate in technical discourse (e.g. "first" triggering
// anchoring, "clearly" triggering confirmation). Claude's native
// reasoning already avoids these patterns.
//
// Retains: agent self-reported bias via `biasDetected` parameter.

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
            BiasType::Anchoring => "Anchoring Bias",
            BiasType::Confirmation => "Confirmation Bias",
            BiasType::Availability => "Availability Bias",
            BiasType::SunkCost => "Sunk Cost Bias",
            BiasType::Bandwagon => "Bandwagon Bias",
        }
    }
}

/// Detect biases — only accepts agent-reported bias.
///
/// Keyword-based automatic detection was removed (high false-positive
/// rate in technical discourse). The `biasDetected` parameter allows
/// the agent to self-report when it recognizes its own bias.
pub fn detect_biases(
    _thought: &str,
    _thought_number: usize,
    _previous_thoughts: &[&str],
    agent_reported_bias: Option<&str>,
) -> Vec<DetectedBias> {
    let mut biases = Vec::new();

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
            biases.push(DetectedBias {
                bias_type: bt,
                confidence: 0.9,
                explanation: "Self-reported by the agent during reasoning",
                suggestion: "Agent acknowledged this bias — take corrective action",
            });
        }
    }

    biases
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_bias_without_report() {
        let biases = detect_biases(
            "Obviously the answer is to use PostgreSQL for everything",
            1,
            &[],
            None,
        );
        assert!(biases.is_empty(), "No auto-detection — should be empty without agent report");
    }

    #[test]
    fn test_agent_reported_bias() {
        let biases = detect_biases("Let me explore this approach", 3, &[], Some("confirmation"));
        assert!(biases
            .iter()
            .any(|b| b.bias_type == BiasType::Confirmation && b.confidence == 0.9));
    }

    #[test]
    fn test_agent_reported_sunk_cost() {
        let biases = detect_biases("continuing this path", 5, &[], Some("sunk_cost"));
        assert!(biases.iter().any(|b| b.bias_type == BiasType::SunkCost));
    }

    #[test]
    fn test_unknown_bias_type_ignored() {
        let biases = detect_biases("some thought", 1, &[], Some("unknown_bias"));
        assert!(biases.is_empty());
    }
}
