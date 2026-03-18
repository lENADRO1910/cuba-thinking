// src/engine/formatter.rs
//
// R12: Compact Output Formatter (V14)
//
// Produces actionable, concise output from the cognitive engine.
// Only shows metrics that matter — no noise, no padding.
//
// Output structure:
// 1. Stage indicator + EWMA health
// 2. Quality dimensions (only weak ones flagged)
// 3. Anti-hallucination warnings (if any)
// 4. Bias alerts (if detected)
// 5. Memory instructions (if applicable)
// 6. GoT topology (compact)

use crate::engine::anti_hallucination::HallucinationVerdict;
use crate::engine::bias_detector::DetectedBias;
use crate::engine::budget::BudgetMode;
use crate::engine::ewma_reward::EwmaTracker;
use crate::engine::memory_bridge::MemoryInstruction;
use crate::engine::metacognition::MetacognitiveReport;
use crate::engine::quality_metrics::QualityScores;
use crate::engine::stage_engine::{CognitiveStage, StageSession};
use crate::engine::thought_graph::TopologySummary;

/// Format the complete engine output into a compact, actionable string.
#[allow(clippy::too_many_arguments)]
pub fn format_engine_output(
    stage: CognitiveStage,
    session: &StageSession,
    quality: &QualityScores,
    ewma: &EwmaTracker,
    verdict: &HallucinationVerdict,
    metacog: &MetacognitiveReport,
    biases: &[DetectedBias],
    memory_instructions: &[MemoryInstruction],
    topology: Option<&TopologySummary>,
    _thought_number: usize,
    is_sandbox_result: bool,
    sandbox_output: Option<&str>,
    budget: BudgetMode,
) -> String {
    let mut output = String::with_capacity(2048);

    // ─── Header: Stage + EWMA Health + Budget ────────────────────
    output.push_str(&format!(
        "{} **Stage {}/6: {:?}** | {} | EWMA: {:.0}% | Trust: {:.0}%\n",
        stage.emoji(),
        stage.index() + 1,
        stage,
        budget.label(),
        ewma.percentage(),
        verdict.trust_score * 100.0,
    ));

    // ─── Quality Dimensions (only flag weak ones) ────────────────
    let weak_threshold = 0.4;
    let mut weak_dims = Vec::new();
    if quality.clarity < weak_threshold {
        weak_dims.push(format!("Clarity {:.0}%", quality.clarity * 100.0));
    }
    if quality.depth < weak_threshold {
        weak_dims.push(format!("Depth {:.0}%", quality.depth * 100.0));
    }
    if quality.breadth < weak_threshold {
        weak_dims.push(format!("Breadth {:.0}%", quality.breadth * 100.0));
    }
    if quality.logic < weak_threshold {
        weak_dims.push(format!("Logic {:.0}%", quality.logic * 100.0));
    }
    if quality.relevance < weak_threshold {
        weak_dims.push(format!("Relevance {:.0}%", quality.relevance * 100.0));
    }
    if quality.actionability < weak_threshold {
        weak_dims.push(format!(
            "Actionability {:.0}%",
            quality.actionability * 100.0
        ));
    }

    if !weak_dims.is_empty() {
        output.push_str(&format!("📊 Weak dimensions: {}\n", weak_dims.join(", ")));
    }

    // ─── Anti-Hallucination Warnings ─────────────────────────────
    for warning in &verdict.warnings {
        output.push_str(&format!("{}\n", warning));
    }

    // ─── Metacognitive Warnings ──────────────────────────────────
    for warning in &metacog.warnings {
        output.push_str(&format!("{}\n", warning));
    }

    // ─── Bias Alerts ─────────────────────────────────────────────
    if !biases.is_empty() {
        for bias in biases {
            output.push_str(&format!(
                "{} (confidence {:.0}%): {}\n  → {}\n",
                bias.bias_type.label(),
                bias.confidence * 100.0,
                bias.explanation,
                bias.suggestion,
            ));
        }
    }

    // ─── Sandbox Result (if Python was executed) ─────────────────
    if is_sandbox_result {
        if let Some(sandbox_out) = sandbox_output {
            output.push_str(&format!(
                "🐍 **Sandbox Result**:\n```\n{}\n```\n",
                sandbox_out
            ));
        }
    }

    // ─── Session Progress ────────────────────────────────────────
    if !session.assumptions.is_empty() {
        output.push_str(&format!(
            "📋 Assumptions: {} recorded\n",
            session.assumptions.len(),
        ));
    }

    // ─── GoT Topology (compact) ──────────────────────────────────
    if let Some(topo) = topology {
        if topo.total_nodes > 1 {
            output.push_str(&format!("{}\n", topo.display()));
        }
    }

    // ─── Memory Instructions ─────────────────────────────────────
    if !memory_instructions.is_empty() {
        output.push_str("\n🧠 **Memory Instructions**:\n");
        for (i, instr) in memory_instructions.iter().enumerate() {
            output.push_str(&format!(
                "{}. Invoke `{}` — {}\n",
                i + 1,
                instr.tool_name,
                instr.reason
            ));
        }
    }

    // ─── Rejection Notice ────────────────────────────────────────
    if verdict.should_reject {
        output.push_str(
            "\n🚫 **REJECTION**: Quality below minimum threshold. Backtrack recommended.\n",
        );
    }

    // ─── Early Stop Recommendation ───────────────────────────────
    if verdict.should_early_stop {
        output.push_str("⏹️ **Early stop recommended**: Stagnation or fatigue detected.\n");
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::budget::BudgetMode;

    #[test]
    fn test_formatter_includes_stage() {
        let session = StageSession::new();
        let quality = QualityScores {
            clarity: 0.7,
            depth: 0.6,
            breadth: 0.5,
            logic: 0.8,
            relevance: 0.7,
            actionability: 0.6,
        };
        let ewma = EwmaTracker::new(BudgetMode::Balanced);
        let verdict = HallucinationVerdict {
            trust_score: 0.75,
            layers: crate::engine::anti_hallucination::LayerResults {
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
        };
        let metacog = MetacognitiveReport {
            filler_ratio: 0.05,
            content_word_ratio: 0.7,
            claim_density: 0.5,
            fallacies: vec![],
            has_dialectical: true,
            warnings: vec![],
        };

        let output = format_engine_output(
            CognitiveStage::Analyze,
            &session,
            &quality,
            &ewma,
            &verdict,
            &metacog,
            &[],
            &[],
            None,
            3,
            false,
            None,
            BudgetMode::Balanced,
        );

        assert!(output.contains("Stage 3/6"));
        assert!(output.contains("Analyze"));
        assert!(output.contains("EWMA"));
    }

    #[test]
    fn test_formatter_flags_weak_dimensions() {
        let session = StageSession::new();
        let quality = QualityScores {
            clarity: 0.2,
            depth: 0.1,
            breadth: 0.8,
            logic: 0.9,
            relevance: 0.7,
            actionability: 0.3,
        };
        let ewma = EwmaTracker::new(BudgetMode::Balanced);
        let verdict = HallucinationVerdict {
            trust_score: 0.5,
            layers: crate::engine::anti_hallucination::LayerResults {
                assumption_count: 0,
                confidence_calibrated: true,
                cove_passed: true,
                evidence_strength: 0.5,
                claim_count: 0,
                grounding_ratio: 1.0,
                ewma_above_threshold: true,
                no_contradictions: true,
                warmup_suppressed: false,
            },
            warnings: vec![],
            should_reject: false,
            should_early_stop: false,
        };
        let metacog = MetacognitiveReport {
            filler_ratio: 0.0,
            content_word_ratio: 0.7,
            claim_density: 0.5,
            fallacies: vec![],
            has_dialectical: true,
            warnings: vec![],
        };

        let output = format_engine_output(
            CognitiveStage::Define,
            &session,
            &quality,
            &ewma,
            &verdict,
            &metacog,
            &[],
            &[],
            None,
            1,
            false,
            None,
            BudgetMode::Balanced,
        );

        assert!(output.contains("Clarity"));
        assert!(output.contains("Depth"));
        assert!(output.contains("Actionability"));
        assert!(!output.contains("Logic")); // 0.9 is not weak
    }
}
