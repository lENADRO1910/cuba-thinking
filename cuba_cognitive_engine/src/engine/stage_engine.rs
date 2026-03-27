// src/engine/stage_engine.rs
//
// R1: 6-Stage Cognitive Engine based on Bloom's Revised Taxonomy
// (Anderson & Krathwohl, 2001)
//
// Provides structured thinking progression: DEFINE → RESEARCH → ANALYZE →
// HYPOTHESIZE → VERIFY → SYNTHESIZE. Each stage has confidence ranges,
// quality dimension boosts, and transition rules.

use serde::{Deserialize, Serialize};

/// The 6 cognitive stages based on Bloom's Revised Taxonomy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum CognitiveStage {
    Define,
    Research,
    Analyze,
    Hypothesize,
    Verify,
    Synthesize,
}

impl CognitiveStage {
    /// All stages in progression order.
    pub const ALL: [CognitiveStage; 6] = [
        CognitiveStage::Define,
        CognitiveStage::Research,
        CognitiveStage::Analyze,
        CognitiveStage::Hypothesize,
        CognitiveStage::Verify,
        CognitiveStage::Synthesize,
    ];

    /// Index of this stage in the progression (0-5).
    pub fn index(self) -> usize {
        match self {
            CognitiveStage::Define => 0,
            CognitiveStage::Research => 1,
            CognitiveStage::Analyze => 2,
            CognitiveStage::Hypothesize => 3,
            CognitiveStage::Verify => 4,
            CognitiveStage::Synthesize => 5,
        }
    }

    /// Expected confidence range [min, max] for this stage.
    /// Calibrated per Shewhart (1931) control chart principles.
    pub fn confidence_range(self) -> (f64, f64) {
        match self {
            CognitiveStage::Define => (0.30, 0.60),
            CognitiveStage::Research => (0.30, 0.70),
            CognitiveStage::Analyze => (0.40, 0.80),
            CognitiveStage::Hypothesize => (0.50, 0.85),
            CognitiveStage::Verify => (0.60, 0.90),
            CognitiveStage::Synthesize => (0.70, 0.95),
        }
    }

    /// Quality dimension boost weights for this stage.
    /// Returns [clarity, depth, breadth, logic, relevance, actionability].
    ///
    /// V9: Reduced from 3.0x to 2.0x per Deep Research (2025).
    /// 3.0x caused gradient instability in multi-objective reward
    /// (one dimension dominates, others become noise).
    /// 2.0x provides clear stage prevalence without domination.
    pub fn quality_boosts(self) -> [f64; 6] {
        match self {
            // DEFINE boosts Clarity (2x)
            CognitiveStage::Define => [2.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            // RESEARCH boosts Breadth (2x)
            CognitiveStage::Research => [1.0, 1.0, 2.0, 1.0, 1.0, 1.0],
            // ANALYZE boosts Depth (2x)
            CognitiveStage::Analyze => [1.0, 2.0, 1.0, 1.0, 1.0, 1.0],
            // HYPOTHESIZE boosts Logic (2x)
            CognitiveStage::Hypothesize => [1.0, 1.0, 1.0, 2.0, 1.0, 1.0],
            // VERIFY boosts Relevance (2x)
            CognitiveStage::Verify => [1.0, 1.0, 1.0, 1.0, 2.0, 1.0],
            // SYNTHESIZE boosts Actionability (2x)
            CognitiveStage::Synthesize => [1.0, 1.0, 1.0, 1.0, 1.0, 2.0],
        }
    }

    /// Emoji representation for compact output.
    pub fn emoji(self) -> &'static str {
        match self {
            CognitiveStage::Define => "📋",
            CognitiveStage::Research => "🔍",
            CognitiveStage::Analyze => "🔬",
            CognitiveStage::Hypothesize => "💡",
            CognitiveStage::Verify => "✅",
            CognitiveStage::Synthesize => "🎯",
        }
    }

    /// Whether this stage requires CoVe (Chain-of-Verification) checkpoint.
    /// V13: 3 transitions require verification.
    pub fn requires_cove_checkpoint(self, previous: CognitiveStage) -> bool {
        matches!(
            (previous, self),
            (CognitiveStage::Research, CognitiveStage::Analyze)
                | (CognitiveStage::Analyze, CognitiveStage::Hypothesize)
                | (CognitiveStage::Hypothesize, CognitiveStage::Verify)
        )
    }
}

/// Stage detection from explicit metadata string.
pub fn detect_stage_from_metadata(stage_str: &str) -> Option<CognitiveStage> {
    match stage_str.to_uppercase().as_str() {
        "DEFINE" => Some(CognitiveStage::Define),
        "RESEARCH" => Some(CognitiveStage::Research),
        "ANALYZE" => Some(CognitiveStage::Analyze),
        "HYPOTHESIZE" => Some(CognitiveStage::Hypothesize),
        "VERIFY" => Some(CognitiveStage::Verify),
        "SYNTHESIZE" => Some(CognitiveStage::Synthesize),
        _ => None,
    }
}

/// Auto-detection of cognitive stage from thought content.
/// Uses keyword heuristics when the user doesn't specify a stage.
/// Prefer `detect_stage_from_metadata()` when explicit stage is available.
pub fn detect_stage(thought: &str) -> CognitiveStage {
    let lower = thought.to_lowercase();

    // Check keywords in reverse order (later stages are more specific)
    let synthesize_kw = [
        "therefore",
        "in conclusion",
        "recommend",
        "summary",
        "final",
        "decision",
        "solution",
        "result",
        "conclude",
        "synthesize",
        "por lo tanto",
        "en conclusión",
        "recomendación",
        "resumen",
    ];
    let verify_kw = [
        "verify",
        "test",
        "validate",
        "confirm",
        "check",
        "proof",
        "evidence",
        "assert",
        "ensure",
        "verificar",
        "comprobar",
    ];
    let hypothesize_kw = [
        "hypothesis",
        "propose",
        "predict",
        "suggest",
        "could",
        "might",
        "perhaps",
        "approach",
        "strategy",
        "hipótesis",
        "proponer",
        "estrategia",
    ];
    let analyze_kw = [
        "analyze",
        "compare",
        "trade-off",
        "evaluate",
        "examine",
        "consider",
        "weigh",
        "assess",
        "analizar",
        "comparar",
        "evaluar",
        "pros",
        "cons",
    ];
    let research_kw = [
        "explore",
        "investigate",
        "search",
        "find",
        "look into",
        "gather",
        "options",
        "alternatives",
        "investigar",
        "explorar",
        "opciones",
        "buscar",
    ];

    if synthesize_kw.iter().any(|kw| lower.contains(kw)) {
        return CognitiveStage::Synthesize;
    }
    if verify_kw.iter().any(|kw| lower.contains(kw)) {
        return CognitiveStage::Verify;
    }
    if hypothesize_kw.iter().any(|kw| lower.contains(kw)) {
        return CognitiveStage::Hypothesize;
    }
    if analyze_kw.iter().any(|kw| lower.contains(kw)) {
        return CognitiveStage::Analyze;
    }
    if research_kw.iter().any(|kw| lower.contains(kw)) {
        return CognitiveStage::Research;
    }

    CognitiveStage::Define
}

/// Session state tracking across multiple thought invocations.
#[derive(Debug, Clone, Serialize)]
pub struct StageSession {
    /// Current cognitive stage.
    pub current_stage: CognitiveStage,
    /// Total thoughts processed.
    pub thought_count: usize,
    /// History of stages visited (with thought number).
    pub stage_history: Vec<(usize, CognitiveStage)>,
    /// Accumulated assumptions (deduplicated).
    pub assumptions: Vec<String>,
    /// Whether the session has completed (reached SYNTHESIZE with nextThoughtNeeded=false).
    pub completed: bool,
}

impl Default for StageSession {
    fn default() -> Self {
        Self {
            current_stage: CognitiveStage::Define,
            thought_count: 0,
            stage_history: Vec::new(),
            assumptions: Vec::new(),
            completed: false,
        }
    }
}

impl StageSession {
    pub fn new() -> Self {
        Self::default()
    }

    /// Advance to the next thought, optionally with an explicit stage.
    /// Returns warnings if the stage transition is unusual.
    pub fn advance(
        &mut self,
        explicit_stage: Option<CognitiveStage>,
        thought_content: &str,
        new_assumptions: &[String],
    ) -> Vec<String> {
        self.thought_count += 1;
        let mut warnings = Vec::new();

        // Determine stage: explicit > auto-detected
        let stage = explicit_stage.unwrap_or_else(|| detect_stage(thought_content));

        // Check for stage regression (going backward)
        if stage.index() < self.current_stage.index() && self.thought_count > 2 {
            warnings.push(format!(
                "⚠️ Stage regression: {} → {} (thought #{}). Consider if this is intentional revision.",
                self.current_stage.emoji(),
                stage.emoji(),
                self.thought_count
            ));
        }

        // Check for CoVe checkpoint requirement
        if stage.requires_cove_checkpoint(self.current_stage) {
            let open_assumptions = self.assumptions.len();
            if open_assumptions > 0 {
                warnings.push(format!(
                    "🔍 CoVe Checkpoint: {} → {} transition with {} open assumption(s). Verify before proceeding.",
                    self.current_stage.emoji(),
                    stage.emoji(),
                    open_assumptions
                ));
            }
        }

        // Deduplicate and add new assumptions
        for assumption in new_assumptions {
            let lower = assumption.to_lowercase();
            if !self.assumptions.iter().any(|a| a.to_lowercase() == lower) {
                self.assumptions.push(assumption.clone());
            }
        }

        self.current_stage = stage;
        self.stage_history.push((self.thought_count, stage));

        warnings
    }

    /// Check if confidence is within expected range for current stage.
    /// Returns calibration warning if outside range.
    pub fn check_confidence(&self, confidence: f64) -> Option<String> {
        let (min, max) = self.current_stage.confidence_range();
        if confidence < min {
            Some(format!(
                "📊 Confidence {:.0}% is below expected range for {} stage [{:.0}%-{:.0}%]. Consider gathering more evidence.",
                confidence * 100.0,
                self.current_stage.emoji(),
                min * 100.0,
                max * 100.0,
            ))
        } else if confidence > max {
            Some(format!(
                "📊 Confidence {:.0}% exceeds expected range for {} stage [{:.0}%-{:.0}%]. Possible overconfidence — verify assumptions.",
                confidence * 100.0,
                self.current_stage.emoji(),
                min * 100.0,
                max * 100.0,
            ))
        } else {
            None
        }
    }

    /// Mark the session as completed.
    pub fn complete(&mut self) {
        self.completed = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stage_progression_order() {
        for (i, stage) in CognitiveStage::ALL.iter().enumerate() {
            assert_eq!(stage.index(), i);
        }
    }

    #[test]
    fn test_detect_stage_keywords() {
        assert_eq!(
            detect_stage("Let me define the problem scope"),
            CognitiveStage::Define
        );
        assert_eq!(
            detect_stage("I need to explore options"),
            CognitiveStage::Research
        );
        assert_eq!(
            detect_stage("Let me compare the trade-offs"),
            CognitiveStage::Analyze
        );
        assert_eq!(
            detect_stage("I propose the following approach"),
            CognitiveStage::Hypothesize
        );
        assert_eq!(
            detect_stage("Let me verify this assumption"),
            CognitiveStage::Verify
        );
        assert_eq!(
            detect_stage("In conclusion, the best solution is"),
            CognitiveStage::Synthesize
        );
    }

    #[test]
    fn test_confidence_calibration() {
        let session = StageSession {
            current_stage: CognitiveStage::Define,
            ..Default::default()
        };
        // 0.45 is within [0.30, 0.60] — no warning
        assert!(session.check_confidence(0.45).is_none());
        // 0.10 is below range — warning
        assert!(session.check_confidence(0.10).is_some());
        // 0.90 is above range — warning
        assert!(session.check_confidence(0.90).is_some());
    }

    #[test]
    fn test_cove_checkpoint_transitions() {
        assert!(CognitiveStage::Analyze.requires_cove_checkpoint(CognitiveStage::Research));
        assert!(CognitiveStage::Hypothesize.requires_cove_checkpoint(CognitiveStage::Analyze));
        assert!(CognitiveStage::Verify.requires_cove_checkpoint(CognitiveStage::Hypothesize));
        // Non-CoVe transitions
        assert!(!CognitiveStage::Research.requires_cove_checkpoint(CognitiveStage::Define));
        assert!(!CognitiveStage::Synthesize.requires_cove_checkpoint(CognitiveStage::Verify));
    }

    #[test]
    fn test_session_advance_with_regression_warning() {
        let mut session = StageSession::new();
        // Advance to ANALYZE (thought 1, 2, then 3 with regression)
        session.advance(Some(CognitiveStage::Research), "exploring", &[]);
        session.advance(Some(CognitiveStage::Analyze), "analyzing", &[]);
        let warnings = session.advance(Some(CognitiveStage::Define), "redefine", &[]);
        assert!(!warnings.is_empty(), "Expected regression warning");
        assert!(warnings[0].contains("regression"));
    }

    #[test]
    fn test_assumption_deduplication() {
        let mut session = StageSession::new();
        session.advance(
            None,
            "define problem",
            &["DB is PostgreSQL".to_string(), "API is REST".to_string()],
        );
        session.advance(
            None,
            "research options",
            &["db is postgresql".to_string(), "Cache is Redis".to_string()],
        );
        // "DB is PostgreSQL" should be deduplicated (case-insensitive)
        assert_eq!(session.assumptions.len(), 3);
    }
}
