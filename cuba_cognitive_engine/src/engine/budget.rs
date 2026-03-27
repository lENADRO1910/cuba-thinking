// src/engine/budget.rs
//
// R5: Budget Modes — 4 levels of reasoning depth
//
// Controls exploration vs exploitation trade-offs:
// - fast: Cut losses early, exploit known paths (50% threshold)
// - balanced: Default mode (40% threshold)
// - thorough: More exploration room (35% threshold)
// - exhaustive: Maximum exploration (30% threshold)
//
// Based on UCB1 (Kocsis & Szepesvári, 2006) and
// EWMA risk management (Zangari, 1994).

use serde::{Deserialize, Serialize};

/// Budget mode controls depth vs speed trade-offs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BudgetMode {
    Fast,
    #[default]
    Balanced,
    Thorough,
    Exhaustive,
}

impl BudgetMode {
    /// Parse from string, defaulting to Balanced.
    pub fn from_str_opt(s: Option<&str>) -> Self {
        match s {
            Some("fast") => BudgetMode::Fast,
            Some("balanced") => BudgetMode::Balanced,
            Some("thorough") => BudgetMode::Thorough,
            Some("exhaustive") => BudgetMode::Exhaustive,
            _ => BudgetMode::Balanced,
        }
    }

    /// EWMA α floor — prevents sluggish response in long chains.
    /// V4: Budget-aware floor (Roberts 1959, Zangari 1994).
    pub fn ewma_alpha_floor(self) -> f64 {
        match self {
            BudgetMode::Fast => 0.30,
            BudgetMode::Balanced => 0.25,
            BudgetMode::Thorough => 0.20,
            BudgetMode::Exhaustive => 0.15,
        }
    }

    /// MCTS backtracking threshold.
    /// Below this EWMA percentage, the thought is rejected.
    /// V12: Budget-aware thresholds (UCB1, Kocsis 2006).
    pub fn mcts_threshold(self) -> f64 {
        match self {
            BudgetMode::Fast => 0.50,       // 50% — cut losses early
            BudgetMode::Balanced => 0.40,   // 40% — default
            BudgetMode::Thorough => 0.35,   // 35% — give chains room
            BudgetMode::Exhaustive => 0.30, // 30% — maximum exploration
        }
    }

    /// G3: Word-count threshold for length-proportional quality penalty.
    /// Thoughts exceeding this length get a quality penalty unless they
    /// add proportional information density (DeepSeek R1, 2025).
    pub fn length_penalty_threshold(self) -> usize {
        match self {
            BudgetMode::Fast => 80,
            BudgetMode::Balanced => 150,
            BudgetMode::Thorough => 250,
            BudgetMode::Exhaustive => 400,
        }
    }

    /// Display label.
    pub fn label(self) -> &'static str {
        match self {
            BudgetMode::Fast => "⚡ Fast",
            BudgetMode::Balanced => "⚖️ Balanced",
            BudgetMode::Thorough => "🔎 Thorough",
            BudgetMode::Exhaustive => "🔬 Exhaustive",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_is_balanced() {
        assert_eq!(BudgetMode::default(), BudgetMode::Balanced);
    }

    #[test]
    fn test_parse_modes() {
        assert_eq!(BudgetMode::from_str_opt(Some("fast")), BudgetMode::Fast);
        assert_eq!(
            BudgetMode::from_str_opt(Some("thorough")),
            BudgetMode::Thorough
        );
        assert_eq!(
            BudgetMode::from_str_opt(Some("invalid")),
            BudgetMode::Balanced
        );
        assert_eq!(BudgetMode::from_str_opt(None), BudgetMode::Balanced);
    }

    #[test]
    fn test_thresholds_decrease_with_depth() {
        assert!(BudgetMode::Fast.mcts_threshold() > BudgetMode::Balanced.mcts_threshold());
        assert!(BudgetMode::Balanced.mcts_threshold() > BudgetMode::Thorough.mcts_threshold());
        assert!(BudgetMode::Thorough.mcts_threshold() > BudgetMode::Exhaustive.mcts_threshold());
    }

    #[test]
    fn test_alpha_floors_decrease_with_depth() {
        assert!(BudgetMode::Fast.ewma_alpha_floor() > BudgetMode::Exhaustive.ewma_alpha_floor());
    }

}
