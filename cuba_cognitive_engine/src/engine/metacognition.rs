// src/engine/metacognition.rs
//
// R8: Metacognitive Analysis
//
// Detects metacognitive issues in reasoning:
// - Filler ratio (Flavell, 1979 — Metacognition and Cognitive Monitoring)
// - Claim density (verifiable assertions per sentence)
// - Fallacy detection (hasty generalization, false dichotomy)
// - Dialectical reasoning check (counter-arguments in VERIFY/SYNTHESIZE)
// - Content-word ratio / Verbosity (V3, Graesser 2004 — Coh-Metrix)

use serde::Serialize;
use std::collections::HashSet;
use std::sync::LazyLock;

/// Metacognitive analysis results.
#[derive(Debug, Clone, Serialize)]
pub struct MetacognitiveReport {
    /// Ratio of filler words to total words (lower is better).
    pub filler_ratio: f64,
    /// Content-Word Ratio: content_words / total_words (V3, Coh-Metrix).
    pub content_word_ratio: f64,
    /// Verifiable claims per sentence.
    pub claim_density: f64,
    /// Detected fallacies.
    pub fallacies: Vec<DetectedFallacy>,
    /// Whether dialectical reasoning is present (counter-arguments).
    pub has_dialectical: bool,
    /// Actionable warnings (only non-empty when issues detected).
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DetectedFallacy {
    pub fallacy_type: &'static str,
    pub evidence: String,
}

/// Perform metacognitive analysis on a thought.
pub fn analyze_metacognition(thought: &str, is_verify_or_synthesize: bool) -> MetacognitiveReport {
    let filler_ratio = compute_filler_ratio(thought);
    let cwr = compute_content_word_ratio(thought);
    let claim_density = compute_claim_density(thought);
    let fallacies = detect_fallacies(thought);
    let has_dialectical = check_dialectical(thought);

    let mut warnings = Vec::new();

    // Filler warning: > 30% filler ratio (Flavell 1979)
    if filler_ratio > 0.30 {
        warnings.push(format!(
            "🧠 Metacognition: {:.0}% filler ratio — reduce hedging and padding",
            filler_ratio * 100.0
        ));
    }

    // Verbosity warning: CWR < 40% (V3, Graesser 2004)
    if cwr < 0.40 && cwr > 0.0 {
        warnings.push(format!(
            "📝 Verbosity: Content-word ratio {:.0}% — be more concise",
            cwr * 100.0
        ));
    }

    // Fallacy warnings
    for fallacy in &fallacies {
        warnings.push(format!(
            "⚠️ Fallacy detected: {} — \"{}\"",
            fallacy.fallacy_type, fallacy.evidence
        ));
    }

    // Dialectical check for VERIFY/SYNTHESIZE stages
    if is_verify_or_synthesize && !has_dialectical {
        warnings.push(
            "🔄 Dialectical gap: VERIFY/SYNTHESIZE stage without counter-arguments. Consider opposing viewpoints.".to_string()
        );
    }

    MetacognitiveReport {
        filler_ratio,
        content_word_ratio: cwr,
        claim_density,
        fallacies,
        has_dialectical,
        warnings,
    }
}

/// Compute filler word ratio.
/// Filler words: hedging, padding, social lubricant phrases.
fn compute_filler_ratio(text: &str) -> f64 {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return 0.0;
    }

    let fillers = [
        "basically",
        "actually",
        "literally",
        "really",
        "just",
        "quite",
        "rather",
        "somewhat",
        "pretty",
        "kind",
        "sort",
        "honestly",
        "certainly",
        "definitely",
        "absolutely",
        "essentially",
        "fundamentally",
        "obviously",
        "clearly",
        "simply",
        "merely",
        "well",
        "so",
        "like",
        "you know",
        "i think",
        "i believe",
        "i feel",
        "in my opinion",
        "básicamente",
        "realmente",
        "simplemente",
        "obviamente",
        "claramente",
        "ciertamente",
        "definitivamente",
    ];

    let filler_count = words
        .iter()
        .filter(|w| {
            let lower = w.to_lowercase();
            let trimmed = lower.trim_matches(|c: char| !c.is_alphabetic());
            fillers.contains(&trimmed)
        })
        .count();

    filler_count as f64 / words.len() as f64
}

/// Compute Content-Word Ratio (V3, Coh-Metrix / Graesser 2004).
/// Content words = nouns, verbs, adjectives, adverbs (approximated).
/// Function words = articles, prepositions, conjunctions, pronouns.
fn compute_content_word_ratio(text: &str) -> f64 {
    static FUNCTION_WORDS: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
        [
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "have", "has",
            "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "can",
            "shall", "must", "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
            "into", "through", "during", "before", "after", "above", "below", "between", "and",
            "but", "or", "nor", "not", "so", "yet", "both", "either", "neither", "it", "its",
            "this", "that", "these", "those", "he", "she", "we", "they", "them", "their", "my",
            "your", "our", "i", "me", "you", // Spanish function words
            "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "en", "con", "por",
            "para", "sin", "sobre", "entre", "y", "o", "ni", "que", "se", "lo", "le", "les", "su",
            "sus", "mi", "tu", "nos", "es", "son", "fue", "ser", "estar", "hay", "como", "más",
            "no",
        ]
        .iter()
        .copied()
        .collect()
    });

    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return 0.0;
    }

    let content_count = words
        .iter()
        .filter(|w| {
            let lower = w.to_lowercase();
            let trimmed = lower.trim_matches(|c: char| !c.is_alphabetic());
            !trimmed.is_empty() && !FUNCTION_WORDS.contains(trimmed)
        })
        .count();

    content_count as f64 / words.len() as f64
}

/// Compute claim density: verifiable assertions per sentence.
fn compute_claim_density(text: &str) -> f64 {
    let sentences: Vec<&str> = text
        .split(['.', '!', '?'])
        .filter(|s| s.trim().len() > 10)
        .collect();

    if sentences.is_empty() {
        return 0.0;
    }

    // Claim indicators: numbers, comparisons, specific assertions
    let claim_markers = [
        "is",
        "are",
        "was",
        "causes",
        "results in",
        "equals",
        "greater than",
        "less than",
        "requires",
        "must",
        "always",
        "never",
        "every",
        "%",
        "=",
        ">",
        "<",
        "es",
        "son",
        "causa",
        "resulta en",
        "requiere",
        "siempre",
        "nunca",
        "cada",
        "todo",
    ];

    let claims: usize = sentences
        .iter()
        .filter(|s| {
            let lower = s.to_lowercase();
            claim_markers.iter().any(|m| lower.contains(m))
                && (s.chars().any(|c| c.is_ascii_digit())
                    || lower.contains("must")
                    || lower.contains("always")
                    || lower.contains("never")
                    || lower.contains("siempre")
                    || lower.contains("nunca"))
        })
        .count();

    claims as f64 / sentences.len() as f64
}

/// Detect logical fallacies.
fn detect_fallacies(text: &str) -> Vec<DetectedFallacy> {
    let lower = text.to_lowercase();
    let mut fallacies = Vec::new();

    // Hasty Generalization: "all X are Y", "every X is Y" without evidence
    let hasty_markers = [
        "all ", "every ", "always ", "never ", "todos ", "siempre ", "nunca ",
    ];
    for marker in &hasty_markers {
        if let Some(pos) = lower.find(marker) {
            let snippet = &text[pos..text.len().min(pos + 60)];
            // Only flag if no qualifying evidence nearby
            if !lower[pos..].contains("because")
                && !lower[pos..].contains("based on")
                && !lower[pos..].contains("according to")
                && !lower[pos..].contains("porque")
                && !lower[pos..].contains("según")
            {
                fallacies.push(DetectedFallacy {
                    fallacy_type: "Hasty Generalization",
                    evidence: snippet.to_string(),
                });
                break; // Only report once
            }
        }
    }

    // False Dichotomy: "either X or Y" with only 2 options
    if (lower.contains("either") && lower.contains("or"))
        || (lower.contains("only two") || lower.contains("solo dos"))
    {
        // Check that there are genuinely only 2 options presented
        let or_count = lower.matches(" or ").count() + lower.matches(" o ").count();
        if or_count == 1 {
            fallacies.push(DetectedFallacy {
                fallacy_type: "False Dichotomy",
                evidence: "Presenting only two options — consider if more alternatives exist"
                    .to_string(),
            });
        }
    }

    fallacies
}

/// Check for dialectical reasoning (presence of counter-arguments).
fn check_dialectical(text: &str) -> bool {
    let lower = text.to_lowercase();
    let counter_markers = [
        "however",
        "on the other hand",
        "alternatively",
        "counter",
        "against",
        "drawback",
        "limitation",
        "risk",
        "weakness",
        "trade-off",
        "downside",
        "challenge",
        "caveat",
        "sin embargo",
        "por otro lado",
        "alternativamente",
        "en contra",
        "limitación",
        "riesgo",
        "desventaja",
    ];
    counter_markers.iter().any(|m| lower.contains(m))
}

/// G8: Reasoning Type Classification (Walton 2006).
///
/// Classifies a thought as deductive, inductive, abductive, or analogical.
/// Informational-only — helps the agent understand what type of reasoning
/// it's employing and choose appropriate next strategies.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub enum ReasoningType {
    Deductive,
    Inductive,
    Abductive,
    Analogical,
    Mixed,
}

impl ReasoningType {
    pub fn label(self) -> &'static str {
        match self {
            ReasoningType::Deductive => "Deductive",
            ReasoningType::Inductive => "Inductive",
            ReasoningType::Abductive => "Abductive",
            ReasoningType::Analogical => "Analogical",
            ReasoningType::Mixed => "Mixed",
        }
    }
}

pub fn classify_reasoning_type(thought: &str) -> ReasoningType {
    let lower = thought.to_lowercase();

    let deductive_kw = [
        "therefore",
        "must be",
        "necessarily",
        "follows that",
        "proves",
        "if and only if",
        "by definition",
        "logically",
        "deduction",
    ];
    let inductive_kw = [
        "pattern suggests",
        "in most cases",
        "evidence shows",
        "likely",
        "tends to",
        "usually",
        "observed that",
        "data indicates",
        "correlation",
        "frequency",
    ];
    let abductive_kw = [
        "best explanation",
        "probably because",
        "hypothesis is",
        "could be explained by",
        "most likely cause",
        "plausible",
        "inference to",
        "suggests that",
    ];
    let analogical_kw = [
        "similar to",
        "just as",
        "comparable",
        "analogous",
        "like",
        "resembles",
        "parallels",
        "same way",
    ];

    let d: usize = deductive_kw.iter().filter(|k| lower.contains(**k)).count();
    let i: usize = inductive_kw.iter().filter(|k| lower.contains(**k)).count();
    let a: usize = abductive_kw.iter().filter(|k| lower.contains(**k)).count();
    let g: usize = analogical_kw.iter().filter(|k| lower.contains(**k)).count();

    let max = d.max(i).max(a).max(g);
    if max == 0 {
        return ReasoningType::Mixed;
    }

    // Check for mixed: top two types are close
    let scores = [d, i, a, g];
    let mut sorted = scores;
    sorted.sort_unstable_by(|a, b| b.cmp(a));
    if sorted[0] > 0 && sorted[1] > 0 && (sorted[0] - sorted[1]) <= 1 {
        return ReasoningType::Mixed;
    }

    if d == max {
        ReasoningType::Deductive
    } else if i == max {
        ReasoningType::Inductive
    } else if a == max {
        ReasoningType::Abductive
    } else {
        ReasoningType::Analogical
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_high_filler_ratio() {
        let text = "Well, basically I think that obviously clearly \
                     the answer is definitely just simply really quite good";
        let report = analyze_metacognition(text, false);
        assert!(
            report.filler_ratio > 0.3,
            "Expected high filler ratio, got {:.2}",
            report.filler_ratio
        );
        assert!(!report.warnings.is_empty());
    }

    #[test]
    fn test_low_content_word_ratio() {
        let text = "It is the one that was in the of and for the by with";
        let report = analyze_metacognition(text, false);
        assert!(
            report.content_word_ratio < 0.4,
            "Expected low CWR, got {:.2}",
            report.content_word_ratio
        );
    }

    #[test]
    fn test_dialectical_check_in_verify() {
        let text = "The approach works perfectly and validates all assumptions.";
        let report = analyze_metacognition(text, true);
        assert!(!report.has_dialectical);
        assert!(report.warnings.iter().any(|w| w.contains("Dialectical")));
    }

    #[test]
    fn test_hasty_generalization() {
        let text = "All databases are slow. Every framework has bugs.";
        let report = analyze_metacognition(text, false);
        assert!(!report.fallacies.is_empty());
        assert_eq!(report.fallacies[0].fallacy_type, "Hasty Generalization");
    }

    #[test]
    fn test_clean_analysis() {
        let text = "The PostgreSQL query takes 250ms due to sequential scan. \
                     However, adding an index on user_id could reduce this to 5ms \
                     because B-tree lookup is O(log n).";
        let report = analyze_metacognition(text, true);
        assert!(report.has_dialectical);
        assert!(report.filler_ratio < 0.1);
    }
}
