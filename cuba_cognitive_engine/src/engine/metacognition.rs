// src/engine/metacognition.rs
//
// R8: Metacognitive Analysis
//
// Simplified: retains filler ratio and content-word ratio as useful metrics.
// Removed: fallacy detection (keyword-based, high false-positive rate) and
// dialectical reasoning check (Claude already generates counter-arguments).
//
// References:
// - Filler ratio: Flavell 1979, "Metacognition and Cognitive Monitoring"
// - Content-word ratio: Graesser 2004, Coh-Metrix

use serde::Serialize;
use std::collections::HashSet;
use std::sync::LazyLock;

/// Metacognitive analysis results.
#[derive(Debug, Clone, Serialize)]
pub struct MetacognitiveReport {
    /// Ratio of filler words to total words (lower is better).
    pub filler_ratio: f64,
    /// Content-Word Ratio: content_words / total_words (Coh-Metrix).
    pub content_word_ratio: f64,
    /// Verifiable claims per sentence.
    pub claim_density: f64,
    /// Actionable warnings (only non-empty when issues detected).
    pub warnings: Vec<String>,
}

/// Perform metacognitive analysis on a thought.
pub fn analyze_metacognition(thought: &str, _is_verify_or_synthesize: bool) -> MetacognitiveReport {
    let filler_ratio = compute_filler_ratio(thought);
    let cwr = compute_content_word_ratio(thought);
    let claim_density = compute_claim_density(thought);

    let mut warnings = Vec::new();

    // Filler warning: > 30% filler ratio (Flavell 1979)
    if filler_ratio > 0.30 {
        warnings.push(format!(
            "Metacognition: {:.0}% filler ratio — reduce hedging and padding",
            filler_ratio * 100.0
        ));
    }

    // Verbosity warning: CWR < 40% (Graesser 2004)
    if cwr < 0.40 && cwr > 0.0 {
        warnings.push(format!(
            "Verbosity: Content-word ratio {:.0}% — be more concise",
            cwr * 100.0
        ));
    }

    MetacognitiveReport {
        filler_ratio,
        content_word_ratio: cwr,
        claim_density,
        warnings,
    }
}

/// Compute filler word ratio.
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

/// Compute Content-Word Ratio (Coh-Metrix / Graesser 2004).
fn compute_content_word_ratio(text: &str) -> f64 {
    static FUNCTION_WORDS: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
        [
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "have", "has",
            "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "can",
            "shall", "must", "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
            "into", "through", "during", "before", "after", "above", "below", "between", "and",
            "but", "or", "nor", "not", "so", "yet", "both", "either", "neither", "it", "its",
            "this", "that", "these", "those", "he", "she", "we", "they", "them", "their", "my",
            "your", "our", "i", "me", "you",
            // Spanish function words
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

    let claim_markers = [
        "is", "are", "was", "causes", "results in", "equals",
        "greater than", "less than", "requires", "must", "always", "never", "every",
        "%", "=", ">", "<",
        "es", "son", "causa", "resulta en", "requiere", "siempre", "nunca", "cada", "todo",
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

/// G8: Reasoning Type Classification (Walton 2006).
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
        "therefore", "must be", "necessarily", "follows that", "proves",
        "if and only if", "by definition", "logically", "deduction",
    ];
    let inductive_kw = [
        "pattern suggests", "in most cases", "evidence shows", "likely",
        "tends to", "usually", "observed that", "data indicates", "correlation", "frequency",
    ];
    let abductive_kw = [
        "best explanation", "probably because", "hypothesis is",
        "could be explained by", "most likely cause", "plausible", "inference to", "suggests that",
    ];
    let analogical_kw = [
        "similar to", "just as", "comparable", "analogous", "like",
        "resembles", "parallels", "same way",
    ];

    let d: usize = deductive_kw.iter().filter(|k| lower.contains(**k)).count();
    let i: usize = inductive_kw.iter().filter(|k| lower.contains(**k)).count();
    let a: usize = abductive_kw.iter().filter(|k| lower.contains(**k)).count();
    let g: usize = analogical_kw.iter().filter(|k| lower.contains(**k)).count();

    let max = d.max(i).max(a).max(g);
    if max == 0 {
        return ReasoningType::Mixed;
    }

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
    fn test_clean_analysis() {
        let text = "The PostgreSQL query takes 250ms due to sequential scan. \
                     However, adding an index on user_id could reduce this to 5ms \
                     because B-tree lookup is O(log n).";
        let report = analyze_metacognition(text, true);
        assert!(report.filler_ratio < 0.1);
    }
}
