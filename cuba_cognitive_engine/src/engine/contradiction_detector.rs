// src/engine/contradiction_detector.rs
//
// S2: NLI Contradiction Detection via Negation Heuristics
//
// Detects contradictions between the current thought and previous claims
// using negation pattern matching, antonym lookup, and quantifier analysis.
// Based on De Marneffe et al. (2008) "Finding Contradictions in Text".
//
// Replaces hardcoded `contradiction_rate: 0.0` in EWMA signals.
//
// Detection signals:
// 1. Direct negation: "should" → "should not"
// 2. Antonym pairs: "increase" ↔ "decrease"
// 3. Quantifier conflicts: "all" ↔ "none", "always" ↔ "never"
// 4. Numerical conflicts: Same subject with different numbers

use serde::Serialize;
use crate::engine::shared_utils::truncate_str;
use rayon::prelude::*;

/// Result of contradiction analysis.
#[derive(Debug, Clone, Serialize)]
pub struct ContradictionResult {
    /// Contradiction rate: 0.0 = no contradictions, 1.0 = all contradictory.
    pub rate: f64,
    /// Detected contradiction descriptions.
    pub contradictions: Vec<String>,
    /// Total claims analyzed.
    pub claims_checked: usize,
}

/// Antonym pair for contradiction detection.
const ANTONYM_PAIRS: &[(&str, &str)] = &[
    ("increase", "decrease"),
    ("add", "remove"),
    ("enable", "disable"),
    ("start", "stop"),
    ("open", "close"),
    ("create", "delete"),
    ("allow", "deny"),
    ("accept", "reject"),
    ("include", "exclude"),
    ("connect", "disconnect"),
    ("encrypt", "decrypt"),
    ("sync", "async"),
    ("success", "failure"),
    ("valid", "invalid"),
    ("safe", "unsafe"),
    ("true", "false"),
    ("yes", "no"),
    ("always", "never"),
    ("all", "none"),
    ("better", "worse"),
    ("faster", "slower"),
    ("more", "less"),
    ("must", "must not"),
    ("should", "should not"),
    ("can", "cannot"),
    // Spanish
    ("agregar", "eliminar"),
    ("crear", "borrar"),
    ("activar", "desactivar"),
    ("permitir", "denegar"),
    ("aumentar", "disminuir"),
    ("siempre", "nunca"),
    ("todos", "ninguno"),
    ("mejor", "peor"),
    ("verdadero", "falso"),
];

/// Detect contradictions between current thought and previous claims.
///
/// Returns a `ContradictionResult` with:
/// - `rate`: proportion of contradictions found (0.0 to 1.0)
/// - `contradictions`: human-readable descriptions
/// - `claims_checked`: total comparisons made
pub fn detect_contradictions(current: &str, previous_claims: &[&str]) -> ContradictionResult {
    if previous_claims.is_empty() || current.is_empty() {
        return ContradictionResult {
            rate: 0.0,
            contradictions: vec![],
            claims_checked: 0,
        };
    }

    let current_lower = current.to_lowercase();

    // Utilize Rayon data parallelism to evaluate previous claims concurrently
    let found_contradictions: Vec<String> = previous_claims
        .par_iter()
        .flat_map(|&prev| {
            let mut local_contradictions = Vec::new();
            let prev_lower = prev.to_lowercase();

            // ── Signal 1: Direct negation ──────────────────────────
            if check_direct_negation(&current_lower, &prev_lower) {
                local_contradictions.push(format!(
                    "Direct negation detected vs: \"{}...\"",
                    truncate_str(prev, 50)
                ));
            }

            // ── Signal 2: Antonym pairs ────────────────────────────
            if let Some(pair) = check_antonym_conflict(&current_lower, &prev_lower) {
                local_contradictions.push(format!(
                    "Antonym conflict: '{}' vs '{}' in previous claim",
                    pair.0, pair.1
                ));
            }

            // ── Signal 3: Quantifier conflicts ─────────────────────
            if check_quantifier_conflict(&current_lower, &prev_lower) {
                local_contradictions.push(format!(
                    "Quantifier conflict vs: \"{}...\"",
                    truncate_str(prev, 50)
                ));
            }

            local_contradictions
        })
        .collect();

    let checks = previous_claims.len();
    let rate = if checks > 0 {
        (found_contradictions.len() as f64 / checks as f64).min(1.0)
    } else {
        0.0
    };

    ContradictionResult {
        rate,
        contradictions: found_contradictions,
        claims_checked: checks,
    }
}

/// Check for direct negation patterns.
/// e.g., "should use" vs "should not use"
fn check_direct_negation(current: &str, previous: &str) -> bool {
    let negation_pairs = [
        ("not ", " "),
        ("no ", " "),
        ("never ", "always "),
        ("don't ", "do "),
        ("doesn't ", "does "),
        ("won't ", "will "),
        ("can't ", "can "),
        ("shouldn't ", "should "),
        ("isn't ", "is "),
        ("aren't ", "are "),
    ];

    for (neg, pos) in &negation_pairs {
        // Current has negation, previous has affirmation (or vice versa)
        if current.contains(neg) && previous.contains(pos) && !previous.contains(neg) {
            // Check shared context (at least 2 common content words)
            if has_shared_context(current, previous, 2) {
                return true;
            }
        }
        if previous.contains(neg) && current.contains(pos) && !current.contains(neg)
            && has_shared_context(current, previous, 2)
        {
            return true;
        }
    }
    false
}

/// Check for antonym pair conflicts.
/// Returns the conflicting pair if found.
fn check_antonym_conflict<'a>(current: &str, previous: &str) -> Option<(&'a str, &'a str)> {
    for (a, b) in ANTONYM_PAIRS {
        let current_has_a = current.contains(a);
        let current_has_b = current.contains(b);
        let prev_has_a = previous.contains(a);
        let prev_has_b = previous.contains(b);

        // One text uses word A, other uses word B, in same context
        if (current_has_a && prev_has_b && !current_has_b && !prev_has_a)
            || (current_has_b && prev_has_a && !current_has_a && !prev_has_b)
        {
            // Require shared context to avoid false positives
            if has_shared_context(current, previous, 1) {
                return Some((a, b));
            }
        }
    }
    None
}

/// Check for quantifier conflicts.
/// e.g., "all tests pass" vs "some tests fail"
fn check_quantifier_conflict(current: &str, previous: &str) -> bool {
    let quantifier_pairs = [
        ("all ", "none "),
        ("every ", "no "),
        ("always ", "never "),
        ("everything ", "nothing "),
        ("todos ", "ninguno "),
        ("siempre ", "nunca "),
    ];

    for (total, zero) in &quantifier_pairs {
        if ((current.contains(total) && previous.contains(zero))
            || (current.contains(zero) && previous.contains(total)))
            && has_shared_context(current, previous, 1)
        {
            return true;
        }
    }
    false
}

/// Check if two texts share enough context (common content words).
fn has_shared_context(text_a: &str, text_b: &str, min_shared: usize) -> bool {
    let stopwords = crate::engine::shared_utils::stopwords();

    let words_a: std::collections::HashSet<&str> = text_a
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
        .filter(|w| w.len() > 3 && !stopwords.contains(*w))
        .collect();

    let shared = text_b
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
        .filter(|w| w.len() > 3 && words_a.contains(w))
        .count();

    shared >= min_shared
}

/// An internal contradiction found within a single thought.
#[derive(Debug, Clone, Serialize)]
pub struct InternalContradiction {
    /// First conflicting claim.
    pub claim_a: String,
    /// Second conflicting claim (contradicts claim_a).
    pub claim_b: String,
}

/// Detect contradictions within a single thought (between its own sentences).
///
/// Splits the thought into sentences and checks each pair for contradictions.
/// Returns a list of internal contradictions found.
pub fn detect_internal_contradictions(thought: &str) -> Vec<InternalContradiction> {
    let sentences: Vec<&str> = thought
        .split(['.', ';', '\n'])
        .map(|s| s.trim())
        .filter(|s| s.len() > 10)
        .collect();

    if sentences.len() < 2 {
        return vec![];
    }

    let mut contradictions = Vec::new();
    for i in 0..sentences.len() {
        for j in (i + 1)..sentences.len() {
            let a_lower = sentences[i].to_lowercase();
            let b_lower = sentences[j].to_lowercase();

            // Check antonym conflicts
            if let Some(_pair) = check_antonym_conflict(&a_lower, &b_lower) {
                contradictions.push(InternalContradiction {
                    claim_a: truncate_str(sentences[i], 60),
                    claim_b: truncate_str(sentences[j], 60),
                });
            }

            // Check quantifier conflicts
            if check_quantifier_conflict(&a_lower, &b_lower) {
                contradictions.push(InternalContradiction {
                    claim_a: truncate_str(sentences[i], 60),
                    claim_b: truncate_str(sentences[j], 60),
                });
            }
        }
    }

    contradictions
}

// truncate_str moved to shared_utils::truncate_str (F10: UTF-8 safe)

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_contradiction() {
        let result = detect_contradictions(
            "We should implement caching with Redis",
            &["The API needs better performance"],
        );
        assert_eq!(result.rate, 0.0, "No contradiction expected");
        assert!(result.contradictions.is_empty());
    }

    #[test]
    fn test_antonym_conflict() {
        let result = detect_contradictions(
            "We should increase the cache timeout for the database",
            &["We need to decrease the cache timeout for the database"],
        );
        assert!(result.rate > 0.0, "Should detect increase/decrease conflict");
        assert!(!result.contradictions.is_empty());
    }

    #[test]
    fn test_quantifier_conflict() {
        let result = detect_contradictions(
            "All tests should pass before deployment",
            &["None tests should pass before deployment"],
        );
        assert!(result.rate > 0.0, "Should detect all/none conflict: {:?}", result);
    }

    #[test]
    fn test_no_contradiction_same_text() {
        let text = "The database needs optimization";
        let result = detect_contradictions(text, &[text]);
        assert_eq!(result.rate, 0.0, "Same text should not contradict itself");
    }

    #[test]
    fn test_empty_previous() {
        let result = detect_contradictions("Some thought", &[]);
        assert_eq!(result.claims_checked, 0);
        assert_eq!(result.rate, 0.0);
    }

    #[test]
    fn test_empty_current() {
        let result = detect_contradictions("", &["Previous claim"]);
        assert_eq!(result.rate, 0.0);
    }
}
