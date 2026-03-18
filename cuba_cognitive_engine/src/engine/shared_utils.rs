// src/engine/shared_utils.rs
//
// Shared utilities for the cognitive engine to avoid duplication.
// F1: Centralized stopword list (EN+ES)
// F10: UTF-8 safe truncation

use std::collections::HashSet;

/// Shared bilingual stopword set (EN + ES).
/// Used by: semantic_similarity, novelty_tracker, quality_metrics, contradiction_detector.
///
/// Union of all previous per-module stopword lists — deduplicated.
pub fn stopwords() -> HashSet<&'static str> {
    [
        // English
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
        "do", "does", "did", "will", "would", "could", "should", "may", "might", "shall", "can",
        "to", "of", "in", "for", "on", "with", "at", "by", "from", "as", "into", "through",
        "during", "before", "after", "above", "below", "between", "under", "and", "but", "or",
        "nor", "not", "so", "yet", "both", "either", "neither", "each", "every", "all", "any",
        "few", "more", "most", "other", "some", "such", "no", "only", "own", "same", "than", "too",
        "very", "just", "it", "its", "this", "that", "these", "those", "i", "me", "my", "we",
        "our", "you", "your", "he", "him", "his", "she", "her", "they", "them", "their", "what",
        "which", "who", "whom", "if", "also", // Spanish
        "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "en", "con", "por",
        "para", "sin", "sobre", "es", "son", "fue", "ser", "estar", "hay", "como", "pero", "que",
        "se", "su", "sus", "este", "esta", "esto", "eso", "sino", "donde", "puede", "tiene",
        "entre",
    ]
    .iter()
    .copied()
    .collect()
}

/// UTF-8 safe string truncation.
///
/// Truncates to a max byte length without panicking on multi-byte boundaries.
/// Appends "..." if truncated.
pub fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        // Find the last char boundary that fits within max_len bytes
        let mut boundary = 0;
        for (i, c) in s.char_indices() {
            let end = i + c.len_utf8();
            if end > max_len {
                break;
            }
            boundary = end;
        }
        format!("{}...", &s[..boundary])
    }
}

/// Detect if input is primarily code (Python/Rust/JS).
///
/// Used by F16/F18: code-awareness for quality metrics and corrective directives.
/// Returns true if >50% of lines look like code.
pub fn is_code_input(text: &str) -> bool {
    let lines: Vec<&str> = text.lines().collect();
    if lines.is_empty() {
        return false;
    }

    let code_indicators = [
        "def ",
        "class ",
        "import ",
        "from ",
        "return ",
        "assert ",
        "fn ",
        "let ",
        "pub ",
        "use ",
        "struct ",
        "impl ",
        "mod ",
        "const ",
        "var ",
        "function ",
        "if ",
        "for ",
        "while ",
        "=",
        "(",
        ")",
        "{",
        "}",
        "[",
        "]",
        ";",
        "->",
        "=>",
    ];

    let code_lines = lines
        .iter()
        .filter(|line| {
            let trimmed = line.trim();
            !trimmed.is_empty()
                && (trimmed.starts_with('#') // comments/decorators
                    || trimmed.starts_with("//")
                    || code_indicators.iter().any(|ind| trimmed.contains(ind)))
        })
        .count();

    let non_empty = lines.iter().filter(|l| !l.trim().is_empty()).count();
    if non_empty == 0 {
        return false;
    }

    code_lines as f64 / non_empty as f64 > 0.5
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stopwords_not_empty() {
        let sw = stopwords();
        assert!(sw.len() > 80, "Should have 80+ stopwords, got {}", sw.len());
    }

    #[test]
    fn test_stopwords_contains_en_es() {
        let sw = stopwords();
        assert!(sw.contains("the"), "Missing English stopword");
        assert!(sw.contains("el"), "Missing Spanish stopword");
        assert!(sw.contains("para"), "Missing Spanish stopword");
    }

    #[test]
    fn test_truncate_ascii() {
        assert_eq!(truncate_str("hello", 10), "hello");
        assert_eq!(truncate_str("hello world", 5), "hello...");
    }

    #[test]
    fn test_truncate_utf8_safe() {
        // Spanish: 'á' is 2 bytes in UTF-8
        let text = "caño de ácero inoxidable";
        // Should NOT panic even if max_len falls in middle of 'ñ' or 'á'
        let result = truncate_str(text, 3);
        assert!(result.ends_with("..."), "Should be truncated: {}", result);
        // Verify it doesn't panic at various boundary positions
        for i in 0..text.len() + 5 {
            let _ = truncate_str(text, i); // Must not panic
        }
    }

    #[test]
    fn test_truncate_empty() {
        assert_eq!(truncate_str("", 10), "");
    }

    #[test]
    fn test_is_code_python() {
        let code = "def verify():\n    assert 2 + 2 == 4\n    return True";
        assert!(is_code_input(code), "Python code should be detected");
    }

    #[test]
    fn test_is_code_rust() {
        let code = "fn main() {\n    let x = 42;\n    println!(\"{}\", x);\n}";
        assert!(is_code_input(code), "Rust code should be detected");
    }

    #[test]
    fn test_is_not_code_natural_language() {
        let nl = "The database migration requires careful planning.\nWe should verify all constraints before proceeding.";
        assert!(
            !is_code_input(nl),
            "Natural language should NOT be detected as code"
        );
    }

    #[test]
    fn test_is_code_mixed() {
        let mixed = "# Verify the implementation\ndef test():\n    assert x > 5\n    assert y == 10\n    return True";
        assert!(
            is_code_input(mixed),
            "Mixed but mostly code should be detected"
        );
    }
}
