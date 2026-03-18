// src/engine/quality_metrics.rs
//
// R2: 6D Quality Metrics System
//
// Measures thought quality across 6 empirically validated dimensions:
// 1. Clarity   — Type-Token Ratio (Templin, 1957)
// 2. Depth     — Subordinate clause density (Hunt, 1965)
// 3. Breadth   — Unique noun ratio (lexical diversity)
// 4. Logic     — Connective diversity + conditional chains (ROSCOE, Golovneva 2023)
// 5. Relevance — Keyword similarity (Salton, 1975)
// 6. Actionability — Imperative verbs + specificity (GRACE, Guan 2024)

use crate::engine::stage_engine::CognitiveStage;
use serde::Serialize;
use std::collections::HashSet;

/// Quality scores across 6 dimensions (0.0 to 1.0 each).
#[derive(Debug, Clone, Serialize)]
pub struct QualityScores {
    pub clarity: f64,
    pub depth: f64,
    pub breadth: f64,
    pub logic: f64,
    pub relevance: f64,
    pub actionability: f64,
}

impl QualityScores {
    /// Weighted mean using stage-specific boosts.
    /// Each stage emphasizes one dimension at 3x weight.
    pub fn weighted_mean(&self, stage: CognitiveStage) -> f64 {
        let boosts = stage.quality_boosts();
        let scores = [
            self.clarity,
            self.depth,
            self.breadth,
            self.logic,
            self.relevance,
            self.actionability,
        ];
        let total_weight: f64 = boosts.iter().sum();
        let weighted_sum: f64 = scores.iter().zip(boosts.iter()).map(|(s, w)| s * w).sum();
        weighted_sum / total_weight
    }

    /// Raw unweighted mean across all 6 dimensions.
    pub fn raw_mean(&self) -> f64 {
        (self.clarity
            + self.depth
            + self.breadth
            + self.logic
            + self.relevance
            + self.actionability)
            / 6.0
    }
}

/// Compute all 6 quality dimensions for a thought.
/// When `is_code` is true, scoring adapts for programming constructs
/// instead of natural language markers (F16).
pub fn compute_quality(thought: &str, context_keywords: &[&str]) -> QualityScores {
    let is_code = crate::engine::shared_utils::is_code_input(thought);
    QualityScores {
        clarity: compute_clarity(thought),
        depth: compute_depth(thought, is_code),
        breadth: compute_breadth(thought),
        logic: compute_logic(thought),
        relevance: compute_relevance(thought, context_keywords),
        actionability: compute_actionability(thought, is_code),
    }
}

/// Vector 2: Word-level Shannon Entropy — O(N), pure Rust.
///
/// Measures information density at the WORD level. Repetitive padding
/// collapses to near-zero; genuine analysis stays above 0.85.
///
/// **Critical correction to external audit**: Char-level entropy does NOT
/// detect text padding (verified in PRM sandbox: "word word word..." → 0.999
/// at char level). Word-level entropy is the correct granularity.
///
/// Shannon, C. (1948). "A Mathematical Theory of Communication."
fn compute_word_entropy(text: &str) -> f64 {
    let words: Vec<&str> = text.split_whitespace().collect();
    // P2-1: Degenerate inputs — ≤2 words can't produce meaningful entropy.
    // Single unique word with max(len, 2) creates artificial max_entropy=log2(2).
    if words.len() <= 2 {
        return 0.0;
    }

    let mut counts = std::collections::HashMap::new();
    let total = words.len() as f64;
    for word in &words {
        let lower = word.to_lowercase();
        *counts.entry(lower).or_insert(0_f64) += 1.0;
    }

    let entropy: f64 = counts
        .values()
        .map(|&c| {
            let p = c / total;
            -p * p.log2()
        })
        .sum();

    let unique = counts.len().max(2) as f64;
    let max_entropy = unique.log2().max(1e-5);
    (entropy / max_entropy).clamp(0.0, 1.0)
}

/// L2: Lempel-Ziv (LZ76) Complexity — O(N), pure Rust.
///
/// Measures the rate of NEW pattern creation in word sequences.
/// Shannon entropy assumes i.i.d. words (independent). Markov chain
/// padding like "system fails because DB fails because system fails..."
/// has HIGH Shannon entropy (~0.98) but ZERO information gain.
///
/// LZ76 detects this: repeating word-blocks match existing substrings,
/// so complexity stays low despite diverse individual words.
///
/// Ref: Lempel & Ziv (1976); "Zipf's and Heaps' Laws for LLM-generated
/// Texts" (2024) — compression-based redundancy for LLM outputs.
fn compute_lz76_complexity(text: &str) -> f64 {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return 0.0;
    }

    let mut complexity = 1.0_f64;
    let mut prefix_end = 1_usize;
    let mut len = 1_usize;

    // C3: Sliding window — prevents O(N²) DoS on large texts (4000+ words).
    // LLMs don't maintain perfect padding at long distance, so 256 words
    // of lookback captures all realistic Markov patterns.
    // Complexity: O(N·K) where K=256 → effectively O(N).
    const MAX_LOOKBACK: usize = 256;

    while prefix_end + len <= words.len() {
        let substring = &words[prefix_end..prefix_end + len];
        let search_start = prefix_end.saturating_sub(MAX_LOOKBACK);
        let history = &words[search_start..prefix_end + len - 1];

        // If the pattern already exists in history, it's not new information
        if history.windows(len).any(|w| w == substring) {
            len += 1;
        } else {
            complexity += 1.0;
            prefix_end += len;
            len = 1;
        }
    }

    let n = words.len() as f64;
    let max_complexity = if n > 1.0 { n / n.ln() } else { 1.0 };
    (complexity / max_complexity).clamp(0.0, 1.0)
}

/// G3: Logistic length-proportional quality penalty (NEW-2 + V2 + L2).
///
/// Uses logistic sigmoid for smooth C¹-continuous transition:
/// penalty(x) = MAX_PENALTY / (1 + e^(-K * (x - threshold)))
///
/// Dual detection (V2 Shannon + L2 LZ76):
/// - Shannon: Detects lexical repetition ("word word word...")
/// - LZ76: Detects structural/Markov repetition ("A because B because A...")
/// - Padding triggers when EITHER metric indicates low information density.
///
/// When padding detected, K increases 4× (0.08 vs 0.02).
pub fn apply_length_penalty(
    quality: QualityScores,
    thought: &str,
    budget: crate::engine::budget::BudgetMode,
) -> QualityScores {
    let word_count = thought.split_whitespace().count();
    let threshold = budget.length_penalty_threshold();

    // Well under threshold — skip calculation entirely
    if word_count <= threshold / 2 {
        return quality;
    }

    const MAX_PENALTY: f64 = 0.30;

    let x = word_count as f64;
    let t = threshold as f64;

    // Dual entropy detection: Shannon (lexical) + LZ76 (structural).
    // Either one detecting low information density triggers harsh penalty.
    let shannon = compute_word_entropy(thought);
    let lz76 = compute_lz76_complexity(thought);
    let is_padding = word_count > threshold && (shannon < 0.60 || lz76 < 0.35);
    let k = if is_padding { 0.08 } else { 0.02 };

    // Sigmoid: 0 → MAX_PENALTY as word_count → ∞
    let sigmoid = 1.0 / (1.0 + (-k * (x - t)).exp());
    let penalty = sigmoid * MAX_PENALTY;
    let multiplier = 1.0 - penalty;

    QualityScores {
        clarity: quality.clarity * multiplier,
        depth: quality.depth * multiplier,
        breadth: quality.breadth * multiplier,
        logic: quality.logic * multiplier,
        relevance: quality.relevance * multiplier,
        actionability: quality.actionability * multiplier,
    }
}

// ─── Dimension 1: Clarity (TTR + Flesch-Kincaid + Sentence Diversity) ─────
// Type-Token Ratio: unique words / total words
// Flesch-Kincaid: readability based on sentence length and syllable count
// Templin (1957), Kincaid et al. (1975)

fn compute_clarity(text: &str) -> f64 {
    let words: Vec<&str> = text
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
        .filter(|w| !w.is_empty())
        .collect();

    if words.is_empty() {
        return 0.0;
    }

    let unique: HashSet<&str> = words.iter().copied().collect();
    // V9: Root-TTR normalization — TTR = unique / √total.
    let ttr = unique.len() as f64 / (words.len() as f64).sqrt();

    // Sentence diversity
    let sentences: Vec<&str> = text
        .split(['.', '!', '?', '\n'])
        .filter(|s| !s.trim().is_empty())
        .collect();

    let sentence_diversity = if sentences.len() > 1 {
        let unique_starts: HashSet<&str> = sentences
            .iter()
            .filter_map(|s| s.split_whitespace().next())
            .collect();
        unique_starts.len() as f64 / sentences.len() as f64
    } else {
        0.5
    };

    // Flesch-Kincaid Reading Ease (Kincaid et al., 1975)
    // FK = 206.835 - 1.015 × (words/sentences) - 84.6 × (syllables/words)
    // Score: 100 = very easy, 0 = very hard. Normalized to [0, 1].
    let fk_score = compute_flesch_kincaid(&words, &sentences);

    // Weighted: 40% TTR + 15% sentence diversity + 30% Flesch-Kincaid + 15% length consistency
    let length_consistency = compute_sentence_length_consistency(&sentences);
    (0.4 * ttr + 0.15 * sentence_diversity + 0.3 * fk_score + 0.15 * length_consistency).min(1.0)
}

/// Flesch-Kincaid Reading Ease — normalized to [0, 1].
/// FK = 206.835 - 1.015 × ASL - 84.6 × ASW
/// Where ASL = average sentence length, ASW = average syllables per word.
fn compute_flesch_kincaid(words: &[&str], sentences: &[&str]) -> f64 {
    let num_sentences = sentences.len().max(1) as f64;
    let num_words = words.len().max(1) as f64;
    let total_syllables: f64 = words.iter().map(|w| count_syllables(w) as f64).sum();

    let asl = num_words / num_sentences; // Average Sentence Length
    let asw = total_syllables / num_words; // Average Syllables per Word

    let fk = 206.835 - 1.015 * asl - 84.6 * asw;
    // Normalize: FK 100 → 1.0 (very easy), FK 0 → 0.0 (very hard)
    (fk / 100.0).clamp(0.0, 1.0)
}

/// Count syllables in an English word (heuristic).
/// Uses vowel group counting with corrections for silent-e and common patterns.
fn count_syllables(word: &str) -> usize {
    let word = word.to_lowercase();
    if word.len() <= 2 {
        return 1;
    }

    let vowels = ['a', 'e', 'i', 'o', 'u', 'y'];
    let mut count: usize = 0;
    let mut prev_vowel = false;
    let chars: Vec<char> = word.chars().collect();

    for &ch in &chars {
        let is_vowel = vowels.contains(&ch);
        if is_vowel && !prev_vowel {
            count += 1;
        }
        prev_vowel = is_vowel;
    }

    // Silent 'e' at end (except -le, -ee, -ie)
    if word.ends_with('e')
        && !word.ends_with("le")
        && !word.ends_with("ee")
        && !word.ends_with("ie")
    {
        count = count.saturating_sub(1);
    }

    count.max(1)
}

/// Measure consistency of sentence lengths (low variance = more readable).
fn compute_sentence_length_consistency(sentences: &[&str]) -> f64 {
    if sentences.len() < 2 {
        return 0.5;
    }

    let lengths: Vec<f64> = sentences
        .iter()
        .map(|s| s.split_whitespace().count() as f64)
        .collect();

    let mean = lengths.iter().sum::<f64>() / lengths.len() as f64;
    if mean < 1.0 {
        return 0.5;
    }

    let variance = lengths.iter().map(|l| (l - mean).powi(2)).sum::<f64>() / lengths.len() as f64;
    let cv = variance.sqrt() / mean; // Coefficient of variation

    // Low CV (consistent lengths) → high score
    // CV 0.0 → 1.0, CV 1.0+ → 0.0
    (1.0 - cv).clamp(0.0, 1.0)
}

// ─── Dimension 2: Depth (Clause Density) ─────────────────────────
// Counts subordinate clauses and causal reasoning markers.
// Hunt (1965), "Grammatical Structures Written at Three Grade Levels"

fn compute_depth(text: &str, is_code: bool) -> f64 {
    if is_code {
        return compute_code_depth(text);
    }

    let lower = text.to_lowercase();

    // Subordinate clause markers
    let clause_markers = [
        "because", "since", "although", "whereas", "while", "unless", "if", "when", "after",
        "before", "that", "which", "who", "donde", "porque", "aunque", "mientras", "cuando", "si",
    ];

    // Causal reasoning markers
    let causal_markers = [
        "therefore",
        "thus",
        "hence",
        "consequently",
        "as a result",
        "implies",
        "causes",
        "leads to",
        "results in",
        "due to",
        "por lo tanto",
        "en consecuencia",
        "implica",
        "causa",
    ];

    let clause_count = clause_markers
        .iter()
        .filter(|m| lower.contains(**m))
        .count();
    let causal_count = causal_markers
        .iter()
        .filter(|m| lower.contains(**m))
        .count();

    let sentences = text
        .split(['.', '!', '?'])
        .filter(|s| !s.trim().is_empty())
        .count()
        .max(1);

    let clause_density = clause_count as f64 / sentences as f64;
    let causal_density = causal_count as f64 / sentences as f64;

    // Normalized: clause density contributes 60%, causal 40%
    (0.6 * clause_density.min(1.0) + 0.4 * causal_density.min(1.0)).min(1.0)
}

/// F16: Code-specific depth scoring.
/// Measures structural complexity: nesting, assertions, error handling.
fn compute_code_depth(text: &str) -> f64 {
    let lines: Vec<&str> = text.lines().collect();
    if lines.is_empty() {
        return 0.0;
    }

    let mut score = 0.0;
    let non_empty = lines.iter().filter(|l| !l.trim().is_empty()).count().max(1);

    // Nesting depth (indentation levels)
    let max_indent = lines
        .iter()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.len() - l.trim_start().len())
        .max()
        .unwrap_or(0);
    // 4+ indent levels = good structural depth
    score += (max_indent as f64 / 16.0).min(0.3);

    // Assertions and error handling = reasoning depth
    let assertion_count = lines
        .iter()
        .filter(|l| {
            let t = l.trim();
            t.starts_with("assert")
                || t.contains("assert_eq")
                || t.contains("assert!")
                || t.starts_with("raise")
                || t.starts_with("return Err")
        })
        .count();
    score += (assertion_count as f64 * 0.1).min(0.3);

    // Control flow complexity (if/for/while/match)
    let control_flow = lines
        .iter()
        .filter(|l| {
            let t = l.trim();
            t.starts_with("if ")
                || t.starts_with("for ")
                || t.starts_with("while ")
                || t.starts_with("match ")
                || t.starts_with("elif ")
                || t.starts_with("else")
        })
        .count();
    score += (control_flow as f64 / non_empty as f64).min(0.2);

    // Function/class definitions = structural organization
    let definitions = lines
        .iter()
        .filter(|l| {
            let t = l.trim();
            t.starts_with("def ")
                || t.starts_with("fn ")
                || t.starts_with("class ")
                || t.starts_with("pub fn ")
                || t.starts_with("impl ")
        })
        .count();
    score += (definitions as f64 * 0.05).min(0.2);

    score.clamp(0.0, 1.0)
}

// ─── Dimension 3: Breadth (Noun Diversity) ───────────────────────
// Measures topic diversity through unique noun-like words.
// Higher ratio = broader exploration of concepts.

fn compute_breadth(text: &str) -> f64 {
    let words: Vec<String> = text
        .split_whitespace()
        .map(|w| {
            w.trim_matches(|c: char| !c.is_alphanumeric())
                .to_lowercase()
        })
        .filter(|w| w.len() > 3) // Skip short words (articles, prepositions)
        .collect();

    if words.is_empty() {
        return 0.0;
    }

    // Filter out common stopwords to focus on content words
    let stopwords = crate::engine::shared_utils::stopwords();

    let content_words: Vec<&str> = words
        .iter()
        .map(|w| w.as_str())
        .filter(|w| !stopwords.contains(w))
        .collect();

    if content_words.is_empty() {
        return 0.0;
    }

    let unique: HashSet<&&str> = content_words.iter().collect();
    let breadth_ratio = unique.len() as f64 / content_words.len() as f64;

    // Also check for domain markers (technical terms hint at breadth)
    let domain_markers = [
        "database",
        "api",
        "server",
        "client",
        "cache",
        "queue",
        "model",
        "schema",
        "endpoint",
        "middleware",
        "service",
        "algorithm",
        "pattern",
        "architecture",
        "protocol",
        "base de datos",
        "servidor",
        "algoritmo",
        "patrón",
    ];
    let domain_count = domain_markers
        .iter()
        .filter(|m| text.to_lowercase().contains(**m))
        .count();
    let domain_bonus = (domain_count as f64 * 0.05).min(0.2);

    (breadth_ratio + domain_bonus).min(1.0)
}

// ─── Dimension 4: Logic (Connective Diversity) ───────────────────
// Measures logical structure through connective types and conditional chains.
// ROSCOE (Golovneva et al., 2023, ICLR)

fn compute_logic(text: &str) -> f64 {
    let lower = text.to_lowercase();

    // Different types of logical connectives
    let additive = [
        "and",
        "also",
        "furthermore",
        "moreover",
        "additionally",
        "además",
        "también",
    ];
    let contrastive = [
        "but",
        "however",
        "although",
        "yet",
        "nevertheless",
        "pero",
        "sin embargo",
    ];
    let causal = [
        "because",
        "therefore",
        "thus",
        "so",
        "hence",
        "porque",
        "por lo tanto",
    ];
    let conditional = [
        "if",
        "unless",
        "when",
        "provided",
        "assuming",
        "si",
        "a menos que",
    ];
    let sequential = [
        "first", "then", "next", "finally", "after", "primero", "luego", "después",
    ];

    let categories_present = [
        additive.iter().any(|c| lower.contains(c)),
        contrastive.iter().any(|c| lower.contains(c)),
        causal.iter().any(|c| lower.contains(c)),
        conditional.iter().any(|c| lower.contains(c)),
        sequential.iter().any(|c| lower.contains(c)),
    ];

    let diversity = categories_present.iter().filter(|&&x| x).count() as f64 / 5.0;

    // Check for conclusion presence (strong logical structure indicator)
    let has_conclusion = [
        "therefore",
        "in conclusion",
        "recommend",
        "the best",
        "should",
        "must",
        "por lo tanto",
        "en conclusión",
        "recomend",
    ]
    .iter()
    .any(|c| lower.contains(c));

    let conclusion_bonus = if has_conclusion { 0.15 } else { 0.0 };

    // Check conditional chain depth (if...then patterns)
    let if_count = lower.matches("if ").count() + lower.matches("si ").count();
    let chain_depth_bonus = (if_count as f64 * 0.05).min(0.15);

    (diversity + conclusion_bonus + chain_depth_bonus).min(1.0)
}

// ─── Dimension 5: Relevance (Keyword Similarity) ────────────────
// Cosine-like similarity between thought keywords and context.
// Salton (1975), "Vector Space Model"

fn compute_relevance(text: &str, context_keywords: &[&str]) -> f64 {
    if context_keywords.is_empty() {
        return 0.5; // Neutral when no context provided
    }

    let text_lower = text.to_lowercase();
    let text_words: HashSet<String> = text_lower
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
        .filter(|w| w.len() > 2)
        .collect();

    let matches = context_keywords
        .iter()
        .filter(|kw| {
            let kw_lower = kw.to_lowercase();
            text_words.contains(&kw_lower) || text_lower.contains(&kw_lower)
        })
        .count();

    let relevance = matches as f64 / context_keywords.len() as f64;
    relevance.min(1.0)
}

// ─── Dimension 6: Actionability ──────────────────────────────────
// Measures specificity and imperative language.
// GRACE (Guan et al., 2024)

fn compute_actionability(text: &str, is_code: bool) -> f64 {
    let lower = text.to_lowercase();

    // F16: Code inputs get automatic actionability boost
    if is_code {
        let mut score: f64 = 0.3; // Base: code IS actionable by nature
                                  // Assertions = verifiable
        let has_asserts = lower.contains("assert");
        if has_asserts {
            score += 0.2;
        }
        // Return values = concrete output
        let has_return = lower.contains("return ");
        if has_return {
            score += 0.1;
        }
        // Numbers/constants
        if text.chars().any(|c| c.is_ascii_digit()) {
            score += 0.1;
        }
        // Function definitions
        if lower.contains("def ") || lower.contains("fn ") {
            score += 0.1;
        }
        // Imports = concrete dependencies
        if lower.contains("import ") || lower.contains("use ") {
            score += 0.1;
        }
        return score.clamp(0.0, 1.0);
    }

    // Imperative verbs / action language
    let imperative_markers = [
        "implement",
        "create",
        "add",
        "remove",
        "modify",
        "use",
        "run",
        "build",
        "deploy",
        "configure",
        "set",
        "update",
        "implementar",
        "crear",
        "agregar",
        "eliminar",
        "modificar",
        "usar",
        "ejecutar",
        "construir",
        "configurar",
        "actualizar",
    ];

    let imperative_count = imperative_markers
        .iter()
        .filter(|m| lower.contains(**m))
        .count();

    // Specificity: numbers, measurements, file paths, code references
    let has_numbers = text.chars().any(|c| c.is_ascii_digit());
    let has_code = text.contains('`') || text.contains("```") || text.contains("::");
    let has_path = text.contains('/') || text.contains('\\');

    // Vagueness penalty
    let vague_markers = [
        "maybe",
        "perhaps",
        "probably",
        "might",
        "somehow",
        "something",
        "somehow",
        "kind of",
        "sort of",
        "quizás",
        "tal vez",
        "probablemente",
        "algo así",
    ];
    let vague_count = vague_markers.iter().filter(|m| lower.contains(**m)).count();

    let action_score = (imperative_count as f64 * 0.1).min(0.5);
    let specificity_score = (has_numbers as u8 as f64 * 0.15)
        + (has_code as u8 as f64 * 0.15)
        + (has_path as u8 as f64 * 0.1);
    let vague_penalty = (vague_count as f64 * 0.1).min(0.3);

    (action_score + specificity_score - vague_penalty).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clarity_high_ttr() {
        let text = "The database schema requires careful migration with zero downtime.";
        let clarity = compute_clarity(text);
        assert!(clarity > 0.5, "Expected high clarity, got {:.2}", clarity);
    }

    #[test]
    fn test_clarity_low_ttr() {
        let text = "the the the the the the the the";
        let clarity = compute_clarity(text);
        assert!(clarity < 0.65, "Expected low clarity, got {:.2}", clarity);
    }

    #[test]
    fn test_depth_with_causal_chains() {
        let text = "Because the cache is stale, therefore the API returns old data. \
                     This implies we need a TTL strategy.";
        let depth = compute_depth(text, false);
        assert!(depth > 0.3, "Expected measurable depth, got {:.2}", depth);
    }

    #[test]
    fn test_logic_connective_diversity() {
        let text = "First, we analyze the problem. However, the data is incomplete. \
                     If we proceed, then we risk errors. Therefore, we should validate first.";
        let logic = compute_logic(text);
        assert!(logic > 0.5, "Expected high logic score, got {:.2}", logic);
    }

    #[test]
    fn test_relevance_matching() {
        let keywords = vec!["database", "migration", "schema"];
        let text = "The database migration requires updating the schema carefully.";
        let relevance = compute_relevance(text, &keywords);
        assert!(
            (relevance - 1.0).abs() < 0.01,
            "Expected 100% relevance, got {:.2}",
            relevance
        );
    }

    #[test]
    fn test_relevance_no_match() {
        let keywords = vec!["quantum", "physics"];
        let text = "The database migration requires updating the schema.";
        let relevance = compute_relevance(text, &keywords);
        assert!(
            relevance < 0.1,
            "Expected low relevance, got {:.2}",
            relevance
        );
    }

    #[test]
    fn test_actionability_specific() {
        let text = "Implement the caching layer using Redis at port 6380. \
                     Create file `cache_service.rs` with TTL of 300 seconds.";
        let actionability = compute_actionability(text, false);
        assert!(
            actionability > 0.3,
            "Expected high actionability, got {:.2}",
            actionability
        );
    }

    #[test]
    fn test_actionability_vague() {
        let text = "Maybe we should probably do something about the thing somehow.";
        let actionability = compute_actionability(text, false);
        assert!(
            actionability < 0.1,
            "Expected low actionability for vague text, got {:.2}",
            actionability
        );
    }

    #[test]
    fn test_weighted_mean_define_stage() {
        let scores = QualityScores {
            clarity: 0.8,
            depth: 0.5,
            breadth: 0.4,
            logic: 0.6,
            relevance: 0.7,
            actionability: 0.3,
        };
        // V9: DEFINE boosts clarity 2x → [2, 1, 1, 1, 1, 1] = total_weight 7
        // weighted = (0.8*2 + 0.5 + 0.4 + 0.6 + 0.7 + 0.3) / 7 = 4.1/7 ≈ 0.5857
        let wm = scores.weighted_mean(CognitiveStage::Define);
        assert!(
            (wm - 0.5857).abs() < 0.01,
            "Expected ~0.5857, got {:.4}",
            wm
        );
    }

    #[test]
    fn test_empty_input() {
        let scores = compute_quality("", &[]);
        assert_eq!(scores.clarity, 0.0);
        assert_eq!(scores.depth, 0.0);
        assert_eq!(scores.breadth, 0.0);
    }

    // ─── NEW-2: Logistic Length Penalty Tests ─────────
    #[test]
    fn test_logistic_penalty_smooth_transition() {
        use crate::engine::budget::BudgetMode;

        let base = QualityScores {
            clarity: 0.8,
            depth: 0.7,
            breadth: 0.6,
            logic: 0.8,
            relevance: 0.7,
            actionability: 0.6,
        };
        let stage = CognitiveStage::Analyze;

        // Short text: no penalty (below threshold/2)
        let short = "Short thought.";
        let result_short = apply_length_penalty(base.clone(), short, BudgetMode::Balanced);
        let wm_short = result_short.weighted_mean(stage);
        let wm_base = base.weighted_mean(stage);
        assert!(
            (wm_short - wm_base).abs() < 0.01,
            "Short should have no penalty: base={:.3} after={:.3}",
            wm_base,
            wm_short
        );

        // Very long text (2000 words): should have measurable penalty
        let long: String = (0..2000)
            .map(|i| format!("word{}", i))
            .collect::<Vec<_>>()
            .join(" ");
        let result_long = apply_length_penalty(base.clone(), &long, BudgetMode::Balanced);
        let wm_long = result_long.weighted_mean(stage);
        assert!(
            wm_long < wm_base * 0.85,
            "Long text should have penalty: base={:.3} after={:.3}",
            wm_base,
            wm_long
        );
    }

    #[test]
    fn test_logistic_penalty_monotonic() {
        use crate::engine::budget::BudgetMode;

        let base = QualityScores {
            clarity: 0.8,
            depth: 0.7,
            breadth: 0.6,
            logic: 0.8,
            relevance: 0.7,
            actionability: 0.6,
        };
        let stage = CognitiveStage::Analyze;

        // Penalty should decrease quality monotonically as length increases
        let lengths = [50, 200, 500, 1000, 2000];
        let scores: Vec<f64> = lengths
            .iter()
            .map(|&n| {
                let text: String = (0..n)
                    .map(|i| format!("w{}", i))
                    .collect::<Vec<_>>()
                    .join(" ");
                let result = apply_length_penalty(base.clone(), &text, BudgetMode::Balanced);
                result.weighted_mean(stage)
            })
            .collect();
        for w in scores.windows(2) {
            assert!(
                w[0] >= w[1],
                "Longer text should have lower score: {:.3} >= {:.3}",
                w[0],
                w[1]
            );
        }
    }

    // ─── Vector 2: Shannon Word Entropy Tests ─────────
    #[test]
    fn test_word_entropy_pure_padding() {
        // Pure repetition → entropy ≈ 0
        let entropy = super::compute_word_entropy("word word word word word word");
        assert!(
            entropy < 0.01,
            "Pure padding should have near-zero entropy: {:.3}",
            entropy
        );
    }

    #[test]
    fn test_word_entropy_diverse_text() {
        // Diverse analysis → high entropy
        let entropy = super::compute_word_entropy(
            "PostgreSQL B-tree index provides logarithmic lookup reducing query latency",
        );
        assert!(
            entropy > 0.85,
            "Diverse text should have high entropy: {:.3}",
            entropy
        );
    }

    #[test]
    fn test_word_entropy_empty() {
        assert_eq!(super::compute_word_entropy(""), 0.0);
    }

    #[test]
    fn test_padding_activates_steeper_penalty() {
        use crate::engine::budget::BudgetMode;

        let base = QualityScores {
            clarity: 0.8,
            depth: 0.7,
            breadth: 0.6,
            logic: 0.8,
            relevance: 0.7,
            actionability: 0.6,
        };
        let stage = CognitiveStage::Analyze;

        // Create padded text (word word word... > threshold with low entropy)
        let threshold = BudgetMode::Balanced.length_penalty_threshold();
        let padded: String = (0..(threshold + 100))
            .map(|_| "word")
            .collect::<Vec<_>>()
            .join(" ");

        // Create diverse text of same length
        let diverse: String = (0..(threshold + 100))
            .map(|i| format!("unique{}", i))
            .collect::<Vec<_>>()
            .join(" ");

        let result_padded = apply_length_penalty(base.clone(), &padded, BudgetMode::Balanced);
        let result_diverse = apply_length_penalty(base.clone(), &diverse, BudgetMode::Balanced);

        // Padded should be penalized MORE than diverse (steeper K)
        assert!(
            result_padded.weighted_mean(stage) < result_diverse.weighted_mean(stage),
            "Padded text should get heavier penalty: padded={:.3} diverse={:.3}",
            result_padded.weighted_mean(stage),
            result_diverse.weighted_mean(stage)
        );
    }

    // ─── L2: Lempel-Ziv LZ76 Complexity Tests ────────
    #[test]
    fn test_lz76_markov_chain_padding() {
        // Markov cycling: high Shannon but low LZ76 (repeating blocks).
        // LLM padding requires enough repetitions for LZ76 to detect patterns.
        // 2-word cycle repeated many times — clearly repetitive.
        let simple_cycle =
            "foo bar foo bar foo bar foo bar foo bar foo bar foo bar foo bar foo bar foo bar";
        let lz76_simple = super::compute_lz76_complexity(simple_cycle);
        assert!(
            lz76_simple < 0.50,
            "Simple cycle should have low LZ76: {:.3}",
            lz76_simple
        );

        // Longer Markov chain — realistic LLM padding with diverse words
        let markov: String = (0..20)
            .map(|i| {
                if i % 2 == 0 {
                    "the system fails because"
                } else {
                    "the database fails because"
                }
            })
            .collect::<Vec<_>>()
            .join(" ");
        let lz76_markov = super::compute_lz76_complexity(&markov);
        assert!(
            lz76_markov < 0.50,
            "Markov chain should have low LZ76: {:.3}",
            lz76_markov
        );

        // Verify Shannon is fooled by the Markov chain
        let shannon = super::compute_word_entropy(&markov);
        assert!(
            shannon > 0.85,
            "Shannon should be high (fooled): {:.3}",
            shannon
        );
    }

    #[test]
    fn test_lz76_diverse_text() {
        let diverse = "PostgreSQL B-tree index provides logarithmic lookup complexity reducing query latency for large datasets with proper vacuum scheduling";
        let lz76 = super::compute_lz76_complexity(diverse);
        assert!(
            lz76 > 0.50,
            "Diverse text should have high LZ76: {:.3}",
            lz76
        );
    }

    #[test]
    fn test_lz76_empty() {
        assert_eq!(super::compute_lz76_complexity(""), 0.0);
    }
}
