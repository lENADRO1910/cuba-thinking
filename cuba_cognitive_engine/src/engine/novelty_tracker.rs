// src/engine/novelty_tracker.rs
//
// S4: Semantic Novelty / Information Gain Tracking
//
// UPGRADED: Jaccard-only → Jaccard + Semantic Embedding Similarity.
// Jaccard detects new vocabulary; embeddings detect paraphrasing.
//
// Algorithm:
// 1. Maintain running set of seen content terms
// 2. For each new thought, compute new_terms / total_terms (Jaccard)
// 3. Compute semantic similarity to most recent thought via embeddings
// 4. If semantically similar (>0.85), penalize novelty by 30% (paraphrase detection)
// 5. High novelty = thought introduces genuinely fresh concepts
// 6. Low novelty = thought repeats or paraphrases what's already been said

use serde::Serialize;
use std::collections::HashSet;

/// Tracks cumulative vocabulary for novelty computation.
#[derive(Debug, Clone, Serialize)]
pub struct NoveltyTracker {
    /// All unique content terms seen so far.
    seen_terms: HashSet<String>,
    /// Number of thoughts analyzed.
    step_count: usize,
    /// Most recent thought text (for semantic similarity comparison).
    #[serde(skip)]
    last_thought: Option<String>,
}

impl NoveltyTracker {
    pub fn new() -> Self {
        Self {
            seen_terms: HashSet::new(),
            step_count: 0,
            last_thought: None,
        }
    }

    /// Track a new thought and compute its novelty score.
    ///
    /// Returns info_gain in [0.0, 1.0]:
    /// - 1.0 = all terms are completely new (maximum information gain)
    /// - 0.0 = all terms were already seen (no new information)
    ///
    /// Combines Jaccard vocabulary novelty with semantic embedding similarity.
    /// If the new thought is semantically very similar to the previous one
    /// (cosine > 0.85), novelty is penalized by 30% to catch paraphrasing.
    pub fn track_novelty(&mut self, thought: &str) -> f64 {
        self.step_count += 1;

        let content_terms = extract_content_terms(thought);
        if content_terms.is_empty() {
            return 0.0;
        }

        // Count genuinely new terms (Jaccard component)
        let new_count = content_terms
            .iter()
            .filter(|t| !self.seen_terms.contains(*t))
            .count();

        let mut novelty = new_count as f64 / content_terms.len() as f64;

        // Semantic paraphrase detection via embeddings
        if let Some(ref prev) = self.last_thought {
            let sim = crate::engine::semantic_similarity::semantic_similarity(thought, prev);
            if sim > 0.85 {
                // High semantic similarity despite new words → likely paraphrase
                novelty *= 0.7; // 30% penalty
            }
        }

        // Update state
        self.seen_terms.extend(content_terms);
        self.last_thought = Some(thought.to_string());

        novelty.clamp(0.0, 1.0)
    }

}

/// Extract content terms from text (filtered, normalized).
///
/// V8: Strips comments before extraction to prevent novelty evasion.
/// An LLM could inflate novelty by appending `# random_hash_123` to each thought.
fn extract_content_terms(text: &str) -> Vec<String> {
    let stopwords = crate::engine::shared_utils::stopwords();

    // V8: Strip comment lines (Python # and Rust //) before term extraction.
    // This prevents inflating novelty with random comment strings.
    let stripped: String = text
        .lines()
        .map(|line| {
            let trimmed = line.trim();
            if trimmed.starts_with('#') || trimmed.starts_with("//") {
                "" // Drop pure comment lines
            } else if let Some(pos) = line.find(" # ") {
                &line[..pos] // Strip inline Python comments
            } else if let Some(pos) = line.find(" // ") {
                &line[..pos] // Strip inline Rust comments
            } else {
                line
            }
        })
        .collect::<Vec<&str>>()
        .join(" ");

    stripped
        .split_whitespace()
        .map(|w| {
            w.chars()
                .filter(|c| c.is_alphanumeric())
                .collect::<String>()
                .to_lowercase()
        })
        .filter(|w| w.len() > 2 && !stopwords.contains(w.as_str()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_first_thought_high_novelty() {
        let mut tracker = NoveltyTracker::new();
        let novelty = tracker
            .track_novelty("Implement database migration with zero downtime using PostgreSQL");
        assert_eq!(
            novelty, 1.0,
            "First thought should have 100% novelty: {:.3}",
            novelty
        );
    }

    #[test]
    fn test_repeated_thought_low_novelty() {
        let mut tracker = NoveltyTracker::new();
        let text = "Implement database migration with zero downtime";
        tracker.track_novelty(text);
        let novelty = tracker.track_novelty(text);
        assert_eq!(
            novelty, 0.0,
            "Repeated thought should have 0% novelty: {:.3}",
            novelty
        );
    }

    #[test]
    fn test_partially_new_medium_novelty() {
        let mut tracker = NoveltyTracker::new();
        tracker.track_novelty("The database needs migration planning");
        let novelty = tracker.track_novelty("The database also needs caching and monitoring");
        assert!(
            novelty > 0.0 && novelty < 1.0,
            "Partially new thought should have medium novelty: {:.3}",
            novelty
        );
    }

    #[test]
    fn test_natural_decay() {
        let mut tracker = NoveltyTracker::new();
        let n1 = tracker.track_novelty("Database migration strategy for PostgreSQL");
        let n2 = tracker.track_novelty("Cache invalidation patterns with Redis cluster");
        let n3 = tracker.track_novelty("Database migration cache Redis PostgreSQL strategy");
        assert!(n1 >= n2, "Novelty should generally decay: {} >= {}", n1, n2);
        assert!(
            n3 < n1,
            "Repeated vocabulary should show lower novelty: {} < {}",
            n3,
            n1
        );
    }

    #[test]
    fn test_empty_thought_zero_novelty() {
        let mut tracker = NoveltyTracker::new();
        let novelty = tracker.track_novelty("");
        assert_eq!(novelty, 0.0, "Empty thought should have 0 novelty");
    }
}
