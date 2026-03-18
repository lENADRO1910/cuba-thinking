// src/engine/semantic_similarity.rs
//
// S1/S3: Step Transition Coherence via TF-IDF Cosine Similarity
//
// Measures semantic coherence between consecutive thoughts using
// bag-of-words TF-IDF cosine similarity (Salton, 1975).
// Replaces hardcoded `coherence: 1.0` in EWMA signals.
//
// Algorithm:
// 1. Tokenize both texts into normalized word vectors
// 2. Build TF vectors (term frequency per document)
// 3. Compute cosine similarity: cos(θ) = (A·B) / (|A|·|B|)
//
// Complexity: O(n) where n = max(|A|, |B|) vocabulary size.

use std::collections::HashMap;

/// Compute coherence between current thought and previous thought.
/// Returns 0.0 (completely unrelated) to 1.0 (identical topics).
///
/// If no previous thought exists, returns 1.0 (first step is always coherent).
pub fn compute_coherence(current: &str, previous: Option<&str>) -> f64 {
    let prev = match previous {
        Some(p) if !p.is_empty() => p,
        _ => return 1.0, // First thought or empty previous → coherent by default
    };

    if current.is_empty() {
        return 0.0;
    }

    let tf_current = build_tf_vector(current);
    let tf_previous = build_tf_vector(prev);

    cosine_similarity(&tf_current, &tf_previous)
}

/// Build a term-frequency vector from text.
/// Filters stopwords and normalizes to lowercase.
fn build_tf_vector(text: &str) -> HashMap<String, f64> {
    let stopwords = crate::engine::shared_utils::stopwords();

    let mut tf: HashMap<String, f64> = HashMap::new();
    let mut total = 0.0_f64;

    for word in text.split_whitespace() {
        let clean: String = word
            .chars()
            .filter(|c| c.is_alphanumeric())
            .collect::<String>()
            .to_lowercase();

        if clean.len() > 2 && !stopwords.contains(clean.as_str()) {
            *tf.entry(clean).or_insert(0.0) += 1.0;
            total += 1.0;
        }
    }

    // Normalize to term frequency
    if total > 0.0 {
        for val in tf.values_mut() {
            *val /= total;
        }
    }

    tf
}

/// Cosine similarity between two TF vectors.
/// cos(θ) = (A·B) / (|A| × |B|)
fn cosine_similarity(a: &HashMap<String, f64>, b: &HashMap<String, f64>) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }

    let dot_product: f64 = a
        .iter()
        .filter_map(|(term, tf_a)| b.get(term).map(|tf_b| tf_a * tf_b))
        .sum();

    let mag_a: f64 = a.values().map(|v| v * v).sum::<f64>().sqrt();
    let mag_b: f64 = b.values().map(|v| v * v).sum::<f64>().sqrt();

    if mag_a == 0.0 || mag_b == 0.0 {
        return 0.0;
    }

    (dot_product / (mag_a * mag_b)).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identical_thoughts_high_coherence() {
        let text = "The database migration requires careful planning with zero downtime";
        let coherence = compute_coherence(text, Some(text));
        assert!(
            coherence > 0.95,
            "Identical texts should have near-perfect coherence: {:.3}",
            coherence
        );
    }

    #[test]
    fn test_unrelated_thoughts_low_coherence() {
        let current = "Implement Redis caching with TTL expiration for session data";
        let previous = "The weather forecast predicts heavy rainfall in tropical regions";
        let coherence = compute_coherence(current, Some(previous));
        assert!(
            coherence < 0.2,
            "Unrelated texts should have low coherence: {:.3}",
            coherence
        );
    }

    #[test]
    fn test_related_thoughts_medium_coherence() {
        let current = "The API endpoint needs rate limiting to prevent abuse";
        let previous = "We should add authentication middleware to protect the API routes";
        let coherence = compute_coherence(current, Some(previous));
        assert!(
            coherence > 0.1,
            "Related texts should have measurable coherence: {:.3}",
            coherence
        );
    }

    #[test]
    fn test_no_previous_thought_returns_one() {
        let coherence = compute_coherence("First thought here", None);
        assert_eq!(coherence, 1.0, "No previous thought should return 1.0");
    }

    #[test]
    fn test_empty_previous_returns_one() {
        let coherence = compute_coherence("Current thought", Some(""));
        assert_eq!(coherence, 1.0, "Empty previous should return 1.0");
    }

    #[test]
    fn test_empty_current_returns_zero() {
        let coherence = compute_coherence("", Some("Previous thought"));
        assert_eq!(coherence, 0.0, "Empty current should return 0.0");
    }

    #[test]
    fn test_spanish_coherence() {
        let current = "La base de datos necesita migración con indices optimizados";
        let previous = "Planificar la migración de la base de datos PostgreSQL";
        let coherence = compute_coherence(current, Some(previous));
        assert!(
            coherence > 0.2,
            "Related Spanish texts should have coherence: {:.3}",
            coherence
        );
    }
}
