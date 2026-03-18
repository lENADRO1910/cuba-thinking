// src/engine/memory_bridge.rs
//
// R11: Cross-MCP Memory Symbiosis (Cortex-Hippocampus)
//
// Bridge between cuba-thinking and cuba-memorys MCP server.
// Generates formatted instructions for the LLM to invoke
// memory tools at specific cognitive stages.
//
// Based on Complementary Learning Systems theory
// (McClelland, Rumelhart & the PDP Group, 1995).
//
// Pattern: RECALL before reasoning, CONSOLIDATE after conclusion.

use crate::engine::stage_engine::CognitiveStage;
use serde::Serialize;

/// Memory bridge instruction to be included in the tool response.
#[derive(Debug, Clone, Serialize)]
pub struct MemoryInstruction {
    /// The MCP tool to call.
    pub tool_name: &'static str,
    /// Formatted arguments for the tool call.
    pub arguments: serde_json::Value,
    /// Human-readable reason for the instruction.
    pub reason: &'static str,
}

/// Generate memory bridge instructions based on cognitive stage.
///
/// Returns instructions for the LLM to invoke cuba-memorys tools
/// at appropriate reasoning stages:
/// - DEFINE (thought ≤ 2): RECALL — search past knowledge
/// - SYNTHESIZE (final thought): CONSOLIDATE — save lesson learned
pub fn generate_memory_instructions(
    stage: CognitiveStage,
    thought_number: usize,
    is_final_thought: bool,
    thought_content: &str,
) -> Vec<MemoryInstruction> {
    let mut instructions = Vec::new();

    // ─── RECALL at DEFINE stage (early thoughts) ─────────────────
    if stage == CognitiveStage::Define && thought_number <= 2 {
        // Extract key terms for search query
        let query = extract_search_query(thought_content);

        // 1. Search past knowledge
        instructions.push(MemoryInstruction {
            tool_name: "cuba_faro",
            arguments: serde_json::json!({
                "query": query,
                "mode": "hybrid",
                "limit": 5
            }),
            reason: "Search past knowledge before reasoning (cortex recall)",
        });

        // 2. Check past errors
        instructions.push(MemoryInstruction {
            tool_name: "cuba_expediente",
            arguments: serde_json::json!({
                "query": query
            }),
            reason: "Check if similar errors occurred before (error avoidance)",
        });
    }

    // ─── Phase 5E: VALIDATE at HYPOTHESIZE (anti-repetition) ─────
    if stage == CognitiveStage::Hypothesize {
        let query = extract_search_query(thought_content);
        instructions.push(MemoryInstruction {
            tool_name: "cuba_expediente",
            arguments: serde_json::json!({
                "query": query,
                "proposed_action": thought_content.chars().take(200).collect::<String>()
            }),
            reason: "Anti-repetition guard: check if similar approach failed before",
        });
    }

    // ─── Phase 5E: GROUND at VERIFY (claim verification) ─────────
    if stage == CognitiveStage::Verify {
        let query = extract_search_query(thought_content);
        instructions.push(MemoryInstruction {
            tool_name: "cuba_faro",
            arguments: serde_json::json!({
                "query": query,
                "mode": "verify",
                "limit": 3
            }),
            reason: "Verify claims against stored evidence (grounding check)",
        });
    }

    // ─── CONSOLIDATE at SYNTHESIZE stage (final thought) ─────────
    if stage == CognitiveStage::Synthesize && is_final_thought {
        // Extract lesson from conclusion
        let lesson = extract_lesson_summary(thought_content);

        instructions.push(MemoryInstruction {
            tool_name: "cuba_cronica",
            arguments: serde_json::json!({
                "action": "add",
                "entity_name": "reasoning_session",
                "content": lesson,
                "observation_type": "lesson",
                "source": "agent"
            }),
            reason: "Consolidate lesson learned (hippocampus consolidation)",
        });
    }

    instructions
}

/// Extract key terms for search query from thought content.
fn extract_search_query(thought: &str) -> String {
    // Extract first 3 significant words (>4 chars, not common)
    let stopwords = [
        "this", "that", "with", "from", "what", "when", "where", "which", "there", "their",
        "about", "would", "could", "should", "have", "been", "some", "para", "como", "esta",
        "este", "esto", "pero", "sino", "donde", "puede", "tiene",
    ];

    let keywords: Vec<&str> = thought
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
        .filter(|w| w.len() > 4 && !stopwords.contains(w))
        .take(5)
        .collect();

    keywords.join(" ")
}

/// Extract a concise lesson summary from conclusion text.
fn extract_lesson_summary(thought: &str) -> String {
    // Take first 200 chars of the thought as lesson summary
    let truncated: String = thought.chars().take(200).collect();
    if thought.len() > 200 {
        format!("{}...", truncated)
    } else {
        truncated
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recall_at_define() {
        let instructions = generate_memory_instructions(
            CognitiveStage::Define,
            1,
            false,
            "Define the database migration strategy for PostgreSQL",
        );
        assert_eq!(instructions.len(), 2);
        assert_eq!(instructions[0].tool_name, "cuba_faro");
        assert_eq!(instructions[1].tool_name, "cuba_expediente");
    }

    #[test]
    fn test_no_recall_at_analyze() {
        let instructions = generate_memory_instructions(
            CognitiveStage::Analyze,
            5,
            false,
            "Analyzing trade-offs between approaches",
        );
        assert!(instructions.is_empty());
    }

    #[test]
    fn test_consolidate_at_synthesize() {
        let instructions = generate_memory_instructions(
            CognitiveStage::Synthesize,
            8,
            true,
            "In conclusion, the best approach is to use connection pooling with PgBouncer",
        );
        assert_eq!(instructions.len(), 1);
        assert_eq!(instructions[0].tool_name, "cuba_cronica");
    }

    #[test]
    fn test_no_consolidate_if_not_final() {
        let instructions = generate_memory_instructions(
            CognitiveStage::Synthesize,
            8,
            false, // NOT final thought
            "In conclusion so far...",
        );
        assert!(instructions.is_empty());
    }

    #[test]
    fn test_search_query_extraction() {
        let query =
            extract_search_query("Define the database migration strategy for PostgreSQL cluster");
        assert!(query.contains("database"));
        assert!(query.contains("migration"));
    }
}
