// src/engine/mod.rs
pub mod micro_prm;
pub mod sandbox;
pub mod shared_utils;

// ─── Phase 1: Cognitive Core (v3.0) ─────────────────────────────
pub mod anti_hallucination;
pub mod bias_detector;
pub mod budget;
pub mod ewma_reward;
pub mod formatter;
pub mod metacognition;
pub mod quality_metrics;
pub mod stage_engine;

// ─── Phase 3: Semantics (v3.1) ──────────────────────────────────
pub mod claim_grounding;
pub mod contradiction_detector;
pub mod novelty_tracker;
pub mod semantic_similarity;

// ─── Phase 5: Deep Reasoning (v3.2) ─────────────────────────────
pub mod corrective_directives;
pub mod stage_validator;
pub mod thought_session;
