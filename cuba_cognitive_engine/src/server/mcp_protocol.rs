
use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;
use tokio::io::{self, AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::time::Instant;

use super::observability::RedMetrics;

use crate::engine::sandbox::LocalReasoningEngine;

/// JSON-RPC 2.0 Request ID
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum RequestId {
    String(String),
    Number(i64),
    Null,
}

/// JSON-RPC 2.0 Error Object
#[derive(Debug, Serialize)]
pub struct RpcError {
    pub code: i64,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

/// Incoming JSON-RPC Request
#[derive(Debug, Deserialize)]
pub struct RpcRequest {
    #[allow(dead_code)]
    pub jsonrpc: String,
    pub id: Option<RequestId>, // None for notifications
    #[serde(flatten)]
    pub action: McpAction,
}

/// JSON-RPC Strict Enum Action Matching
#[derive(Debug, Deserialize)]
#[serde(tag = "method")]
pub enum McpAction {
    #[serde(rename = "initialize")]
    Initialize { #[allow(dead_code)] params: Option<Value> },

    #[serde(rename = "notifications/initialized")]
    Initialized { #[allow(dead_code)] params: Option<Value> },

    #[serde(rename = "tools/list")]
    ToolsList { #[allow(dead_code)] params: Option<Value> },

    #[serde(rename = "tools/call")]
    ToolsCall { params: ToolsCallParams },

    #[serde(rename = "control/steer")]
    ControlSteer { params: ControlSteerParams },

    #[serde(other)]
    Unknown,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ControlSteerParams {
    pub target_branch_id: usize,
    pub action: String, // e.g. "prune", "boost"
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolsCallParams {
    pub name: String,
    #[serde(default)]
    pub arguments: Option<Value>,
    #[serde(default)]
    pub _meta: Option<MetaParams>,
}

#[derive(Debug, Deserialize, Default, Clone)]
#[serde(rename_all = "camelCase")]
pub struct MetaParams {
    pub progress_token: Option<Value>,
}

/// Outgoing JSON-RPC Response (Success)
#[derive(Debug, Serialize)]
pub struct RpcSuccessResponse {
    pub jsonrpc: String,
    pub id: RequestId,
    pub result: Value,
}

/// Outgoing JSON-RPC Response (Error)
#[derive(Debug, Serialize)]
pub struct RpcErrorResponse {
    pub jsonrpc: String,
    pub id: RequestId, 
    pub error: RpcError,
}

/// Outgoing Progress Notification
#[derive(Debug, Serialize)]
pub struct RpcNotification {
    pub jsonrpc: String,
    pub method: String,
    pub params: Value,
}

#[derive(Serialize)]
#[serde(untagged)]
pub enum OutgoingMessage {
    Success(RpcSuccessResponse),
    Error(RpcErrorResponse),
    Notification(RpcNotification),
}

pub struct McpServer {
    sandbox: Arc<LocalReasoningEngine>,
    /// Phase 5A: Persistent thought sessions across calls.
    sessions: Arc<crate::engine::thought_session::SessionStore>,
    /// P2: RED metrics (Rate, Errors, Duration) per tool.
    metrics: Arc<RedMetrics>,
}

impl McpServer {
    pub fn new() -> Self {
        // DEBT-T07: Direct sandbox, no router indirection
        let sandbox = Arc::new(LocalReasoningEngine::new("cognitive-engine-v3", 2).unwrap());
        let sessions = Arc::new(crate::engine::thought_session::SessionStore::new());
        let metrics = Arc::new(RedMetrics::new());
        Self { sandbox, sessions, metrics }
    }

    /// Primary Execution Loop (STDIO)
    pub async fn run(self: Arc<Self>) -> Result<()> {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<OutgoingMessage>(100);

        // Async STDOUT Writer loop to prevent JSON interleaving
        let _stdout_handle = tokio::spawn(async move {
            let mut stdout = io::stdout();
            while let Some(msg) = rx.recv().await {
                if let Ok(mut res_str) = serde_json::to_string(&msg) {
                    res_str.push('\n');
                    let _ = stdout.write_all(res_str.as_bytes()).await;
                    let _ = stdout.flush().await;
                }
            }
        });

        let local = tokio::task::LocalSet::new();

        local.run_until(async move {
            let mut stdin = BufReader::new(io::stdin());
            let mut line = String::new();

            loop {
                line.clear();
                let bytes_read = stdin.read_line(&mut line).await.unwrap_or(0);
                if bytes_read == 0 {
                    // EOF — emit RED metrics summary before shutdown
                    self.metrics.emit_summary();
                    break;
                }

                let req_str = line.clone();
                let self_clone = self.clone();
                let tx_clone = tx.clone();

                // Spawn every request in the local task set so the stdin listener isn't blocked!
                // Essential for Phase 10 Mid-Turn Steering as the task future is !Send due to Bumpalo.
                tokio::task::spawn_local(async move {
                    match serde_json::from_str::<RpcRequest>(&req_str) {
                        Ok(req) => {
                            let result = self_clone.handle_request(&req, tx_clone.clone()).await;

                            if let Some(req_id) = req.id {
                                let msg = match result {
                                    Ok(Some(res_val)) => OutgoingMessage::Success(RpcSuccessResponse {
                                        jsonrpc: "2.0".to_string(),
                                        id: req_id,
                                        result: res_val,
                                    }),
                                    Ok(None) => return,
                                    Err(err) => OutgoingMessage::Error(RpcErrorResponse {
                                        jsonrpc: "2.0".to_string(),
                                        id: req_id,
                                        error: err,
                                    }),
                                };
                                let _ = tx_clone.send(msg).await;
                            }
                        }
                        Err(e) => {
                            tracing::error!("Failed to parse JSON-RPC: {} | Error: {}", req_str, e);
                            let err_msg = OutgoingMessage::Error(RpcErrorResponse {
                                jsonrpc: "2.0".to_string(),
                                id: RequestId::Null,
                                error: RpcError {
                                    code: -32700,
                                    message: "Parse error".to_string(),
                                    data: None,
                                },
                            });
                            let _ = tx_clone.send(err_msg).await;
                        }
                    }
                });
            }
        }).await;

        Ok(())
    }

    async fn handle_request(&self, req: &RpcRequest, tx: tokio::sync::mpsc::Sender<OutgoingMessage>) -> Result<Option<Value>, RpcError> {
        match &req.action {
            McpAction::Initialize { .. } => {
                Ok(Some(serde_json::json!({
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {}
                    },
                    "serverInfo": {
                        "name": "cuba_cognitive_engine_v3",
                        "version": "3.0.0"
                    }
                })))
            },
            McpAction::Initialized { .. } => {
                Ok(None)
            },
            McpAction::ToolsList { .. } => {
                Ok(Some(serde_json::json!({
                    "tools": [
                        {
                            "name": "cuba_thinking",
                            "description": "Cuba Cognitive Engine: Executes Deep Reasoning and MCTS evaluations.\n\nCRITICAL: DO NOT PASS NATURAL LANGUAGE. The engine is 100% Native Silicon (PyO3). You MUST pass the query in formalized SILICON LANGUAGE (Python, Z3 SMT logic, or strict JSON).\nExample: `assert x > 5` or `import z3; return res`.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "thought": {
                                        "type": "string",
                                        "description": "The thought branch to evaluate. MUST BE STRICT PROGRAMMING CODE (Python/Z3/AST evaluateable). Natural language will be rejected."
                                    },
                                    "hypothesis": {
                                        "type": "string",
                                        "description": "Mathematical Context or prior hypothesis constraints"
                                    },
                                    "confidence": {
                                        "type": "number",
                                        "description": "Your confidence in this thought (0.0 to 1.0). Engine will calibrate."
                                    },
                                    "budgetMode": {
                                        "type": "string",
                                        "description": "Reasoning depth: fast, balanced, thorough, or exhaustive",
                                        "enum": ["fast", "balanced", "thorough", "exhaustive"]
                                    },
                                    "thinkingStage": {
                                        "type": "string",
                                        "description": "Current cognitive stage",
                                        "enum": ["DEFINE", "RESEARCH", "ANALYZE", "HYPOTHESIZE", "VERIFY", "SYNTHESIZE"]
                                    },
                                    "biasDetected": {
                                        "type": "string",
                                        "description": "Self-reported bias: anchoring, confirmation, availability, sunk_cost, bandwagon"
                                    },
                                    "assumptions": {
                                        "type": "array",
                                        "items": { "type": "string" },
                                        "description": "Explicit assumptions the AI is making"
                                    },
                                    "thoughtNumber": {
                                        "type": "integer",
                                        "description": "Sequential thought number (1-based)"
                                    },
                                    "nextThoughtNeeded": {
                                        "type": "boolean",
                                        "description": "Set to false when this is the final thought"
                                    }
                                },
                                "required": ["thought"]
                            }
                        },
                        {
                            "name": "run_stress_benchmark",
                            "description": "Simulates 5000 parallel requests to validate engine networking performance.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {}
                            }
                        },
                        {
                            "name": "verify_code",
                            "description": "Execute Python code in a secure sandbox and return PRM (Process Reward Model) verdict with quality metrics. Use for quick code verification without full cognitive analysis.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "code": {
                                        "type": "string",
                                        "description": "Python code to verify (asserts, functions, computations)"
                                    }
                                },
                                "required": ["code"]
                            }
                        },
                        {
                            "name": "analyze_reasoning",
                            "description": "Analyze a multi-step reasoning chain for coherence, contradictions, novelty decay, and grounding quality. Returns semantic quality report.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "thoughts": {
                                        "type": "array",
                                        "items": { "type": "string" },
                                        "description": "Array of reasoning steps to analyze in order"
                                    },
                                    "context": {
                                        "type": "string",
                                        "description": "Optional context or hypothesis the reasoning should stay grounded to"
                                    }
                                },
                                "required": ["thoughts"]
                            }
                        }
                    ]
                })))
            },
            McpAction::ToolsCall { params } => self.handle_tool_call(params, tx).await,
            McpAction::ControlSteer { params } => {
                tracing::info!("Received Mid-Turn Steering Interruption for branch_id: {} with action: {}", params.target_branch_id, params.action);
                // Implementation for runtime interception will follow
                Ok(Some(serde_json::json!({
                    "status": "steered",
                    "branch_id": params.target_branch_id
                })))
            },
            McpAction::Unknown => {
                if req.id.is_none() {
                    Ok(None)
                } else {
                    Err(RpcError {
                        code: -32601, // Method not found
                        message: "Method not found or malformed parameters".to_string(),
                        data: None,
                    })
                }
            }
        }
    }

    async fn handle_tool_call(&self, params: &ToolsCallParams, tx: tokio::sync::mpsc::Sender<OutgoingMessage>) -> Result<Option<Value>, RpcError> {
        let tool_name = params.name.as_str();
        let call_start = std::time::Instant::now();
        let _span = tracing::info_span!("tool_call", tool = tool_name).entered();

        let result = match tool_name {
            "cuba_thinking" => self.handle_cuba_thinking_tool(params, tx).await,
            "run_stress_benchmark" => self.handle_stress_benchmark_tool(params, tx).await,
            "verify_code" => self.handle_verify_code_tool(params).await,
            "analyze_reasoning" => self.handle_analyze_reasoning_tool(params).await,
            _ => Err(RpcError {
                code: -32601,
                message: format!("Unknown tool: {}", tool_name),
                data: None,
            })
        };

        let duration = call_start.elapsed();
        let is_error = result.is_err();
        self.metrics.record_call(tool_name, duration, is_error);

        tracing::debug!(
            tool = tool_name,
            duration_ms = format!("{:.2}", duration.as_secs_f64() * 1000.0),
            success = !is_error,
            "tool call completed"
        );

        result
    }

    /// F5: Shared progress notification emitter.
    async fn emit_progress(tx: &tokio::sync::mpsc::Sender<OutgoingMessage>, token: Option<Value>, progress: f64, total: f64) {
        if let Some(t) = token {
            let notif = OutgoingMessage::Notification(RpcNotification {
                jsonrpc: "2.0".to_string(),
                method: "notifications/progress".to_string(),
                params: serde_json::json!({
                    "progressToken": t,
                    "progress": progress,
                    "total": total
                })
            });
            let _ = tx.send(notif).await;
        }
    }

    async fn handle_cuba_thinking_tool(&self, params: &ToolsCallParams, tx: tokio::sync::mpsc::Sender<OutgoingMessage>) -> Result<Option<Value>, RpcError> {
        let progress_token = params._meta.as_ref().and_then(|m| m.progress_token.clone());
        let tx_prog = tx.clone();
        
        Self::emit_progress(&tx_prog, progress_token.clone(), 5.0, 100.0).await;
        
        // ─── Parse Input Arguments ───────────────────────────────
        let arguments = params.arguments.clone().unwrap_or(serde_json::json!({}));
        let hypothesis = arguments.get("hypothesis").and_then(Value::as_str).unwrap_or("...");
        let thought = arguments.get("thought").and_then(Value::as_str).unwrap_or("...");

        // Optional parameters from the AI
        let confidence = arguments.get("confidence")
            .and_then(Value::as_f64)
            .unwrap_or(0.5);
        let budget_str = arguments.get("budgetMode")
            .and_then(Value::as_str);
        let explicit_stage = arguments.get("thinkingStage")
            .and_then(Value::as_str);
        let bias_detected = arguments.get("biasDetected")
            .and_then(Value::as_str);
        let assumptions_val = arguments.get("assumptions")
            .and_then(Value::as_array);
        let thought_number = arguments.get("thoughtNumber")
            .and_then(Value::as_u64)
            .unwrap_or(1) as usize;
        let is_final = arguments.get("nextThoughtNeeded")
            .and_then(Value::as_bool)
            .map(|v| !v)  // nextThoughtNeeded=false means this IS the final thought
            .unwrap_or(false);

        // ─── Import Cognitive Modules ────────────────────────────
        use crate::engine::budget::BudgetMode;
        use crate::engine::stage_engine::{CognitiveStage, StageSession, detect_stage};
        use crate::engine::quality_metrics;
        use crate::engine::ewma_reward::RewardSignals;
        use crate::engine::anti_hallucination;
        use crate::engine::bias_detector;
        use crate::engine::metacognition;
        use crate::engine::memory_bridge;

        use crate::engine::formatter;

        Self::emit_progress(&tx_prog, progress_token.clone(), 15.0, 100.0).await;

        // ─── R5: Budget Mode ─────────────────────────────────────
        let budget = BudgetMode::from_str_opt(budget_str);

        // ─── R1: Stage Engine ────────────────────────────────────
        let stage = if let Some(stage_str) = explicit_stage {
            match stage_str.to_uppercase().as_str() {
                "DEFINE" => CognitiveStage::Define,
                "RESEARCH" => CognitiveStage::Research,
                "ANALYZE" => CognitiveStage::Analyze,
                "HYPOTHESIZE" => CognitiveStage::Hypothesize,
                "VERIFY" => CognitiveStage::Verify,
                "SYNTHESIZE" => CognitiveStage::Synthesize,
                _ => detect_stage(hypothesis),
            }
        } else {
            detect_stage(hypothesis)
        };

        let mut session = StageSession::new();
        let new_assumptions: Vec<String> = assumptions_val
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();
        let stage_warnings = session.advance(Some(stage), hypothesis, &new_assumptions);

        Self::emit_progress(&tx_prog, progress_token.clone(), 30.0, 100.0).await;

        // ─── R2: Quality Metrics 6D ──────────────────────────────
        // Extract context keywords from hypothesis for relevance scoring
        let context_keywords: Vec<&str> = hypothesis
            .split_whitespace()
            .filter(|w| w.len() > 4)
            .take(10)
            .collect();
        let mut quality = quality_metrics::compute_quality(thought, &context_keywords);

        // ─── G3: Length-Proportional Quality Penalty ──────────────
        quality = quality_metrics::apply_length_penalty(quality, thought, budget);

        // ─── Phase 5A: Session-Persistent State ──────────────────
        // Instead of creating new trackers per call, reuse persistent session.
        use crate::engine::semantic_similarity;
        use crate::engine::contradiction_detector;
        use crate::engine::claim_grounding;
        use crate::engine::thought_session::TrendIndicator;

        let sessions = self.sessions.clone();
        let hypothesis_owned = hypothesis.to_string();
        let thought_owned = thought.to_string();

        // Access session and compute all semantic signals within the lock
        let (coherence, contradictions, info_gain, mut ewma_clone, mut trend, drift, _prev_thought_texts) =
            sessions.with_session(hypothesis, budget, |session| {
                // Get previous thoughts for semantic analysis (G6)
                let prev_texts: Vec<String> = session.previous_thoughts(3)
                    .iter()
                    .map(|s| s.to_string())
                    .collect();

                // Record this thought in session
                session.record_thought(&thought_owned);

                // S1/S3: Coherence against ACTUAL previous thought (not just hypothesis)
                let coh = if !prev_texts.is_empty() {
                    let last_prev = prev_texts.last().unwrap();
                    semantic_similarity::compute_coherence(&thought_owned, Some(last_prev))
                } else if hypothesis_owned != thought_owned {
                    semantic_similarity::compute_coherence(&thought_owned, Some(&hypothesis_owned))
                } else {
                    1.0
                };

                // S2: Contradictions against ALL previous thoughts (not just hypothesis)
                let prev_refs: Vec<&str> = prev_texts.iter().map(|s| s.as_str()).collect();
                let contras = contradiction_detector::detect_contradictions(&thought_owned, &prev_refs);

                // S4: Novelty from persistent tracker (accumulated vocabulary)
                let ig = session.novelty.track_novelty(&thought_owned);

                // Clone EWMA for use outside the lock
                let ewma_snap = session.ewma.clone();

                // G10: Current trend
                let tr = session.trend;

                // G11 + Vector A: Combined drift (hypothesis + root-anchoring)
                let dr = session.combined_drift(&hypothesis_owned, &thought_owned);

                (coh, contras, ig, ewma_snap, tr, dr, prev_texts)
            });

        // ─── S5: Claim Grounding (Phase 3) ───────────────────────
        let grounding_result = claim_grounding::analyze_grounding(thought);

        // S5 grounding feeds into F9 single EWMA update below

        // ─── N1: Sandbox Execution (before EWMA, for single-update) ──
        use crate::engine::sandbox::SandboxResult as SboxResult;
        use crate::engine::micro_prm;

        let sandbox_result: SboxResult = self.sandbox.execute(thought).await;
        let prm_verdict = micro_prm::evaluate_prm(&sandbox_result);
        let prm_score = prm_verdict.composite_score;

        Self::emit_progress(&tx_prog, progress_token.clone(), 50.0, 100.0).await;

        // F9: Single EWMA update with PRM-enriched signals (one lock acquisition)
        let final_grounding = if prm_score > 0.0 { prm_score } else { grounding_result.grounding };
        let final_signals = RewardSignals {
            quality: quality.weighted_mean(stage),
            faithfulness: grounding_result.faithfulness,
            coherence,
            contradiction_rate: contradictions.rate,
            info_gain,
            grounding: final_grounding,
        };
        sessions.with_session(hypothesis, budget, |session| {
            session.ewma.update(&final_signals);
            // NEW-1: Record confidence for oscillation detection
            session.record_confidence(confidence);
            trend = TrendIndicator::from_ewma(&session.ewma);
            ewma_clone = session.ewma.clone();
        });
        let mut ewma = ewma_clone;

        // Format sandbox output
        let has_code = sandbox_result.ast_analysis.assert_count > 0 || sandbox_result.success;
        let is_sandbox = has_code && sandbox_result.execution_ms > 0;
        let sandbox_output_text = if is_sandbox {
            let mut out = format!("PRM: {:.0}% — {}\n", prm_score * 100.0, prm_verdict.verdict);
            for exp in &prm_verdict.explanations {
                out.push_str(&format!("  {}\n", exp));
            }
            if !sandbox_result.stdout.is_empty() {
                out.push_str(&format!("📤 stdout: {}\n", sandbox_result.stdout.trim()));
            }
            if let Some(ref err) = sandbox_result.error {
                out.push_str(&format!("❌ Error: {}\n", err));
            }
            out
        } else {
            String::new()
        };
        let sandbox_text = if is_sandbox { Some(sandbox_output_text.as_str()) } else { None };

        // ─── Phase 5D: Confidence Calibration (G4) ──────────────
        // Bayesian adjustment: blend agent's declared confidence with PRM evidence.
        // Prevents overconfident reasoning and underconfident dismissals.
        let calibrated_confidence = if prm_score > 0.0 {
            // Weighted blend: 70% PRM evidence, 30% declared (evidence dominates)
            let cal = prm_score * 0.7 + confidence * 0.3;
            cal.clamp(0.0, 1.0)
        } else {
            // No PRM evidence: keep declared but report uncertainty
            confidence
        };
        let confidence_delta = calibrated_confidence - confidence;

        Self::emit_progress(&tx_prog, progress_token.clone(), 65.0, 100.0).await;

        // ─── R4+R10: Anti-Hallucination ──────────────────────────
        let verdict = anti_hallucination::verify_thought(
            thought, &session, &quality, &mut ewma, calibrated_confidence, thought_number,
        );

        // ─── R7: Bias Detection (F14: pass real session history) ─
        let prev_bias_refs: Vec<&str> = _prev_thought_texts.iter().map(|s| s.as_str()).collect();
        let biases = bias_detector::detect_biases(
            thought, thought_number, &prev_bias_refs, bias_detected,
        );

        // ─── R8: Metacognitive Analysis ──────────────────────────
        let is_verify_or_synth = matches!(
            stage,
            CognitiveStage::Verify | CognitiveStage::Synthesize
        );
        let metacog = metacognition::analyze_metacognition(thought, is_verify_or_synth);

        Self::emit_progress(&tx_prog, progress_token.clone(), 80.0, 100.0).await;

        // ─── Phase 5B: Stage-Content Alignment (G8) ─────────────
        use crate::engine::stage_validator;
        let stage_alignment = stage_validator::validate_stage_alignment(thought, stage);

        // ─── G7: Logical Step Validity (ReasonEval 2024) ─────────
        let (logical_validity, logical_warning) = stage_validator::validate_logical_validity(
            thought, thought_number,
        );

        // ─── G8: Reasoning Type Classification (Walton 2006) ─────
        let reasoning_type = metacognition::classify_reasoning_type(thought);

        // ─── Phase 5B: Corrective Directives (G2, G7, G9) ───────
        use crate::engine::corrective_directives;
        let claim_count = verdict.layers.claim_count;
        let is_code = crate::engine::shared_utils::is_code_input(thought);
        let mut directives = corrective_directives::generate_directives(
            &quality, &verdict, &metacog, claim_count, is_code,
        );

        // ─── G5: Reflexion Self-Evaluation (Shinn 2023) ──────────
        if let Some(reflexion) = corrective_directives::generate_reflexion_directive(
            verdict.trust_score, thought_number,
        ) {
            directives.push(reflexion);
        }

        // NEW-1: Confidence oscillation detection
        let is_oscillating = sessions.with_session(hypothesis, budget, |session| {
            session.is_confidence_oscillating()
        });
        if is_oscillating {
            directives.push(corrective_directives::generate_oscillation_directive());
        }

        // Vector 4: Kinematic collapse prediction (v_t + a_t)
        let is_collapsing = ewma.is_collapsing_kinematically();
        if is_collapsing {
            directives.push(corrective_directives::Directive {
                severity: corrective_directives::Severity::Warning,
                dimension: "Kinematic",
                instruction: "📉 KINEMATIC COLLAPSE: Your reasoning quality is \
                    falling with increasing acceleration. The current trajectory \
                    predicts rejection within 1-2 steps. STOP and re-anchor: \
                    (1) What was your strongest validated conclusion? \
                    (2) Build from there, discarding speculative branches."
                    .to_string(),
            });
        }

        // ─── R9: Graph-of-Thought (from persistent session) ──────
        let topology = sessions.with_session(hypothesis, budget, |session| {
            session.graph.topology_summary()
        });

        // ─── R11: Memory Bridge ──────────────────────────────────
        let memory_instructions = memory_bridge::generate_memory_instructions(
            stage, thought_number, is_final, thought,
        );

        Self::emit_progress(&tx_prog, progress_token.clone(), 90.0, 100.0).await;

        // ─── R12: Format Output ──────────────────────────────────
        let formatted = formatter::format_engine_output(
            stage, &session, &quality, &ewma, &verdict, &metacog,
            &biases, &memory_instructions, Some(&topology),
            thought_number, is_sandbox, sandbox_text, budget,
        );

        // ─── Phase 5B: Append directives + trend + drift ─────────
        let mut final_output = formatted;

        // Trend indicator (G10) — F22: English
        if trend != crate::engine::thought_session::TrendIndicator::Insufficient {
            final_output.push_str(&format!("\n📈 Trend: {} {}", trend.emoji(), trend.label()));
        }

        // Hypothesis drift warning (G11) — F22: English
        if drift > 0.5 {
            final_output.push_str(&format!(
                "\n⚠️ Drift: {:.0}% — Reasoning is diverging from original hypothesis.",
                drift * 100.0
            ));
        }

        // Stage alignment warning
        if let Some(ref alignment_warning) = stage_alignment.warning {
            final_output.push_str(&format!("\n{}", alignment_warning));
        }

        // G7: Logical validity warning
        if let Some(ref lw) = logical_warning {
            final_output.push_str(&format!("\n{}", lw));
        }

        // G8: Reasoning type indicator
        final_output.push_str(&format!("\n🧩 Reasoning: {}", reasoning_type.label()));

        // Corrective directives (G2/G7/G9/G12 — adaptive: suppress when EWMA > 70%)
        if ewma.percentage() < 70.0 || corrective_directives::has_mandatory_corrections(&directives) {
            let directives_text = corrective_directives::format_directives(&directives);
            if !directives_text.is_empty() {
                final_output.push_str(&format!("\n{}", directives_text));
            }
        }

        // Confidence calibration delta (G4) — F22: English
        if confidence_delta.abs() > 0.1 {
            final_output.push_str(&format!(
                "\n🎯 Calibrated confidence: {:.0}% → {:.0}% (delta: {:+.0}%)",
                confidence * 100.0, calibrated_confidence * 100.0, confidence_delta * 100.0
            ));
        }

        // Stage warnings
        for w in &stage_warnings {
            final_output.push_str(&format!("\n{}", w));
        }

        Self::emit_progress(&tx_prog, progress_token, 100.0, 100.0).await;

        tracing::info!(
            "Cognitive Engine v3.0 | Stage: {:?} | EWMA: {:.1}% | Trust: {:.1}% | Budget: {:?}",
            stage, ewma.percentage(), verdict.trust_score * 100.0, budget
        );

        // ─── Build Response ──────────────────────────────────────
        let is_error = verdict.should_reject;
        let session_thought_count = sessions.with_session(hypothesis, budget, |s| s.thought_count());
        let response = serde_json::json!({
            "content": [{
                "type": "text",
                "text": final_output
            }],
            "isError": is_error,
            "_meta": {
                "stage": stage,
                "ewma": ewma.percentage(),
                "trustScore": verdict.trust_score,
                "qualityMean": quality.raw_mean(),
                "budget": budget,
                "biasCount": biases.len(),
                "shouldEarlyStop": verdict.should_early_stop,
                "prmScore": prm_verdict.composite_score,
                "trend": trend.label(),
                "hypothesisDrift": drift,
                "sessionThoughts": session_thought_count,
                "logicalValidity": logical_validity,
                "reasoningType": reasoning_type.label(),
                "chainScore": ewma.chain_score(),
            }
        });

        Ok(Some(response))
    }

    async fn handle_stress_benchmark_tool(&self, params: &ToolsCallParams, tx: tokio::sync::mpsc::Sender<OutgoingMessage>) -> Result<Option<Value>, RpcError> {
        let progress_token = params._meta.as_ref().and_then(|m| m.progress_token.clone());
        let tx_prog = tx.clone();

        Self::emit_progress(&tx_prog, progress_token.clone(), 5.0, 100.0).await;
        tracing::info!("Starting MASSIVE STRESS TEST. 5000 Parallel asynchronous requests...");
        let start = Instant::now();
        let mut handles = vec![];
        
        Self::emit_progress(&tx_prog, progress_token.clone(), 20.0, 100.0).await;
        
        for _i in 0..5000 {
            let handle = tokio::spawn(async move {
                tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
                format!("{{\"response\": \"{}\"}}", "A".repeat(100))
            });
            handles.push(handle);
        }
        
        Self::emit_progress(&tx_prog, progress_token.clone(), 80.0, 100.0).await;
        
        let _results = futures::future::join_all(handles).await;
        let elapsed = start.elapsed();
        let final_msg = format!("Stress Test Complete: 5000 branches processed in {:.2?}s.", elapsed);
        tracing::info!("{}", final_msg);
        
        Self::emit_progress(&tx_prog, progress_token, 100.0, 100.0).await;
        
        Ok(Some(serde_json::json!({
            "content": [{
                "type": "text",
                "text": final_msg
            }]
        })))
    }

    // ─── Phase 4: verify_code Tool ──────────────────────────────────
    async fn handle_verify_code_tool(&self, params: &ToolsCallParams) -> Result<Option<Value>, RpcError> {
        let arguments = params.arguments.clone().unwrap_or(serde_json::json!({}));
        let code = arguments.get("code")
            .and_then(|v: &Value| v.as_str())
            .unwrap_or("");

        if code.is_empty() {
            return Err(RpcError {
                code: -32602,
                message: "Missing required parameter: code".to_string(),
                data: None,
            });
        }

        // Execute in sandbox
        use crate::engine::sandbox::SandboxResult as SboxResult;
        use crate::engine::micro_prm;

        let sandbox_result: SboxResult = self.sandbox.execute(code).await;
        let prm_verdict = micro_prm::evaluate_prm(&sandbox_result);

        // Format output — F22: English
        let mut output = "🔍 **Code Verification**\n\n".to_string();
        output.push_str(&format!("PRM: {:.0}% — {}\n", prm_verdict.composite_score * 100.0, prm_verdict.verdict));
        for exp in &prm_verdict.explanations {
            output.push_str(&format!("  {}\n", exp));
        }

        if !sandbox_result.stdout.is_empty() {
            output.push_str(&format!("\n📤 stdout:\n```\n{}\n```\n", sandbox_result.stdout.trim()));
        }
        if let Some(ref err) = sandbox_result.error {
            output.push_str(&format!("\n❌ Error: {}\n", err));
        }
        output.push_str(&format!("\n⏱️ Execution: {}ms | Complexity: CC={}\n",
            sandbox_result.execution_ms, sandbox_result.ast_analysis.cyclomatic_complexity));

        Ok(Some(serde_json::json!({
            "content": [{
                "type": "text",
                "text": output
            }],
            "_meta": {
                "prm_score": prm_verdict.composite_score,
                "execution_ms": sandbox_result.execution_ms,
                "success": sandbox_result.success,
                "assert_count": sandbox_result.ast_analysis.assert_count,
                "complexity": sandbox_result.ast_analysis.cyclomatic_complexity,
            }
        })))
    }

    // ─── Phase 4: analyze_reasoning Tool ────────────────────────────
    async fn handle_analyze_reasoning_tool(&self, params: &ToolsCallParams) -> Result<Option<Value>, RpcError> {
        use crate::engine::semantic_similarity;
        use crate::engine::contradiction_detector;
        use crate::engine::novelty_tracker::NoveltyTracker;
        use crate::engine::claim_grounding;

        let arguments = params.arguments.clone().unwrap_or(serde_json::json!({}));

        let thoughts: Vec<String> = arguments.get("thoughts")
            .and_then(|v: &Value| v.as_array())
            .map(|arr: &Vec<Value>| arr.iter().filter_map(|v: &Value| v.as_str().map(String::from)).collect())
            .unwrap_or_default();

        let context = arguments.get("context")
            .and_then(|v: &Value| v.as_str())
            .unwrap_or("");

        if thoughts.is_empty() {
            return Err(RpcError {
                code: -32602,
                message: "Missing required parameter: thoughts (non-empty array)".to_string(),
                data: None,
            });
        }

        let mut output = format!("📊 **Reasoning Chain Analysis** ({} steps)\n\n", thoughts.len());
        let mut novelty_tracker = NoveltyTracker::new();
        let mut all_coherence = Vec::new();
        let mut all_novelty = Vec::new();
        let mut all_grounding = Vec::new();
        let mut total_contradictions = 0usize;

        // Seed novelty with context if provided
        if !context.is_empty() {
            novelty_tracker.track_novelty(context);
            output.push_str(&format!("🎯 Context: \"{}\"\n\n", crate::engine::shared_utils::truncate_str(context, 80)));
        }

        for (i, thought) in thoughts.iter().enumerate() {
            let step = i + 1;
            let prev = if i > 0 { Some(thoughts[i - 1].as_str()) } else { None };

            // S1/S3: Coherence
            let coherence = semantic_similarity::compute_coherence(thought, prev);
            all_coherence.push(coherence);

            // S2: Contradictions
            let prev_claims: Vec<&str> = thoughts[..i].iter().map(|s: &String| s.as_str()).collect();
            let contra = contradiction_detector::detect_contradictions(thought, &prev_claims);
            total_contradictions += contra.contradictions.len();

            // S4: Novelty
            let novelty = novelty_tracker.track_novelty(thought);
            all_novelty.push(novelty);

            // S5: Grounding
            let grounding = claim_grounding::analyze_grounding(thought);
            all_grounding.push(grounding.grounding);

            // Per-step report
            let coherence_icon = if coherence > 0.5 { "✅" } else if coherence > 0.2 { "⚠️" } else { "🔴" };
            let novelty_icon = if novelty > 0.3 { "✅" } else if novelty > 0.1 { "⚠️" } else { "🔄" };

            output.push_str(&format!(
                "**Step {}**: Coherence {} {:.0}% | Novelty {} {:.0}% | Grounding {:.0}%",
                step, coherence_icon, coherence * 100.0,
                novelty_icon, novelty * 100.0,
                grounding.grounding * 100.0
            ));
            if !contra.contradictions.is_empty() {
                output.push_str(&format!(" | ⚡ {} contradictions", contra.contradictions.len()));
            }
            output.push('\n');
        }

        // Aggregate metrics
        let avg_coherence = all_coherence.iter().sum::<f64>() / all_coherence.len().max(1) as f64;
        let avg_novelty = all_novelty.iter().sum::<f64>() / all_novelty.len().max(1) as f64;
        let avg_grounding = all_grounding.iter().sum::<f64>() / all_grounding.len().max(1) as f64;
        let novelty_decay = if all_novelty.len() >= 2 {
            all_novelty.last().unwrap_or(&0.0) - all_novelty.first().unwrap_or(&0.0)
        } else {
            0.0
        };

        output.push_str("\n─── Summary ───\n");
        output.push_str(&format!("📈 Avg coherence: {:.0}%\n", avg_coherence * 100.0));
        output.push_str(&format!("💡 Avg novelty: {:.0}%\n", avg_novelty * 100.0));
        output.push_str(&format!("📉 Novelty decay: {:.0}% ({})\n",
            novelty_decay * 100.0,
            if novelty_decay < -0.3 { "⚠️ high repetition" } else { "normal" }
        ));
        output.push_str(&format!("🔗 Avg grounding: {:.0}%\n", avg_grounding * 100.0));
        output.push_str(&format!("⚡ Total contradictions: {}\n", total_contradictions));

        // Overall verdict
        let chain_quality = (avg_coherence * 0.3 + avg_novelty * 0.2 + avg_grounding * 0.3
            + (1.0 - total_contradictions as f64 / thoughts.len().max(1) as f64).max(0.0) * 0.2)
            .clamp(0.0, 1.0);

        let verdict = if chain_quality > 0.7 {
            "🟢 SOLID — Coherent, informative and well-grounded chain"
        } else if chain_quality > 0.4 {
            "🟡 ACCEPTABLE — Chain with areas for improvement"
        } else {
            "🔴 WEAK — Significant coherence or grounding gaps detected"
        };
        output.push_str(&format!("\n🏆 Overall quality: {:.0}% — {}\n", chain_quality * 100.0, verdict));

        Ok(Some(serde_json::json!({
            "content": [{
                "type": "text",
                "text": output
            }],
            "_meta": {
                "chain_quality": chain_quality,
                "avg_coherence": avg_coherence,
                "avg_novelty": avg_novelty,
                "avg_grounding": avg_grounding,
                "total_contradictions": total_contradictions,
                "novelty_decay": novelty_decay,
                "steps_analyzed": thoughts.len(),
            }
        })))
    }
}

// truncate_str moved to shared_utils::truncate_str (F10: UTF-8 safe)
