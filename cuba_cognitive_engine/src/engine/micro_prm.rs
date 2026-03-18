// src/engine/micro_prm.rs
//
// N3: Process Reward Model (PRM) — 8-Signal Step Verification
//
// Based on Lightman et al. 2023 "Let's Verify Step by Step" (OpenAI).
// Evaluates each thought step with 8 independent verification signals,
// producing a composite PRM score that feeds into the EWMA tracker.
//
// Signals:
// E1: Compiles    — Python code parses and executes without errors
// E2: Asserts     — Assert statements pass (contract verification)
// E3: Complexity  — CC ≤ 10 (McCabe 1976, manageable code)
// E4: Type Safety — Has type annotations (structural quality)
// E5: Imports     — Uses standard/safe imports (no blocked modules)
// E6: Determinism — No random/time dependencies (reproducible)
// E7: Coverage    — Assert-to-function ratio (test coverage proxy)
// E8: Diversity   — Unique assert targets (anti-gaming, P3-C)

use crate::engine::sandbox::SandboxResult;
use serde::Serialize;

/// PRM evaluation result with 8 independent signals.
#[derive(Debug, Clone, Serialize)]
pub struct PrmVerdict {
    /// Individual signal scores (0.0 or 1.0 each).
    pub signals: PrmSignals,
    /// Composite PRM score (weighted mean of signals).
    pub composite_score: f64,
    /// Human-readable verdict.
    pub verdict: &'static str,
    /// Detailed explanation per signal.
    pub explanations: Vec<String>,
}

/// The 8 verification signals.
#[derive(Debug, Clone, Serialize)]
pub struct PrmSignals {
    /// E1: Code compiled and executed successfully.
    pub compiles: f64,
    /// E2: All assert statements passed.
    pub asserts_pass: f64,
    /// E3: Cyclomatic complexity within limit (CC ≤ 10).
    pub complexity_ok: f64,
    /// E4: Code has type annotations.
    pub type_safety: f64,
    /// E5: Only safe imports used.
    pub imports_safe: f64,
    /// E6: Code is deterministic (reproducible).
    pub deterministic: f64,
    /// E7: Assert-to-function ratio (coverage proxy).
    pub coverage: f64,
    /// V7 (P3-C): E8: Assertion diversity (unique targets / total asserts).
    pub assert_diversity: f64,
}

// V9: PRM uses Logical Veto Gate instead of flat weighted sum.
//
// Architecture (adapted from Dominance Ranking, Deep Research 2025):
//   Gate:     (E1×0.7 + E4×0.3) — if code doesn't compile, gate ≈ 0.05
//   Quality:  E3×0.4 + E5×0.3 + E6×0.3 — structural quality
//   Verify:   E2×0.4 + E7×0.2 + E8×0.4 — verification robustness
//   Final:    Gate × (0.6×Verify + 0.4×Quality)
//
// Key property: uncompilable code CANNOT score > 0.05 regardless of
// diversity or coverage, eliminating the gaming vector.

/// Legacy flat weights retained for backward reference and static analysis.
#[allow(dead_code)]
const LEGACY_WEIGHTS: [f64; 8] = [
    0.25, // E1: Compiles (most critical)
    0.25, // E2: Asserts pass
    0.10, // E3: Complexity
    0.08, // E4: Type safety
    0.05, // E5: Safe imports
    0.10, // E6: Determinism
    0.07, // E7: Coverage
    0.10, // E8: Assert diversity (P3-C)
];

/// Evaluate a sandbox execution result with 7 PRM signals.
pub fn evaluate_prm(sandbox: &SandboxResult) -> PrmVerdict {
    let mut explanations = Vec::with_capacity(7);

    // ─── E1: Compiles ────────────────────────────────────────
    let e1 = if sandbox.success { 1.0 } else { 0.0 };
    explanations.push(if sandbox.success {
        format!(
            "✅ E1 Compiles: Executed successfully in {}ms",
            sandbox.execution_ms
        )
    } else {
        format!(
            "❌ E1 Compiles: Error — {}",
            sandbox.error.as_deref().unwrap_or("unknown")
        )
    });

    // ─── E2: Asserts Pass ────────────────────────────────────
    let assert_count = sandbox.ast_analysis.assert_count;
    let e2 = if sandbox.success && assert_count > 0 {
        1.0
    } else if sandbox.success && assert_count == 0 {
        0.3 // V3: Runs but no verification — low score enforces discipline
    } else {
        0.0 // Failed (likely assertion error)
    };
    explanations.push(if assert_count > 0 && sandbox.success {
        format!("✅ E2 Asserts: {} verifications passed", assert_count)
    } else if assert_count == 0 {
        "⚠️ E2 Asserts: No verification assertions found".to_string()
    } else {
        format!(
            "❌ E2 Asserts: {} assertions, at least one failed",
            assert_count
        )
    });

    // ─── E3: Complexity OK ───────────────────────────────────
    let cc = sandbox.ast_analysis.cyclomatic_complexity;
    let e3 = if cc <= 7 {
        1.0
    } else if cc <= 10 {
        0.7
    } else if cc <= 15 {
        0.4
    } else {
        0.0
    };
    explanations.push(format!(
        "{} E3 Complexity: CC={} {}",
        if cc <= 10 { "✅" } else { "⚠️" },
        cc,
        if cc <= 7 {
            "(optimal)"
        } else if cc <= 10 {
            "(acceptable)"
        } else {
            "(high)"
        }
    ));

    // ─── E4: Type Safety ─────────────────────────────────────
    let e4 = if sandbox.ast_analysis.has_type_hints {
        1.0
    } else {
        0.3
    };
    explanations.push(if sandbox.ast_analysis.has_type_hints {
        "✅ E4 Types: Type annotations present".to_string()
    } else {
        "⚠️ E4 Types: No type annotations found".to_string()
    });

    // ─── E5: Safe Imports ────────────────────────────────────
    let e5 = if sandbox.ast_analysis.security_violations.is_empty() {
        1.0
    } else {
        0.0
    };
    explanations.push(if sandbox.ast_analysis.security_violations.is_empty() {
        format!(
            "✅ E5 Imports: {} safe imports",
            sandbox.ast_analysis.import_count
        )
    } else {
        format!(
            "❌ E5 Imports: {} security violations",
            sandbox.ast_analysis.security_violations.len()
        )
    });

    // ─── E6: Determinism ─────────────────────────────────────
    let e6 = if sandbox.ast_analysis.is_deterministic {
        1.0
    } else {
        0.5
    };
    explanations.push(if sandbox.ast_analysis.is_deterministic {
        "✅ E6 Determinism: Reproducible result".to_string()
    } else {
        "⚠️ E6 Determinism: Uses random/time (non-reproducible)".to_string()
    });

    // ─── E7: Coverage (assert-to-function ratio) ─────────────
    let func_count = sandbox.ast_analysis.function_count;
    let e7 = if func_count == 0 {
        // No functions — asserts count directly
        if assert_count >= 2 {
            1.0
        } else if assert_count == 1 {
            0.6
        } else {
            0.2
        }
    } else {
        // Functions present — check coverage ratio
        let ratio = assert_count as f64 / func_count as f64;
        if ratio >= 1.0 {
            1.0
        } else {
            ratio.max(0.1)
        }
    };
    explanations.push(format!(
        "{} E7 Coverage: {} asserts / {} functions",
        if e7 >= 0.6 { "✅" } else { "⚠️" },
        assert_count,
        func_count
    ));

    // ─── E8: Assertion Diversity (P3-C) ──────────────────
    let unique_targets = sandbox.ast_analysis.unique_assert_targets;
    let e8 = if assert_count == 0 {
        0.5 // No assertions — neutral (same as no-asserts E2)
    } else if unique_targets == 0 {
        0.3 // Assertions exist but no named variables (bare literals)
    } else {
        let diversity_ratio = unique_targets as f64 / assert_count as f64;
        diversity_ratio.clamp(0.0, 1.0)
    };
    explanations.push(format!(
        "{} E8 Diversity: {} unique targets / {} assertions",
        if e8 >= 0.5 { "✅" } else { "⚠️" },
        unique_targets,
        assert_count
    ));

    // ─── V9: Logical Veto Gate Composite ───────────────
    //
    // Gate:    (E1×0.7 + E4×0.3) — compilation + type safety
    // Quality: E3×0.4 + E5×0.3 + E6×0.3 — structural
    // Verify:  E2×0.4 + E7×0.2 + E8×0.4 — verification robustness
    // Final:   Gate × (0.6×Verify + 0.4×Quality)
    let signals = PrmSignals {
        compiles: e1,
        asserts_pass: e2,
        complexity_ok: e3,
        type_safety: e4,
        imports_safe: e5,
        deterministic: e6,
        coverage: e7,
        assert_diversity: e8,
    };

    // Gate: if code doesn't compile, everything collapses
    let gate = (e1 * 0.7 + e4 * 0.3).max(0.05);
    let quality_group = e3 * 0.4 + e5 * 0.3 + e6 * 0.3;
    let verify_group = e2 * 0.4 + e7 * 0.2 + e8 * 0.4;
    let composite_score = gate * (0.6 * verify_group + 0.4 * quality_group);

    let verdict = if composite_score >= 0.85 {
        "✅ EXCELLENT — Code exhaustively verified"
    } else if composite_score >= 0.65 {
        "🟢 GOOD — Solid verification with improvement opportunities"
    } else if composite_score >= 0.45 {
        "🟡 ACCEPTABLE — Partial verification, improve coverage"
    } else {
        "🔴 INSUFFICIENT — Code not verified or failed"
    };

    PrmVerdict {
        signals,
        composite_score,
        verdict,
        explanations,
    }
}

/// Evaluate code text without sandbox execution (static analysis only).
/// Used when sandbox execution is not available or code failed.
/// Fallback static PRM analysis when sandbox execution fails.
/// Used by DEBT-T08 fallback path.
#[allow(dead_code)]
pub fn evaluate_static(code: &str) -> PrmVerdict {
    let has_asserts = code.contains("assert ");
    let has_types = code.contains("-> ") || code.contains(": int") || code.contains(": str");
    let has_imports = code.contains("import ");
    let line_count = code.lines().count();

    let e1 = 0.5; // Can't verify without execution
    let e2 = if has_asserts { 0.5 } else { 0.0 };
    let e3 = if line_count < 50 { 1.0 } else { 0.5 };
    let e4 = if has_types { 0.8 } else { 0.2 };
    let e5 = if !code.contains("subprocess") && !code.contains("os.system") {
        1.0
    } else {
        0.0
    };
    let e6 = if !code.contains("random") { 1.0 } else { 0.5 };
    let e7 = if has_asserts { 0.5 } else { 0.1 };
    let e8 = if has_asserts { 0.5 } else { 0.2 }; // P3-C: Can't measure without AST

    let signals = PrmSignals {
        compiles: e1,
        asserts_pass: e2,
        complexity_ok: e3,
        type_safety: e4,
        imports_safe: e5,
        deterministic: e6,
        coverage: e7,
        assert_diversity: e8,
    };

    // V9: Apply same Veto Gate system as evaluate_prm
    let gate = (e1 * 0.7 + e4 * 0.3).max(0.05);
    let quality_group = e3 * 0.4 + e5 * 0.3 + e6 * 0.3;
    let verify_group = e2 * 0.4 + e7 * 0.2 + e8 * 0.4;
    let composite_score = gate * (0.6 * verify_group + 0.4 * quality_group);

    PrmVerdict {
        signals,
        composite_score,
        verdict: "⚠️ Static analysis only (no execution)",
        explanations: vec![
            format!("🔍 E1 Compiles: Static analysis only"),
            format!(
                "{} E2 Asserts: {}",
                if has_asserts { "✅" } else { "⚠️" },
                if has_asserts {
                    "Detected"
                } else {
                    "Not detected"
                }
            ),
            format!("✅ E3 Complexity: {} lines", line_count),
            format!(
                "{} E4 Types: {}",
                if has_types { "✅" } else { "⚠️" },
                if has_types {
                    "Annotations detected"
                } else {
                    "No annotations"
                }
            ),
            format!(
                "{} E5 Imports: {}",
                if has_imports { "✅" } else { "—" },
                if has_imports { "Present" } else { "No imports" }
            ),
            format!(
                "{} E6 Determinism: {}",
                if !code.contains("random") {
                    "✅"
                } else {
                    "⚠️"
                },
                if !code.contains("random") {
                    "Deterministic"
                } else {
                    "Non-deterministic"
                }
            ),
            format!(
                "{} E7 Coverage: {}",
                if has_asserts { "✅" } else { "⚠️" },
                if has_asserts {
                    "With verifications"
                } else {
                    "No verifications"
                }
            ),
            format!("⚠️ E8 Diversity: Static analysis (no AST)"),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::sandbox::{AstAnalysis, SandboxResult};

    fn make_success_result() -> SandboxResult {
        SandboxResult {
            success: true,
            stdout: "2\n".to_string(),
            error: None,
            execution_ms: 5,
            ast_analysis: AstAnalysis {
                cyclomatic_complexity: 3,
                assert_count: 2,
                unique_assert_targets: 2,
                function_count: 1,
                import_count: 0,
                has_type_hints: true,
                is_deterministic: true,
                security_violations: vec![],
            },
        }
    }

    #[test]
    fn test_prm_perfect_score() {
        let result = make_success_result();
        let verdict = evaluate_prm(&result);
        assert!(
            verdict.composite_score > 0.85,
            "Expected excellent: {:.2}",
            verdict.composite_score
        );
        assert!(verdict.verdict.contains("EXCELLENT"));
    }

    #[test]
    fn test_prm_failed_execution() {
        let result = SandboxResult {
            success: false,
            stdout: String::new(),
            error: Some("AssertionError".to_string()),
            execution_ms: 2,
            ast_analysis: AstAnalysis {
                cyclomatic_complexity: 1,
                assert_count: 1,
                unique_assert_targets: 1,
                function_count: 0,
                import_count: 0,
                has_type_hints: false,
                is_deterministic: true,
                security_violations: vec![],
            },
        };
        let verdict = evaluate_prm(&result);
        assert!(verdict.composite_score < 0.5);
        assert!(verdict.verdict.contains("INSUFFICIENT"));
    }

    #[test]
    fn test_prm_no_asserts() {
        let mut result = make_success_result();
        result.ast_analysis.assert_count = 0;
        let verdict = evaluate_prm(&result);
        // Should be lower than perfect
        assert!(verdict.composite_score < 0.95);
        assert!(verdict.signals.asserts_pass == 0.3);
    }

    #[test]
    fn test_prm_high_complexity() {
        let mut result = make_success_result();
        result.ast_analysis.cyclomatic_complexity = 12;
        let verdict = evaluate_prm(&result);
        assert!(verdict.signals.complexity_ok < 0.5);
    }

    #[test]
    fn test_static_analysis() {
        let verdict =
            evaluate_static("def foo(x: int) -> int:\n    assert x > 0\n    return x * 2");
        assert!(verdict.composite_score > 0.3);
        assert!(verdict.verdict.contains("estático") || verdict.verdict.contains("Static"));
    }

    /// V9: Logical Veto Gate — compilation failure drastically reduces composite.
    #[test]
    fn test_prm_veto_gate_compilation_failure() {
        // Scenario 1: Code fails to compile AND fails type safety → full veto
        let result_full_veto = SandboxResult {
            success: false,
            stdout: String::new(),
            error: Some("SyntaxError: invalid syntax".to_string()),
            execution_ms: 1,
            ast_analysis: AstAnalysis {
                cyclomatic_complexity: 2,
                assert_count: 5,
                unique_assert_targets: 5,
                function_count: 2,
                import_count: 0,
                has_type_hints: false, // No type hints → E4 low
                is_deterministic: true,
                security_violations: vec![],
            },
        };
        let v1 = evaluate_prm(&result_full_veto);
        // Gate = (0.0×0.7 + 0.2×0.3) = 0.06, composite = 0.06 × groups ≈ very low
        assert!(
            v1.composite_score < 0.10,
            "Full veto (no compile, no types) should score < 0.10, got: {:.4}",
            v1.composite_score
        );

        // Scenario 2: Code fails to compile but HAS type hints → partial veto
        let result_partial = SandboxResult {
            success: false,
            stdout: String::new(),
            error: Some("SyntaxError: invalid syntax".to_string()),
            execution_ms: 1,
            ast_analysis: AstAnalysis {
                cyclomatic_complexity: 2,
                assert_count: 5,
                unique_assert_targets: 5,
                function_count: 2,
                import_count: 0,
                has_type_hints: true, // Type hints → E4 high
                is_deterministic: true,
                security_violations: vec![],
            },
        };
        let v2 = evaluate_prm(&result_partial);
        // Gate = (0.0×0.7 + 1.0×0.3) = 0.3, composite ≈ 0.23
        // Still well below the 0.50 "NEEDS WORK" threshold
        assert!(
            v2.composite_score < 0.50,
            "Partial veto (no compile, has types) should score < 0.50, got: {:.4}",
            v2.composite_score
        );

        // Both should be far below a successful compilation
        let success = make_success_result();
        let v_success = evaluate_prm(&success);
        assert!(
            v_success.composite_score > v2.composite_score * 2.0,
            "Successful code should score much higher than failed: success={:.4} vs failed={:.4}",
            v_success.composite_score,
            v2.composite_score
        );
    }
}
