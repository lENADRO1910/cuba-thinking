// src/engine/sandbox.rs
//
// N1: PyO3 Execution Sandbox — Program-Aided Language (PAL)
//
// Real Python code execution via embedded PyO3 interpreter.
// Based on Gao et al. 2023 (ICML) "Program-Aided Language Models".
//
// Security model (DEBT-T01/T02/T04 resolved):
// - Timeout: 5 seconds max via tokio::time::timeout (DEBT-T01)
// - AST-based security scan, not string matching (DEBT-T02)
// - Recursion limit + memory guard via sys.setrecursionlimit (DEBT-T04)
// - No network access (no socket, no requests, no urllib)
// - No filesystem writes (no open("w"), no os.remove)
// - No subprocess/os.system/exec/eval
// - Semaphore: max 2 concurrent executions
//
// Allowed:
// - ast module (parsing, analysis)
// - math, statistics, decimal, fractions
// - z3 (SMT solver) — if installed
// - sympy — if installed
// - json, re, collections, itertools, functools
// - assert statements (primary verification mechanism)

use anyhow::Result;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Semaphore;
use tracing::{debug, info, warn};

/// Maximum execution time for sandbox code.
const SANDBOX_TIMEOUT: Duration = Duration::from_secs(5);

/// Maximum Python recursion depth (prevents stack overflow).
const PYTHON_RECURSION_LIMIT: usize = 100;

/// Maximum stdout capture size in bytes.
const MAX_STDOUT_BYTES: usize = 8192;

/// FIX-1: Maximum virtual memory for sandboxed Python execution (512 MB).
/// Prevents OOM bombs (e.g. `x = [0] * 10**10`) from crashing the process.
/// 512 MB instead of 256 MB because PyO3 shares the Rust process's
/// virtual address space — the combined footprint needs headroom.
/// Applied via `resource.setrlimit(RLIMIT_AS, ...)` in the setup script.
const SANDBOX_MEMORY_LIMIT: usize = 512 * 1024 * 1024;

/// Execution result from the Python sandbox.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxResult {
    /// Whether the code executed successfully (no exceptions).
    pub success: bool,
    /// Captured stdout output (truncated to MAX_STDOUT_BYTES).
    pub stdout: String,
    /// Error message if execution failed.
    pub error: Option<String>,
    /// Execution time in milliseconds.
    pub execution_ms: u64,
    /// AST analysis results.
    pub ast_analysis: AstAnalysis,
}

/// Static AST analysis performed before execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AstAnalysis {
    /// Cyclomatic complexity (McCabe 1976).
    pub cyclomatic_complexity: usize,
    /// Number of assert statements found.
    pub assert_count: usize,
    /// Number of function definitions.
    pub function_count: usize,
    /// Number of import statements.
    pub import_count: usize,
    /// Whether code contains type hints.
    pub has_type_hints: bool,
    /// Whether code is deterministic (no random, no time).
    pub is_deterministic: bool,
    /// Security violations found (blocked imports/calls).
    pub security_violations: Vec<String>,
}

impl AstAnalysis {
    /// Default empty analysis for error cases.
    fn empty() -> Self {
        Self {
            cyclomatic_complexity: 0,
            assert_count: 0,
            function_count: 0,
            import_count: 0,
            has_type_hints: false,
            is_deterministic: true,
            security_violations: vec![],
        }
    }

    fn with_violations(violations: Vec<String>) -> Self {
        Self {
            security_violations: violations,
            ..Self::empty()
        }
    }
}

/// Native Rust Local Reasoning Engine with real Python execution.
///
/// Uses PyO3 embedded interpreter with security restrictions
/// to execute and validate code submitted by the AI.
pub struct LocalReasoningEngine {
    /// Concurrency limiter: max 2 sandboxes at once.
    rate_limiter: Arc<Semaphore>,
}

impl LocalReasoningEngine {
    /// Initializes the engine and pre-warms PyO3.
    pub fn new(_model_name: &str, max_concurrent_requests: usize) -> Result<Self> {
        // Pre-initialize PyO3 to avoid first-call latency.
        // Even during initialization, we wrap GIL acquisition so it's
        // explicitly documented that Python shouldn't block the main thread directly
        // if this was ever converted to async.
        std::thread::spawn(|| {
            Python::with_gil(|py| {
                debug!("PyO3 v3.0 sandbox pre-warmed. Python {}", py.version());
            });
        }).join().unwrap();

        Ok(Self {
            rate_limiter: Arc::new(Semaphore::new(max_concurrent_requests)),
        })
    }

    /// Execute and validate a thought branch. Returns SandboxResult directly.
    ///
    /// Pipeline:
    /// 1. Extract code blocks (Python, JSON)
    /// 2. AST-based security scan (DEBT-T02: not string matching)
    /// 3. AST analysis (complexity, asserts, types)
    /// 4. Real execution with 5s timeout (DEBT-T01)
    /// 5. Return structured SandboxResult
    pub async fn execute(&self, thought: &str) -> SandboxResult {
        let Ok(_permit) = self.rate_limiter.acquire().await else {
            return SandboxResult {
                success: false,
                stdout: String::new(),
                error: Some("Failed to acquire sandbox semaphore".into()),
                execution_ms: 0,
                ast_analysis: AstAnalysis::empty(),
            };
        };
        debug!("Sandbox permit acquired for code evaluation");

        // 1. Extract Python code block
        let Some(code) = extract_python_block(thought) else {
            return SandboxResult {
                success: true,
                stdout: String::new(),
                error: None,
                execution_ms: 0,
                ast_analysis: AstAnalysis::empty(),
            };
        };

        // 2+3+4. AST scan + analysis + execution in blocking thread with timeout
        let code_clone = code.clone();

        // DEBT-T01: Real timeout via tokio::time::timeout
        let timeout_result = tokio::time::timeout(
            SANDBOX_TIMEOUT,
            tokio::task::spawn_blocking(move || execute_in_sandbox(&code_clone)),
        )
        .await;

        match timeout_result {
            Ok(Ok(result)) => {
                if result.success {
                    info!(
                        "Sandbox OK: {}ms, CC={}, asserts={}",
                        result.execution_ms,
                        result.ast_analysis.cyclomatic_complexity,
                        result.ast_analysis.assert_count
                    );
                } else {
                    warn!("Sandbox FAILED: {:?}", result.error);
                }
                result
            }
            Ok(Err(join_err)) => SandboxResult {
                success: false,
                stdout: String::new(),
                error: Some(format!("Tokio join error: {}", join_err)),
                execution_ms: SANDBOX_TIMEOUT.as_millis() as u64,
                ast_analysis: AstAnalysis::empty(),
            },
            Err(_elapsed) => {
                warn!("Sandbox TIMEOUT after {}s", SANDBOX_TIMEOUT.as_secs());
                SandboxResult {
                    success: false,
                    stdout: String::new(),
                    error: Some(format!(
                        "Execution timeout: exceeded {}s limit (possible infinite loop)",
                        SANDBOX_TIMEOUT.as_secs()
                    )),
                    execution_ms: SANDBOX_TIMEOUT.as_millis() as u64,
                    ast_analysis: AstAnalysis::empty(),
                }
            }
        }
    }

    /// Legacy interface for AgentRouter compatibility (DEBT-T09 partial).
    /// Delegates to execute() and serializes result.
    #[allow(dead_code)]
    pub async fn evaluate_branch(
        &self,
        _system_prompt: &str,
        _context: &str,
        thought: &str,
    ) -> Result<String> {
        let result = self.execute(thought).await;
        Ok(serde_json::to_string(&result)?)
    }
}

/// Extract Python code from markdown-style code blocks.
pub fn extract_python_block(text: &str) -> Option<String> {
    // Try ```python\n...\n``` first
    if let Some(start) = text.find("```python\n") {
        let code_start = start + 10;
        if let Some(end) = text[code_start..].find("\n```") {
            return Some(text[code_start..code_start + end].to_string());
        }
    }

    // Bare code detection (starts with common Python patterns)
    let trimmed = text.trim();
    let code_starters = [
        "import ", "from ", "def ", "class ", "assert ",
        "x =", "result =", "for ", "if ", "while ",
    ];
    if code_starters.iter().any(|s| trimmed.starts_with(s)) {
        return Some(trimmed.to_string());
    }

    None
}

/// Execute Python code in sandboxed PyO3 with AST-based security + analysis.
fn execute_in_sandbox(code: &str) -> SandboxResult {
    let start = std::time::Instant::now();

    Python::with_gil(|py| {
        // ─── Step 1: AST-Based Security Scan (DEBT-T02) ──────
        let security_result = ast_security_scan(py, code);
        if !security_result.is_empty() {
            return SandboxResult {
                success: false,
                stdout: String::new(),
                error: Some(format!("Security: {}", security_result.join(", "))),
                execution_ms: start.elapsed().as_millis() as u64,
                ast_analysis: AstAnalysis::with_violations(security_result),
            };
        }

        // ─── Step 2: AST Analysis ────────────────────────────
        let ast_analysis = run_ast_analysis(py, code);

        // Reject overly complex code
        if ast_analysis.cyclomatic_complexity > 15 {
            return SandboxResult {
                success: false,
                stdout: String::new(),
                error: Some(format!(
                    "Cyclomatic complexity too high: {} > 15",
                    ast_analysis.cyclomatic_complexity
                )),
                execution_ms: start.elapsed().as_millis() as u64,
                ast_analysis,
            };
        }

        // ─── Step 3: Sandboxed Execution ─────────────────────
        // DEBT-T04: Set recursion limit and resource guards
        // FIX-1: RLIMIT_DATA 512MB prevents OOM bombs (e.g. `x = [0] * 10**10`)
        // Skipped in test builds: rlimit is process-wide and leaks between test cases.
        #[cfg(not(test))]
        let rlimit_setup = format!(
            r#"
# FIX-1: Memory limit — 512 MB data segment cap
_orig_rlimit_data = None
try:
    import resource as _sandbox_resource
    _orig_rlimit_data = _sandbox_resource.getrlimit(_sandbox_resource.RLIMIT_DATA)
    _MEM_LIMIT = {mem_limit}
    _sandbox_resource.setrlimit(
        _sandbox_resource.RLIMIT_DATA, (_MEM_LIMIT, _MEM_LIMIT)
    )
    del _MEM_LIMIT
except (ImportError, ValueError, OSError):
    _orig_rlimit_data = None
"#,
            mem_limit = SANDBOX_MEMORY_LIMIT,
        );
        #[cfg(test)]
        let rlimit_setup = String::from("\n_orig_rlimit_data = None\n");

        let setup_script = format!(
            r#"
import sys, io
sys.setrecursionlimit({recursion_limit})
_captured_stdout = io.StringIO()
sys.stdout = _captured_stdout

# Vector 1a: Prevent exponential string attacks (e.g. str(10**10**10))
# Zero-overhead — only limits int-to-str conversion length.
try:
    sys.set_int_max_str_digits(4000)
except AttributeError:
    pass  # Python < 3.11

# V5-2a: PEP 578 Kernel Audit Hooks (CPython 3.8+)
# Catches metaprogramming evasions that bypass AST scanning:
# e.g. getattr(__import__(bytes([111,115]).decode()), 'system')('rm -rf')
# The audit hook monitors CPython's internal C-level dispatch — irrescrutable from Python.
#
# NOTE: builtins.eval/exec/compile are NOT blocked here — they are already caught
# by the AST scanner, and blocking builtins.compile kills py.run_bound() (the sandbox itself).
# sys.addaudithook is PERMANENT and process-wide — focus only on OS-level escapes.
def _security_audit_hook(event, args):
    _BLOCKED_EVENTS = {{'os.system', 'os.exec', 'os.posix_spawn', 'os.spawn',
                        'subprocess.Popen', 'shutil.rmtree', 'socket.connect',
                        'webbrowser.open'}}
    if event in _BLOCKED_EVENTS:
        raise RuntimeError(f"SECURITY_AUDIT: Kernel call blocked: {{event}}")
try:
    sys.addaudithook(_security_audit_hook)
except AttributeError:
    pass  # Python < 3.8

# V5-2b: ReDoS Guard — monkey-patch re.compile to block catastrophic backtracking.
# re.match(r"(a+)+b", "a"*10000) executes in C — line tracer never fires.
# The guard detects nested quantifiers that cause O(2^N) backtracking.
#
# CRITICAL: This must be IDEMPOTENT — sys.addaudithook and re.compile patches persist
# across the entire process lifetime. On 2nd+ calls, re.compile is already patched.
# Without this guard: _original_re_compile = re.compile → captures patched version → recursion.
import re as _re_module
if not hasattr(_re_module.compile, '_is_redos_guard'):
    _original_re_compile = _re_module.compile
    # Pre-compile the guard pattern BEFORE monkey-patching (avoids recursion)
    _redos_detector = _original_re_compile(r'\([^)]*[+*]\)[+*]')
    def _safe_re_compile(pattern, *args, **kwargs):
        pat_str = str(pattern)
        if len(pat_str) > 200:
            raise ValueError("ReDoS_GUARD: Regex pattern exceeds 200 char safety limit")
        # Detect nested quantifiers: (X+)+ or (X*)*  — catastrophic backtracking
        # Uses pre-compiled pattern to avoid recursion through monkey-patch
        if _redos_detector.search(pat_str):
            raise ValueError("ReDoS_GUARD: Nested quantifiers detected — catastrophic backtracking")
        return _original_re_compile(pattern, *args, **kwargs)
    _safe_re_compile._is_redos_guard = True
    _re_module.compile = _safe_re_compile

# Vector 1b: Gas Limit — LINE tracer (NOT 'call' — catches tight loops).
# 'call' event misses `while True: x += 1` (zero function calls → GIL deadlock).
# 'line' event fires per source line (~2.5x overhead, acceptable for <5s sandbox).
# Limit: 50,000 lines — enough for any legitimate PRM script.
#
# C2: OOM Bomb Protection — checks CPython heap every 100 lines.
# Catches O(2^N) memory attacks like `x.extend(x)` × 30 = 8.5GB RAM.
# 200,000 allocated blocks ≈ 64-128MB (well within sandbox limits).
_line_count = [0]
def _gas_tracer(frame, event, arg):
    if event == 'line':
        _line_count[0] += 1
        if _line_count[0] > 50000:
            raise RuntimeError("GAS_LIMIT_EXCEEDED: Infinite loop detected (>50000 lines)")
        if _line_count[0] % 100 == 0:
            if sys.getallocatedblocks() > 200000:
                raise MemoryError("SPACE_LIMIT_EXCEEDED: Exponential memory blowup detected")
    return _gas_tracer
sys.settrace(_gas_tracer)

{rlimit}
"#,
            recursion_limit = PYTHON_RECURSION_LIMIT,
            rlimit = rlimit_setup,
        );
        let globals = pyo3::types::PyDict::new_bound(py);

        if let Err(e) = py.run_bound(&setup_script, Some(&globals), None) {
            return SandboxResult {
                success: false,
                stdout: String::new(),
                error: Some(format!("Sandbox setup error: {}", e)),
                execution_ms: start.elapsed().as_millis() as u64,
                ast_analysis,
            };
        }

        // Execute user code
        let exec_result = py.run_bound(code, Some(&globals), None);

        // Capture stdout (truncated to MAX_STDOUT_BYTES)
        let stdout = py
            .eval_bound("_captured_stdout.getvalue()", Some(&globals), None)
            .and_then(|v| v.extract::<String>())
            .unwrap_or_default();
        let stdout = if stdout.len() > MAX_STDOUT_BYTES {
            format!("{}...[truncated]", &stdout[..MAX_STDOUT_BYTES])
        } else {
            stdout
        };

        // Restore stdout
        let _ = py.run_bound("sys.stdout = sys.__stdout__", Some(&globals), None);

        // FIX-1: Restore original RLIMIT_DATA to prevent leaking to subsequent calls
        let _ = py.run_bound(
            r#"
if _orig_rlimit_data is not None:
    try:
        _sandbox_resource.setrlimit(_sandbox_resource.RLIMIT_DATA, _orig_rlimit_data)
    except (NameError, ValueError, OSError):
        pass
"#,
            Some(&globals),
            None,
        );

        match exec_result {
            Ok(_) => {
                // C1: Z3 SMT Semantic Guard — Formal Verification Completeness.
                //
                // CRITICAL FIX: L5 penalized `unsat`, but in proof by contradiction:
                //   solver.add(Not(theorem)) → check() = unsat → NO model satisfies ¬T
                //   → Theorem T is TRUE → LLM reasoning was CORRECT!
                //
                // Correct semantics:
                //   "unknown"/"timeout" → Undecidable (Gödel/Presburger limits) → fail
                //   "counterexample"/"failed to prove" → Explicit refutation → fail
                //   "unsat"/"sat" → Valid solver result → respect Python exit code 0
                if code.contains("import z3") || code.contains("from z3") {
                    let stdout_lower = stdout.to_lowercase();

                    // Undecidable: solver hit complexity/time limits
                    if stdout_lower.contains("unknown") || stdout_lower.contains("timeout") {
                        return SandboxResult {
                            success: false,
                            stdout,
                            error: Some(
                                "Z3_UNDECIDABLE: Solver returned 'unknown' or 'timeout'. \
                                 Constraints are too complex or non-linear."
                                    .into(),
                            ),
                            execution_ms: start.elapsed().as_millis() as u64,
                            ast_analysis,
                        };
                    }

                    // Explicit refutation: Z3 found a counterexample
                    if stdout_lower.contains("failed to prove")
                        || stdout_lower.contains("counterexample")
                    {
                        return SandboxResult {
                            success: false,
                            stdout,
                            error: Some(
                                "Z3_REFUTED: Solver found a counterexample. \
                                 Mathematical hypothesis is FALSE."
                                    .into(),
                            ),
                            execution_ms: start.elapsed().as_millis() as u64,
                            ast_analysis,
                        };
                    }
                    // "unsat"/"sat" after check() → valid result, respect exit code 0
                }

                SandboxResult {
                    success: true,
                    stdout,
                    error: None,
                    execution_ms: start.elapsed().as_millis() as u64,
                    ast_analysis,
                }
            }
            Err(e) => SandboxResult {
                success: false,
                stdout,
                error: Some(format!("{}", e)),
                execution_ms: start.elapsed().as_millis() as u64,
                ast_analysis,
            },
        }
    })
}

static AST_SCAN_SCRIPT: &str = r#"
import ast

violations = []
try:
    tree = ast.parse(_code_input_)
except SyntaxError:
    # Syntax errors are caught later during execution
    pass
else:
    BLOCKED_MODULES = {
        'subprocess', 'os', 'shutil', 'socket', 'requests',
        'urllib', 'http', 'ctypes', 'pickle', 'shelve', 'marshal',
        'importlib', 'signal', 'multiprocessing', 'threading',
    }
    BLOCKED_CALLS = {'exec', 'eval', 'compile', '__import__', 'globals', 'locals', 'breakpoint'}

    for node in ast.walk(tree):
        # Check imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                top_module = alias.name.split('.')[0]
                if top_module in BLOCKED_MODULES:
                    violations.append(f"Blocked import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top_module = node.module.split('.')[0]
                if top_module in BLOCKED_MODULES:
                    violations.append(f"Blocked import: from {node.module}")
        # Check dangerous function calls
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in BLOCKED_CALLS:
                    violations.append(f"Blocked call: {node.func.id}()")
            elif isinstance(node.func, ast.Attribute):
                if node.func.attr in ('system', 'popen', 'remove', 'rmdir', 'unlink'):
                    violations.append(f"Blocked call: .{node.func.attr}()")
                # Block open() with write modes
                if node.func.attr == 'open' or (isinstance(node.func, ast.Name) and node.func.id == 'open'):
                    for kw in node.keywords:
                        if kw.arg == 'mode' and isinstance(kw.value, ast.Constant):
                            if 'w' in str(kw.value.value) or 'a' in str(kw.value.value):
                                violations.append("Blocked: file write access")

            # V5-3a: Z3 Vacuous Truth — Implies(False, H) always satisfiable
            # LLMs farm PRM scores by submitting tautologies that prove nothing.
            if isinstance(node.func, ast.Attribute) and node.func.attr == 'Implies':
                if len(node.args) >= 1:
                    antecedent = node.args[0]
                    if isinstance(antecedent, ast.Constant) and antecedent.value in (False, 0):
                        violations.append("Z3_VACUOUS_TRUTH: Implies(False, ...) proves nothing")

        # Also check bare open() calls
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'open':
            for arg in node.args[1:]:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    if 'w' in arg.value or 'a' in arg.value:
                        violations.append("Blocked: file write access via open()")
            for kw in node.keywords:
                if kw.arg == 'mode' and isinstance(kw.value, ast.Constant):
                    if 'w' in str(kw.value.value) or 'a' in str(kw.value.value):
                        violations.append("Blocked: file write access via open()")

        # V5-3b: Trivial Assertions — assert CONST == CONST proves nothing
        # Catches: assert 1 == 1, assert True, assert "a" == "a"
        if isinstance(node, ast.Assert) and hasattr(node, 'test'):
            test = node.test
            if isinstance(test, ast.Compare) and len(test.ops) == 1:
                if isinstance(test.ops[0], ast.Eq):
                    left = test.left
                    right = test.comparators[0] if test.comparators else None
                    if (isinstance(left, ast.Constant) and isinstance(right, ast.Constant)
                            and left.value == right.value):
                        violations.append("TRIVIAL_ASSERTION: Comparing identical constants")

_violations_result_ = violations
"#;

use std::sync::OnceLock;

static COMPILED_SCAN_SCRIPT: OnceLock<pyo3::Py<pyo3::types::PyAny>> = OnceLock::new();

/// DEBT-T02: AST-based security scan using Python's ast module.
/// Analyzes the actual parse tree, not string patterns.
/// This eliminates false positives from comments/strings containing blocked words.
fn ast_security_scan(py: Python<'_>, code: &str) -> Vec<String> {
    let globals = pyo3::types::PyDict::new_bound(py);
    globals.set_item("_code_input_", code).unwrap_or(());

    let compiled_code = COMPILED_SCAN_SCRIPT.get_or_init(|| {
        let builtins = py.import_bound("builtins").unwrap();
        builtins.call_method1("compile", (AST_SCAN_SCRIPT, "scan.py", "exec")).unwrap().into()
    });

    let builtins = py.import_bound("builtins").unwrap();
    if builtins.call_method1("exec", (compiled_code.bind(py), &globals)).is_err() {
        return vec![]; // Parse errors handled during execution
    }

    // Extract violations list
    py.eval_bound("_violations_result_", Some(&globals), None)
        .and_then(|v| v.extract::<Vec<String>>())
        .unwrap_or_default()
}

static AST_ANALYSIS_SCRIPT: &str = r#"
import ast

try:
    tree = ast.parse(_code_input_)
except SyntaxError as e:
    _result_ = {
        "error": f"SyntaxError: {e.msg} at line {e.lineno}",
        "cc": 0, "asserts": 0, "functions": 0, "imports": 0,
        "type_hints": False, "deterministic": True
    }
else:
    cc = 1
    asserts = 0
    functions = 0
    imports = 0
    type_hints = False
    non_deterministic = False

    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.For, ast.While)):
            cc += 1
        elif isinstance(node, (ast.And, ast.Or)):
            cc += 1
        elif isinstance(node, ast.Try):
            cc += len(node.handlers)
        elif isinstance(node, ast.Assert):
            asserts += 1
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions += 1
            if node.returns:
                type_hints = True
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            imports += 1
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module in ('random', 'time', 'datetime'):
                    non_deterministic = True
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in ('random', 'time', 'datetime'):
                        non_deterministic = True
        elif isinstance(node, ast.AnnAssign):
            type_hints = True

    _result_ = {
        "cc": cc, "asserts": asserts, "functions": functions,
        "imports": imports, "type_hints": type_hints,
        "deterministic": not non_deterministic
    }
"#;

static COMPILED_ANALYSIS_SCRIPT: OnceLock<pyo3::Py<pyo3::types::PyAny>> = OnceLock::new();

/// Run AST analysis on Python code via PyO3.
fn run_ast_analysis(py: Python<'_>, code: &str) -> AstAnalysis {
    let globals = pyo3::types::PyDict::new_bound(py);
    globals.set_item("_code_input_", code).unwrap_or(());

    let compiled_code = COMPILED_ANALYSIS_SCRIPT.get_or_init(|| {
        let builtins = py.import_bound("builtins").unwrap();
        builtins.call_method1("compile", (AST_ANALYSIS_SCRIPT, "analyze.py", "exec")).unwrap().into()
    });

    let builtins = py.import_bound("builtins").unwrap();
    if builtins.call_method1("exec", (compiled_code.bind(py), &globals)).is_err() {
        return AstAnalysis {
            security_violations: vec!["AST analysis failed".to_string()],
            ..AstAnalysis::empty()
        };
    }

    // Extract results using eval_bound (most reliable across PyO3 versions)
    let extract_usize = |key: &str, default: usize| -> usize {
        let expr = format!("_result_.get('{}', {})", key, default);
        py.eval_bound(&expr, Some(&globals), None)
            .and_then(|v| v.extract::<usize>())
            .unwrap_or(default)
    };
    let extract_bool = |key: &str, default: bool| -> bool {
        let expr = format!("_result_.get('{}', {})", key, if default { "True" } else { "False" });
        py.eval_bound(&expr, Some(&globals), None)
            .and_then(|v| v.extract::<bool>())
            .unwrap_or(default)
    };

    let has_error = py
        .eval_bound("'error' in _result_", Some(&globals), None)
        .and_then(|v| v.extract::<bool>())
        .unwrap_or(false);

    if has_error {
        return AstAnalysis {
            security_violations: vec!["AST parse error".to_string()],
            ..AstAnalysis::empty()
        };
    }

    AstAnalysis {
        cyclomatic_complexity: extract_usize("cc", 0),
        assert_count: extract_usize("asserts", 0),
        function_count: extract_usize("functions", 0),
        import_count: extract_usize("imports", 0),
        has_type_hints: extract_bool("type_hints", false),
        is_deterministic: extract_bool("deterministic", true),
        security_violations: vec![],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_python_block() {
        let input = "Some text\n```python\nassert 1 + 1 == 2\n```\nMore text";
        assert_eq!(extract_python_block(input), Some("assert 1 + 1 == 2".to_string()));
    }

    #[test]
    fn test_extract_bare_code() {
        assert_eq!(extract_python_block("assert x > 5"), Some("assert x > 5".to_string()));
    }

    #[test]
    fn test_extract_no_code() {
        assert_eq!(extract_python_block("Just natural language, no code."), None);
    }

    #[test]
    fn test_sandbox_execution_success() {
        let result = execute_in_sandbox("x = 1 + 1\nassert x == 2\nprint(x)");
        assert!(result.success, "Expected success: {:?}", result.error);
        assert_eq!(result.stdout.trim(), "2");
        assert_eq!(result.ast_analysis.assert_count, 1);
    }

    #[test]
    fn test_sandbox_execution_failure() {
        let result = execute_in_sandbox("assert 1 == 2");
        assert!(!result.success);
        assert!(result.error.is_some());
    }

    #[test]
    fn test_sandbox_syntax_error() {
        let result = execute_in_sandbox("def broken(:\n  pass");
        assert!(!result.success);
    }

    #[test]
    fn test_sandbox_cc_analysis() {
        let code = "def foo(x):\n    if x > 0:\n        for i in range(x):\n            if i % 2 == 0:\n                print(i)\n    else:\n        while x < 0:\n            x += 1";
        let result = execute_in_sandbox(code);
        assert!(result.ast_analysis.cyclomatic_complexity >= 4);
        assert_eq!(result.ast_analysis.function_count, 1);
    }

    #[test]
    fn test_ast_security_scan_clean() {
        Python::with_gil(|py| {
            let violations = ast_security_scan(py, "x = 1 + 2\nassert x == 3");
            assert!(violations.is_empty());
        });
    }

    #[test]
    fn test_ast_security_scan_blocks_import() {
        Python::with_gil(|py| {
            let violations = ast_security_scan(py, "import subprocess\nsubprocess.run(['ls'])");
            assert!(!violations.is_empty(), "Expected violation for subprocess import");
        });
    }

    #[test]
    fn test_ast_security_no_false_positive_comments() {
        // DEBT-T02: Comments mentioning blocked modules should NOT trigger
        Python::with_gil(|py| {
            let code = "# This uses subprocess internally\nx = 42\nassert x == 42";
            let violations = ast_security_scan(py, code);
            assert!(violations.is_empty(), "Comment should not trigger: {:?}", violations);
        });
    }

    #[test]
    fn test_ast_security_blocks_eval() {
        Python::with_gil(|py| {
            let violations = ast_security_scan(py, "result = eval('1 + 1')");
            assert!(!violations.is_empty(), "Expected violation for eval()");
        });
    }

    #[test]
    fn test_recursion_limit_enforced() {
        // DEBT-T04: Deep recursion should be caught
        let code = "def f(n): return f(n+1)\nf(0)";
        let result = execute_in_sandbox(code);
        assert!(!result.success, "Deep recursion should fail");
        assert!(result.error.as_deref().unwrap_or("").contains("RecursionError")
            || result.error.as_deref().unwrap_or("").contains("maximum recursion"));
    }
}
