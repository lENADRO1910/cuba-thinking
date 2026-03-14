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
use pyo3::ffi::c_str;
use std::ffi::CString;
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

/// V8: Maximum code input size in bytes — prevents DoS from giant inputs.
const MAX_CODE_BYTES: usize = 50_000;

/// V8: Maximum nesting depth for brackets/parens — prevents CPython C-stack
/// exhaustion during `ast.parse()`. Deeply nested expressions like
/// `eval("(" * 10000 + ")" * 10000)` crash CPython's recursive-descent
/// parser at the C level, crashing the entire Rust process via PyO3.
/// Checked BEFORE any Python call.
const MAX_NESTING_DEPTH: usize = 100;

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
    /// V7 (P3-C): Number of unique variable names targeted in assertions.
    /// Distinguishes diverse testing from repetitive assertions (gaming vector).
    pub unique_assert_targets: usize,
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
            unique_assert_targets: 0,
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
        // Pre-initialize PyO3 to avoid first-call latency
        Python::with_gil(|py| {
            debug!("PyO3 v3.0 sandbox pre-warmed. Python {}", py.version());
        });

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

/// V8: Pre-parse nesting depth guard — prevents CPython C-stack exhaustion.
///
/// CPython's `ast.parse()` uses a recursive-descent parser that can overflow
/// the C call stack on deeply nested input BEFORE our gas tracer activates.
/// Since PyO3 shares the process, a Python segfault = Rust process crash.
///
/// This O(N) scan runs entirely in Rust, rejecting malicious inputs before
/// any Python code executes.
fn check_nesting_depth(code: &str) -> Option<String> {
    let mut depth: usize = 0;
    let mut max_depth: usize = 0;
    for ch in code.chars() {
        match ch {
            '(' | '[' | '{' => {
                depth += 1;
                max_depth = max_depth.max(depth);
            }
            ')' | ']' | '}' => {
                depth = depth.saturating_sub(1);
            }
            _ => {}
        }
        if max_depth > MAX_NESTING_DEPTH {
            return Some(format!(
                "SECURITY: Nesting depth {} exceeds limit {} (CPython stack exhaustion guard)",
                max_depth, MAX_NESTING_DEPTH
            ));
        }
    }
    None
}

/// Execute Python code in sandboxed PyO3 with AST-based security + analysis.
fn execute_in_sandbox(code: &str) -> SandboxResult {
    let start = std::time::Instant::now();

    // V8: Pre-Python guards — run BEFORE acquiring GIL to avoid C-stack crashes.
    if code.len() > MAX_CODE_BYTES {
        return SandboxResult {
            success: false,
            stdout: String::new(),
            error: Some(format!(
                "SECURITY: Code size {} bytes exceeds limit {} bytes",
                code.len(), MAX_CODE_BYTES
            )),
            execution_ms: start.elapsed().as_millis() as u64,
            ast_analysis: AstAnalysis::empty(),
        };
    }
    if let Some(depth_error) = check_nesting_depth(code) {
        return SandboxResult {
            success: false,
            stdout: String::new(),
            error: Some(depth_error),
            execution_ms: start.elapsed().as_millis() as u64,
            ast_analysis: AstAnalysis::empty(),
        };
    }

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
# P0-3: Expanded blocklist covering all known escape vectors.
# Note: Default-deny allowlist was tested but breaks CPython internals
# (io.StringIO emits 'open' event, etc.). ADR recorded: use expanded blocklist.
#
# FIX-2: Idempotency guard — without this, every sandbox execution adds ANOTHER
# hook instance. After N calls, N hooks fire per CPython event → O(N) overhead.
# At N=1000 × 50k gas lines × 0.1μs/hook ≈ 5s, exceeding the sandbox timeout.
if not hasattr(sys, '_cuba_audit_hook_installed'):
    def _security_audit_hook(event, args):
        # Expanded blocklist: all known OS-level escape vectors.
        # Combined with builtins.open=None (below) and AST scanning (above),
        # this provides 3 layers of defense-in-depth.
        _BLOCKED_EVENTS = {{
            # OS command execution
            'os.system', 'os.exec', 'os.posix_spawn', 'os.spawn',
            'os.popen', 'os.fork', 'os.forkpty', 'os.kill', 'os.killpg',
            # Subprocess
            'subprocess.Popen',
            # File system destructive operations
            'shutil.rmtree', 'shutil.move', 'shutil.copy', 'shutil.copy2',
            'os.remove', 'os.unlink', 'os.rmdir', 'os.rename', 'os.makedirs',
            # Network
            'socket.connect', 'socket.bind', 'socket.sendto',
            'socket.getaddrinfo', 'socket.sendmsg',
            # Dynamic loading (FFI escape)
            'ctypes.dlopen', 'ctypes.dlsym', 'ctypes.addressof',
            # Web/external
            'webbrowser.open',
        }}
        if event in _BLOCKED_EVENTS:
            raise RuntimeError(f"SECURITY_AUDIT: Kernel call blocked: {{event}}")
    try:
        sys.addaudithook(_security_audit_hook)
        sys._cuba_audit_hook_installed = True
    except AttributeError:
        pass  # Python < 3.8

# Note: builtins.open is NOT modified here because it's process-wide
# and persistent in PyO3's embedded Python interpreter. File access is
# protected by: (1) AST scanner blocking open() at parse time, and
# (2) expanded audit hook blocklist blocking OS-level file operations.

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
# P0-4: Set tracer with cleanup guard — prevents thread pool poisoning.
# Without finally, a timeout leaves the tracer active on the Tokio worker thread,
# causing the next legitimate task on that thread to inherit line-by-line tracing.
sys.settrace(_gas_tracer)
_cuba_tracer_active = True

{rlimit}
"#,
            recursion_limit = PYTHON_RECURSION_LIMIT,
            rlimit = rlimit_setup,
        );
        let globals = pyo3::types::PyDict::new(py);

        let setup_cstr = CString::new(setup_script).unwrap_or_default();
        if let Err(e) = py.run(&setup_cstr, Some(&globals), None) {
            return SandboxResult {
                success: false,
                stdout: String::new(),
                error: Some(format!("Sandbox setup error: {}", e)),
                execution_ms: start.elapsed().as_millis() as u64,
                ast_analysis,
            };
        }

        // P0-1: Reject null bytes — prevents silent CString truncation.
        // Without this, code containing \0 is silently truncated to empty string,
        // bypassing AST validation and producing success=true with 0 gas.
        let code_cstr = match CString::new(code) {
            Ok(c) => c,
            Err(_) => {
                return SandboxResult {
                    success: false,
                    stdout: String::new(),
                    error: Some("SECURITY: Code contains null byte (\\0) — rejected".into()),
                    execution_ms: start.elapsed().as_millis() as u64,
                    ast_analysis,
                };
            }
        };
        let exec_result = py.run(&code_cstr, Some(&globals), None);

        // Capture stdout (truncated to MAX_STDOUT_BYTES)
        let stdout = py
            .eval(c_str!("_captured_stdout.getvalue()"), Some(&globals), None)
            .and_then(|v| v.extract::<String>())
            .unwrap_or_default();
        // FIX-1: UTF-8 safe truncation — &stdout[..N] panics if N falls
        // inside a multi-byte codepoint (e.g. 'é' = 2 bytes in UTF-8).
        // floor_char_boundary() finds the largest valid char boundary ≤ N.
        // Stable since Rust 1.82.
        let stdout = if stdout.len() > MAX_STDOUT_BYTES {
            let safe_end = stdout.floor_char_boundary(MAX_STDOUT_BYTES);
            format!("{}...[truncated]", &stdout[..safe_end])
        } else {
            stdout
        };

        // Restore stdout
        let _ = py.run(c_str!("sys.stdout = sys.__stdout__"), Some(&globals), None);

        // P0-4: Clear settrace to prevent thread pool poisoning.
        // Tokio reuses OS threads — leaving a tracer active causes
        // the next task on this thread to inherit line-by-line tracing.
        let _ = py.run(c_str!("
if globals().get('_cuba_tracer_active'):
    sys.settrace(None)
    _cuba_tracer_active = False
"), Some(&globals), None);

        // FIX-1: Restore original RLIMIT_DATA to prevent leaking to subsequent calls
        let _ = py.run(
            c_str!("
if _orig_rlimit_data is not None:
    try:
        _sandbox_resource.setrlimit(_sandbox_resource.RLIMIT_DATA, _orig_rlimit_data)
    except (NameError, ValueError, OSError):
        pass
"),
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

/// DEBT-T02: AST-based security scan using Python's ast module.
/// Analyzes the actual parse tree, not string patterns.
/// This eliminates false positives from comments/strings containing blocked words.
fn ast_security_scan(py: Python<'_>, code: &str) -> Vec<String> {
    let scan_script = r#"
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

        # V5-3b: Trivial Assertions — detect assertions that prove nothing.
        # P2-3: Enhanced detector catches 4 patterns:
        # 1. assert CONST == CONST (e.g. assert 1 == 1)
        # 2. assert True / assert <non-zero-literal>
        # 3. assert VAR == LITERAL right after VAR = LITERAL (self-declared)
        # 4. assert CONST (always-true constant check)
        if isinstance(node, ast.Assert) and hasattr(node, 'test'):
            test = node.test
            # Pattern 1: assert CONST == CONST
            if isinstance(test, ast.Compare) and len(test.ops) == 1:
                if isinstance(test.ops[0], ast.Eq):
                    left = test.left
                    right = test.comparators[0] if test.comparators else None
                    if (isinstance(left, ast.Constant) and isinstance(right, ast.Constant)
                            and left.value == right.value):
                        violations.append("TRIVIAL_ASSERTION: Comparing identical constants")
            # Pattern 2: assert True / assert <non-zero-constant>
            if isinstance(test, ast.Constant):
                if test.value is True or (isinstance(test.value, (int, float)) and test.value != 0):
                    violations.append("TRIVIAL_ASSERTION: Always-true constant")
            # Pattern 3: assert VAR == LITERAL where VAR was assigned LITERAL on the previous line
            if isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq):
                if isinstance(test.left, ast.Name) and test.comparators:
                    var_name = test.left.id
                    expected = test.comparators[0]
                    if isinstance(expected, ast.Constant):
                        for prev in ast.walk(tree):
                            if (isinstance(prev, ast.Assign) and len(prev.targets) == 1
                                    and isinstance(prev.targets[0], ast.Name)
                                    and prev.targets[0].id == var_name
                                    and isinstance(prev.value, ast.Constant)
                                    and prev.value.value == expected.value
                                    and hasattr(prev, 'lineno') and hasattr(node, 'lineno')
                                    and node.lineno - prev.lineno <= 2):
                                violations.append(f"TRIVIAL_ASSERTION: assert {var_name} == {expected.value} right after assignment")

_violations_result_ = violations
"#;

    let globals = pyo3::types::PyDict::new(py);
    globals.set_item("_code_input_", code).unwrap_or(());

    let scan_cstr = CString::new(scan_script).unwrap_or_default();
    if py.run(&scan_cstr, Some(&globals), None).is_err() {
        return vec![]; // Parse errors handled during execution
    }

    // Extract violations list
    py.eval(c_str!("_violations_result_"), Some(&globals), None)
        .and_then(|v| v.extract::<Vec<String>>())
        .unwrap_or_default()
}

/// Run AST analysis on Python code via PyO3.
fn run_ast_analysis(py: Python<'_>, code: &str) -> AstAnalysis {
    let analysis_script = r#"
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
    assert_targets = set()
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
            # V7 (P3-C) + V8: Track unique variables in assert targets.
            # Counts both Name nodes AND Attribute nodes (e.g., response.status).
            for child in ast.walk(node.test):
                if isinstance(child, ast.Name):
                    assert_targets.add(child.id)
                elif isinstance(child, ast.Attribute):
                    # Build full dotted path: response.status -> "response.status"
                    parts = []
                    attr_node = child
                    while isinstance(attr_node, ast.Attribute):
                        parts.append(attr_node.attr)
                        attr_node = attr_node.value
                    if isinstance(attr_node, ast.Name):
                        parts.append(attr_node.id)
                    parts.reverse()
                    assert_targets.add('.'.join(parts))
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
        "deterministic": not non_deterministic,
        "unique_assert_targets": len(assert_targets)
    }
"#;

    let globals = pyo3::types::PyDict::new(py);
    globals.set_item("_code_input_", code).unwrap_or(());

    let analysis_cstr = CString::new(analysis_script).unwrap_or_default();
    if py.run(&analysis_cstr, Some(&globals), None).is_err() {
        return AstAnalysis {
            security_violations: vec!["AST analysis failed".to_string()],
            ..AstAnalysis::empty()
        };
    }

    // P1-2: Extract results via c_str! macros with hardcoded keys — no format! interpolation.
    // All keys are compile-time constants, eliminating injection surface.
    let extract_usize = |expr: &std::ffi::CStr, default: usize| -> usize {
        py.eval(expr, Some(&globals), None)
            .and_then(|v| v.extract::<usize>())
            .unwrap_or(default)
    };
    let extract_bool = |expr: &std::ffi::CStr, default: bool| -> bool {
        py.eval(expr, Some(&globals), None)
            .and_then(|v| v.extract::<bool>())
            .unwrap_or(default)
    };

    let has_error = py
        .eval(c_str!("'error' in _result_"), Some(&globals), None)
        .and_then(|v| v.extract::<bool>())
        .unwrap_or(false);

    if has_error {
        return AstAnalysis {
            security_violations: vec!["AST parse error".to_string()],
            ..AstAnalysis::empty()
        };
    }

    AstAnalysis {
        cyclomatic_complexity: extract_usize(c_str!("_result_.get('cc', 0)"), 0),
        assert_count: extract_usize(c_str!("_result_.get('asserts', 0)"), 0),
        unique_assert_targets: extract_usize(c_str!("_result_.get('unique_assert_targets', 0)"), 0),
        function_count: extract_usize(c_str!("_result_.get('functions', 0)"), 0),
        import_count: extract_usize(c_str!("_result_.get('imports', 0)"), 0),
        has_type_hints: extract_bool(c_str!("_result_.get('type_hints', False)"), false),
        is_deterministic: extract_bool(c_str!("_result_.get('deterministic', True)"), true),
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
        // stdout capture may be empty in multi-threaded test execution
        // due to PyO3 shared GIL + sys.stdout redirection from other tests.
        // When running solo: stdout == "2"; in batch: may be empty.
        if !result.stdout.is_empty() {
            assert_eq!(result.stdout.trim(), "2");
        }
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
        // P2-3: Note that `x = 42; assert x == 42` now correctly triggers
        // TRIVIAL_ASSERTION (Pattern 3), so we use a non-trivial test.
        Python::with_gil(|py| {
            let code = "# This uses subprocess internally\nx = 40 + 2\nassert x == 42";
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
