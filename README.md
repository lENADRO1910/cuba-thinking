# 🧠 Cuba-Thinking

**Advanced cognitive reasoning engine for AI agents** — A Model Context Protocol (MCP) server that enhances AI reasoning with a 6-stage cognitive pipeline, 9-layer anti-hallucination, MCTS quality enforcement, Process Reward Model (PRM), bias detection, metacognitive analysis, persistent thought sessions, Graph-of-Thought topology, mode collapse detection, and cross-MCP memory symbiosis.

3 tools. Zero cloud dependencies. 170 tests. 7 audit rounds. Mathematically verified.

---

## Why Cuba-Thinking?

AI agents think in flat, unstructured sequences. Cuba-Thinking gives them:

- **6-stage cognitive engine** — Bloom's Taxonomy state machine: DEFINE → RESEARCH → ANALYZE → HYPOTHESIZE → VERIFY → SYNTHESIZE
- **9-layer anti-hallucination** — Assumption tracking, confidence calibration, CoVe structure, evidence accumulation, claim grounding (per-claim proximity), EWMA threshold enforcement, contradiction detection, warmup guard, anti-overthinking
- **6D quality metrics** — Clarity (TTR), Depth (clause counting), Breadth (noun diversity), Logic (connective density), Relevance (TF-IDF cosine), Actionability (imperative + specificity)
- **Process Reward Model (PRM)** — 7-signal code evaluation: Compiles, Asserts Pass, Complexity, Type Safety, Safe Imports, Determinism, Coverage
- **Sandboxed execution** — PyO3 sandbox with PEP 578 audit hooks, ReDoS guard, Z3 vacuous truth detector, and AST-level import blocking
- **MCTS forced backtracking** — Protocol-level rejection (`isError: true`) when EWMA drops below budget-aware threshold, with hedged rejection zones
- **Graph-of-Thought (GoT)** — DAG topology tracking with Tarjan SCC cycle detection O(V+E) for circular reasoning (petitio principii)
- **Persistent thought sessions** — Cross-call state accumulation: EWMA, novelty, graph, confidence oscillation, depth degradation, root-anchoring, hypothesis drift
- **Epistemological rollback** — Snapshot/rollback of session state when MCTS rejects a thought, preventing hallucinated premises from poisoning future reasoning
- **Mode collapse detection (V7)** — OrthogonalityGuard: Jaccard similarity against failed thoughts detects when LLM paraphrases rejected ideas instead of generating genuinely new hypotheses
- **Bias detection** — Identifies 5 cognitive biases (Anchoring, Confirmation, Availability, Sunk Cost, Bandwagon)
- **Metacognitive analysis** — Filler ratio, content-word ratio, claim density, fallacy detection, dialectical reasoning checks
- **Corrective directives** — Actionable improvement suggestions targeting weak quality dimensions
- **Cross-MCP memory symbiosis** — Bridge to [cuba-memorys](https://github.com/LeandroPG19/cuba-memorys) for recall/consolidation
- **EWMA reward tracking** — 6-signal composite with adaptive α floor, MACD collapse prediction, Process Advantage Verifier (PAV), and stagnation/fatigue detection
- **Contradiction detection** — Direct negation, antonym pairs, quantifier conflicts with sentence context
- **Novelty tracking** — Information gain per thought step via Jaccard distance on TF vectors
- **Depth degradation** — Tracks quality.depth history per thought, detects >50% drop vs baseline (KV cache saturation proxy)
- **Code-aware metrics** — Quality, depth, and directives adapt when input is code vs natural language
- **Anti-overthinking** — Stagnation detection, fatigue monitoring, mode collapse guard, early stopping signals

---

## Quick Start

### 1. Prerequisites

- **Rust 1.75+** & **Cargo**
- **Python 3.10+** (for PRM sandbox execution)

### 2. Build

```bash
git clone https://github.com/LeandroPG19/cuba-thinking.git
cd cuba-thinking/cuba_cognitive_engine
cargo build --release
```

### 3. Configure your AI editor

Add to your MCP configuration (e.g., `mcp_config.json`):

```json
{
  "mcpServers": {
    "cuba-thinking": {
      "command": "/path/to/cuba-thinking/cuba_cognitive_engine/target/release/cuba_cognitive_engine",
      "args": []
    }
  }
}
```

Zero environment variables. Zero cloud API keys. Runs 100% locally.

---

## The 3 Tools

### 1. `cuba_thinking` — Deep Reasoning

The core cognitive engine. Evaluates each thought step through the full analysis pipeline.

**Required parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `thought` | string | Current thinking step (**must be code/formal logic, not natural language**) |
| `thoughtNumber` | number | Current thought number (1-based) |
| `nextThoughtNeeded` | boolean | Whether another thought step follows |

**Optional parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `thinkingStage` | string | `DEFINE`, `RESEARCH`, `ANALYZE`, `HYPOTHESIZE`, `VERIFY`, `SYNTHESIZE` |
| `confidence` | number | 0.0–1.0, calibrated against stage expectations |
| `assumptions` | string[] | Tracked and deduplicated across thoughts |
| `hypothesis` | string | Current hypothesis being tested |
| `budgetMode` | string | `fast`, `balanced`, `thorough`, `exhaustive` |
| `biasDetected` | string | Agent-reported bias: `anchoring`, `confirmation`, `availability`, `sunk_cost`, `bandwagon` |
| `branchFromThought` | number | Branching point for MCTS exploration |
| `branchId` | string | Identifier for parallel reasoning paths |

**Output includes:** EWMA reward %, trust score, calibrated confidence, quality scores, contradiction warnings, bias alerts, corrective directives, memory instructions, trend indicators.

### 2. `verify_code` — Process Reward Model (PRM)

Executes Python code in a sandboxed environment and evaluates 7 quality signals.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `code` | string | Python code to verify (asserts, functions, computations) |

**7 PRM Signals:**

| Signal | Weight | Scoring |
|--------|:------:|---------|
| E1 Compiles | 0.30 | 1.0 if execution succeeds, 0.0 otherwise |
| E2 Asserts Pass | 0.25 | 1.0 with passing asserts, 0.3 without asserts, 0.0 on failure |
| E3 Complexity | 0.10 | 1.0 if CC ≤ 7, 0.7 if CC ≤ 10, 0.0 otherwise |
| E4 Type Safety | 0.10 | 1.0 with type annotations, 0.3 without |
| E5 Safe Imports | 0.05 | 1.0 clean, 0.0 with security violations |
| E6 Determinism | 0.10 | 1.0 reproducible, 0.5 with random/time |
| E7 Coverage | 0.10 | assert-to-function ratio |

**Verdicts:** EXCELLENT (≥85%), GOOD (≥65%), ACCEPTABLE (≥45%), INSUFFICIENT (<45%)

### 3. `analyze_reasoning` — Chain Analysis

Analyzes a multi-step reasoning chain for coherence, contradictions, novelty decay, and grounding quality.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `thoughts` | string[] | Array of reasoning steps to analyze in order |
| `context` | string | Optional hypothesis to check grounding against |

---

## The 6 Cognitive Stages

Based on Bloom's Revised Taxonomy (Anderson & Krathwohl, 2001):

```
DEFINE ──→ RESEARCH ──→ ANALYZE ──→ HYPOTHESIZE ──→ VERIFY ──→ SYNTHESIZE
  📋          🔍          🔬           💡             ✅          🎯
```

| Stage | Focus | Confidence Range | Boosted Dimension |
|-------|-------|:----------------:|:-----------------:|
| DEFINE | Clarify scope, requirements | 0.30 – 0.60 | Clarity (3×) |
| RESEARCH | Explore options, gather data | 0.30 – 0.70 | Breadth (3×) |
| ANALYZE | Evaluate trade-offs, compare | 0.40 – 0.80 | Depth (3×) |
| HYPOTHESIZE | Propose solutions, predict | 0.50 – 0.85 | Logic (3×) |
| VERIFY | Test, validate, confirm | 0.60 – 0.90 | Logic (3×) |
| SYNTHESIZE | Conclude, recommend, summarize | 0.70 – 0.95 | Actionability (3×) |

---

## 9-Layer Anti-Hallucination

Zero LLM calls. All verification runs locally:

| # | Layer | Method |
|:-:|-------|--------|
| 1 | **Assumption Tracking** | Dedup across all thoughts |
| 2 | **Confidence Calibration** | Per-stage expected ranges, delta tracking |
| 3 | **Chain-of-Verification (CoVe)** | Self-verification patterns (Dhuliawala et al., 2023) |
| 4 | **Evidence Accumulation** | Flags unsupported confidence increases (Wald, 1945) |
| 5 | **Claim Counter** | Verifiable assertions per sentence |
| 6 | **Source Grounding** | Per-claim proximity check — evidence in ±1 adjacent sentences |
| 7 | **EWMA Threshold** | Budget-aware MCTS rejection with hedged rejection zone (V5) |
| 8 | **Contradiction Flag** | Internal contradiction detection |
| 9 | **Warmup Guard** | Suppress false alarms for thoughts 1–2 |

**Anti-Overthinking (R10):** Detects stagnation (3+ similar EWMA), fatigue (3+ consecutive drops), and mode collapse (paraphrasing rejected thoughts). Triggers early stopping signals.

**Trust Score:**

```text
trust = quality×0.40 + evidence×0.20 + grounding×0.20 + calibrated×0.10 + ewma_ok×0.10
```

---

## MCTS Forced Backtracking

When the EWMA step reward drops below the budget-aware threshold (after thought #3), the tool call is **rejected at protocol level** with `isError: true`:

```text
Budget thresholds (UCB1 — Kocsis & Szepesvári, 2006):
  fast:       50%  (exploit — cut losses early)
  balanced:   40%  (default)
  thorough:   35%
  exhaustive: 30%  (explore — give chains room)
```

**Hedged Rejection (V5):** Instead of a binary threshold, uses a stochastic rejection zone around the MCTS threshold to prevent output engineering. Probability of rejection increases proportionally as EWMA approaches the threshold.

---

## EWMA Step Reward — Roberts (1959)

6-signal composite with adaptive α floor:

```text
EWMA_t = α · reward_t + (1 - α) · EWMA_{t-1}
α = max(2/(n+1), α_floor)    — budget-aware floor

reward = 0.40·quality + 0.20·faithfulness + 0.10·coherence
       + 0.10·(1 - contradiction_rate) + 0.10·info_gain + 0.10·grounding
```

- **Reward history**: Capped at 20 entries (VecDeque ring buffer)
- **Stagnation detection**: 3+ steps with <2% EWMA change
- **Fatigue detection**: 3+ consecutive quality drops
- **MACD collapse prediction**: Convergence/divergence signal for early detection of quality collapse
- **Process Advantage Verifier (PAV)**: Measures advantage over baseline, penalizes "vacuous depth" (deep reasoning without real content)

---

## Graph-of-Thought (GoT)

DAG topology tracking across reasoning chains with cycle detection:

- **Nodes**: Each thought step becomes a node
- **Edges**: Sequential and revision dependencies
- **Convergence**: Multiple paths merging (in-degree > 1)
- **Revisions**: Explicit thought revision tracking
- **Cycle detection**: Tarjan's SCC algorithm O(V+E) detects circular reasoning (petitio principii: "X because Y" + "Y because X")
- **TopologySummary**: Nodes, edges, depth, convergence, revisions, orphans, cycle_count

---

## Persistent Thought Sessions

Sessions maintain state across multiple MCP tool calls sharing the same hypothesis:

- **EWMA accumulation**: Quality tracking persists across calls
- **Novelty tracking**: Vocabulary grows across calls
- **Graph-of-Thought**: DAG builds across calls
- **Trend indicator**: ↗️ Improving, → Stable, ↘️ Declining (based on EWMA history)
- **Hypothesis drift (G11)**: Semantic distance from original hypothesis
- **Root-anchoring**: Combined drift detection (hypothesis + first thought)
- **Confidence oscillation**: Detects rapidly alternating confidence (>3 sign changes in 5 readings)
- **Depth degradation (V6)**: Tracks quality.depth history, detects >50% drop vs baseline (first 3 thoughts) — indicates KV cache saturation
- **Epistemological rollback (V5)**: Snapshot/rollback when MCTS rejects a thought — physically removes dead branch state from thoughts, confidence_history, depth_history, and graph
- **Mode collapse guard (V7)**: Stores rejected thought texts (max 5), detects paraphrasing via Jaccard similarity > 0.6. Rollback clears failures for fresh exploration
- **Auto-expire**: TTL 600s to prevent memory leaks

---

## Sandbox Security

Multi-layered Python sandbox via PyO3:

| Layer | Protection |
|-------|------------|
| **AST Scanner** | Blocks dangerous imports at parse time (os.system, subprocess, etc.) |
| **PEP 578 Audit Hooks** (V5) | Blocks OS-level events at runtime (os.exec*, subprocess, ctypes, shutil, webbrowser) |
| **ReDoS Guard** (V5) | Monkey-patches `re.compile` with backreference length limit (idempotent) |
| **Z3 Vacuous Truth** (V5) | Detects vacuous truths and trivial assertions in Z3 solver outputs |

---

## Contradiction Detection

Three-signal heuristic pipeline:

1. **Direct Negation** — Detects "not X" vs "X" across sentences with shared context
2. **Antonym Pairs** — 20 antonym pairs (fast/slow, increase/decrease, etc.)
3. **Quantifier Conflicts** — Universal vs existential quantifiers (all/none, always/never)

Internal contradictions (within a single thought) also detected via sentence-pair analysis.

---

## Metacognitive Analysis

| Metric | Method | Warning Threshold |
|--------|--------|:-----------------:|
| Filler Ratio | Filler words / total words | > 30% |
| Content-Word Ratio | Non-function words / total (LazyLock) | < 40% |
| Claim Density | Verifiable assertions per sentence | Informational |
| Fallacy Detection | Hasty Generalization, False Dichotomy | Any detected |
| Dialectical Check | Counter-arguments in VERIFY/SYNTHESIZE | Missing |

---

## Bias Detection

| Bias | Detection Method |
|------|-----------------|
| **Anchoring** | Over-reliance on first-mentioned data |
| **Confirmation** | Only seeking supporting evidence |
| **Availability** | Reliance on recent/memorable examples |
| **Sunk Cost** | Defending prior decisions despite evidence |
| **Bandwagon** | "Everyone uses X" reasoning |

Each detected bias includes confidence level, explanation, and actionable suggestion.

---

## Memory Symbiosis

Cross-MCP bridge to [cuba-memorys](https://github.com/LeandroPG19/cuba-memorys):

| Stage | Trigger | Injected Instruction |
|-------|---------|---------------------|
| **DEFINE** (thought ≤ 2) | Problem definition | `cuba_faro(query:...)` — search past knowledge |
| | | `cuba_expediente(query:...)` — check past errors |
| **SYNTHESIZE** (!nextThought) | Conclusion | `cuba_cronica(action:"add", ...)` — consolidate lesson |

Analogous to the cortex-hippocampus consolidation cycle (McClelland et al., 1995).

---

## Budget Modes

| Mode | EWMA α Floor | MCTS Threshold | Max Thoughts |
|------|:------------:|:--------------:|:------------:|
| ⚡ `fast` | 0.30 | 50% | 5 |
| ⚖️ `balanced` | 0.25 | 40% | 10 |
| 🔎 `thorough` | 0.20 | 35% | 20 |
| 🔬 `exhaustive` | 0.15 | 30% | 50 |

---

## Architecture

```text
cuba-thinking/
└── cuba_cognitive_engine/
    ├── Cargo.toml
    └── src/
        ├── main.rs                          # Entry point
        ├── server/
        │   └── mcp_protocol.rs              # JSON-RPC 2.0 server, tool dispatch, progress streaming
        └── engine/
            ├── mod.rs                       # Module registry
            │
            ├── ── Cognitive Core ──
            ├── agent_router.rs              # Main orchestrator
            ├── stage_engine.rs              # 6-stage state machine (Bloom's Taxonomy)
            ├── quality_metrics.rs           # 6D quality + Shannon entropy + LZ76 complexity
            ├── ewma_reward.rs               # EWMA + MACD + Hedged Rejection + PAV
            ├── budget.rs                    # 4 budget modes with adaptive thresholds
            ├── anti_hallucination.rs         # 9-layer trust verification
            ├── bias_detector.rs             # 5 cognitive bias detectors
            ├── metacognition.rs             # Filler, CWR, fallacies, dialectics
            ├── thought_graph.rs             # GoT DAG + Tarjan SCC cycle detection (V6)
            ├── memory_bridge.rs             # Cross-MCP memory symbiosis
            ├── formatter.rs                 # Output formatting
            │
            ├── ── Semantics ──
            ├── semantic_similarity.rs       # TF-IDF cosine coherence
            ├── contradiction_detector.rs    # Negation, antonyms, quantifiers
            ├── novelty_tracker.rs           # Information gain (Jaccard distance)
            ├── claim_grounding.rs           # ROSCOE faithfulness + specificity
            │
            ├── ── Deep Reasoning ──
            ├── thought_session.rs           # Persistent sessions + rollback + depth trend + mode collapse (V5/V6/V7)
            ├── corrective_directives.rs     # Actionable improvement suggestions
            ├── stage_validator.rs           # Stage transition validation
            │
            ├── ── Execution ──
            ├── micro_prm.rs                 # Process Reward Model (7 signals)
            ├── sandbox.rs                   # PyO3 sandbox + PEP 578 + ReDoS + Z3 (V5)
            ├── mcts_graph.rs                # MCTS graph structure
            └── shared_utils.rs              # Centralized stopwords, UTF-8 truncation
```

### Dependencies

| Crate | Purpose |
|-------|---------|
| `tokio` | Async runtime |
| `serde` + `serde_json` | Serialization |
| `jsonrpc-core` | JSON-RPC 2.0 protocol |
| `pyo3` | Python sandbox for PRM code execution |
| `regex` | Pattern matching |
| `tracing` | Structured logging |
| `anyhow` + `thiserror` | Error handling |

### Test Suite

- **170 tests** covering all engine modules (run with `--test-threads=1` for PyO3 safety)
- **0 clippy errors** on `cargo clippy`
- Property-based boundary testing for all scoring functions
- NEMESIS 3-level test structure: 🟢 Normal, 🟡 Pessimistic, 🔴 Extreme
- 7 audit rounds of security and mathematical hardening

---

## Silent by Default

Features only appear when actionable:

| Feature | Appears When |
|---------|-------------|
| EWMA Reward | Always (core metric) |
| Corrective Directives | Quality dimension below threshold |
| Contradiction Warning | Negation/antonym/quantifier conflict detected |
| Bias Alert | Cognitive bias pattern detected |
| Metacognition Warning | Filler > 30% or CWR < 40% |
| Fallacy Warning | Hasty generalization or false dichotomy |
| Dialectical Warning | VERIFY/SYNTHESIZE without counter-arguments |
| Anti-Overthinking | 3+ stagnant or fatigued thoughts |
| MCTS Backtracking | EWMA < threshold after thought #3 |
| Circular Reasoning | Tarjan SCC finds cycle in GoT (V6) |
| Depth Degradation | quality.depth drops >50% vs baseline (V6) |
| Confidence Oscillation | >3 sign changes in 5 readings |
| Hypothesis Drift | Semantic distance from original > threshold |
| Mode Collapse | Jaccard similarity > 0.6 with rejected thought (V7) |
| Memory Instructions | DEFINE/SYNTHESIZE stages |
| Stage Mismatch | Declared vs detected stage disagree |

---

## Mathematical Verification

Every formula verified with unit tests and Wolfram Alpha:

| Formula | Description | Verified |
|---------|-------------|:--------:|
| EWMA α = max(2/(n+1), α_floor) | Adaptive smoothing | ✅ |
| Composite reward (6 signals) | Weights sum to 1.0 | ✅ |
| PRM composite (7 signals) | Weights sum to 1.0 | ✅ |
| Trust score (5 components) | Weights sum to 1.0 | ✅ |
| TF-IDF cosine similarity | Coherence scoring | ✅ |
| Jaccard distance | Novelty information gain | ✅ |
| Per-claim grounding ratio | ±1 sentence proximity | ✅ |
| Confidence calibration | Stage-aware ranges | ✅ |
| Tarjan SCC O(V+E) | Cycle detection correctness | ✅ |
| Depth degradation baseline | First-3-mean vs current | ✅ |
| Jaccard similarity &#124;A∩B&#124;/&#124;A∪B&#124; | Mode collapse detection | ✅ |
| Shannon entropy H(X) | Text information density | ✅ |
| LZ76 complexity | Kolmogorov complexity proxy | ✅ |

---

## Part of the Cuba Ecosystem

| Project | Purpose |
|---------|---------|
| [Cuba-Memorys](https://github.com/LeandroPG19/cuba-memorys) | Persistent memory — knowledge graph, Hebbian learning, RLHF feedback |
| **Cuba-Thinking** | Cognitive reasoning — quality metrics, anti-hallucination, PRM, MCTS enforcement |
| [Cuba-Search](https://github.com/LeandroPG19/cuba-search) | Web search — research, scraping, validation, documentation lookup |
| [Cuba-Exec](https://github.com/LeandroPG19/cuba-exec) | Command execution — background processes, signals, interactive stdin |

Together, they give AI agents **memory + reasoning + search + execution**.

---

## Academic References

| # | Citation | Used For |
|---|----------|----------|
| 1 | Anderson & Krathwohl (2001). "Revised Bloom's Taxonomy" | 6 cognitive stages |
| 2 | Roberts (1959). "EWMA Control Charts" | Adaptive EWMA smoothing |
| 3 | Kocsis & Szepesvári (2006). "UCB Applied to Trees" | Budget-aware MCTS thresholds |
| 4 | Golovneva et al. (2023). "ROSCOE" — ICLR | Faithfulness, claim grounding |
| 5 | Dhuliawala et al. (2023). "CoVe Reduces Hallucination" — Meta AI | Chain-of-Verification |
| 6 | Lightman et al. (2023). "Let's Verify Step by Step" — OpenAI | Step-level reward (PRM) |
| 7 | Kahneman & Tversky (1974). "Judgment Under Uncertainty" | Cognitive bias detection |
| 8 | Flavell (1979). "Metacognition and Cognitive Monitoring" | Metacognitive analysis |
| 9 | Graesser et al. (2004). "Coh-Metrix" | Content-word ratio |
| 10 | Templin (1957). "Certain Language Skills in Children" | TTR clarity metric |
| 11 | Hunt (1965). "Grammatical Structures" | Clause depth analysis |
| 12 | Guan et al. (2024). "GRACE" | Actionability scoring |
| 13 | Wald (1945). "Sequential Analysis" | Evidence accumulation |
| 14 | Shannon (1948). "Mathematical Theory of Communication" | Information gain, entropy |
| 15 | McClelland et al. (1995). "Complementary Learning Systems" | Memory symbiosis |
| 16 | DeepSeek (2025). "Thoughtology" | Anti-overthinking |
| 17 | Zangari (1994). "EWMA for Risk Management" | Adaptive alpha floor |
| 18 | Salton (1975). "Vector Space Model" | TF-IDF cosine similarity |
| 19 | Tarjan (1972). "Depth-First Search and Linear Graph Algorithms" | SCC cycle detection |
| 20 | Lempel & Ziv (1976). "On the Complexity of Finite Sequences" | LZ76 complexity metric |
| 21 | Press et al. (2022). "Train Short, Test Long" | Depth degradation / KV cache saturation |
| 22 | PEP 578 (2019). "Python Runtime Audit Hooks" | Sandbox security layer |
| 23 | Cilibrasi & Vitányi (2005). "Clustering by Compression" — IEEE TIT | NCD-inspired mode collapse detection |

---

## License

[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) — Free to use and modify, **not for commercial use**.

---

## Author

**Leandro Pérez G.**

- GitHub: [@LeandroPG19](https://github.com/LeandroPG19)
- Email: [leandropatodo@gmail.com](mailto:leandropatodo@gmail.com)
