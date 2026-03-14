# 🧠 Cuba-Thinking

**Advanced cognitive reasoning engine for AI agents** — A Model Context Protocol (MCP) server that enhances AI reasoning with a 6-stage cognitive pipeline, 9-layer anti-hallucination, MCTS quality enforcement, 8-signal Process Reward Model (PRM) with Veto Gate, bias detection, metacognitive analysis, persistent thought sessions, Graph-of-Thought topology with Kahn's DP longest path, mode collapse detection, Softmax Gated Attention reward scoring, CUSUM change-point detection, and cross-MCP memory symbiosis.

3 tools. Zero cloud dependencies. 183 tests. 9 audit rounds. 16 magic constants formally verified.

---

## Why Cuba-Thinking?

AI agents think in flat, unstructured sequences. Cuba-Thinking gives them:

- **6-stage cognitive engine** — Bloom's Taxonomy state machine: DEFINE → RESEARCH → ANALYZE → HYPOTHESIZE → VERIFY → SYNTHESIZE
- **9-layer anti-hallucination** — Assumption tracking, confidence calibration, CoVe structure with verification question detection, evidence accumulation, claim grounding (per-claim proximity), EWMA threshold enforcement, contradiction detection, warmup guard, anti-overthinking
- **6D quality metrics** — Clarity (Root-TTR), Depth (clause counting), Breadth (noun diversity), Logic (connective density), Relevance (TF-IDF cosine), Actionability (imperative + specificity)
- **Process Reward Model (PRM)** — 8-signal code evaluation with V9 Logical Veto Gate: `Gate × (0.6×Verify + 0.4×Quality)` prevents uncompilable code from scoring high
- **Sandboxed execution** — PyO3 sandbox with AST-based security scan, RLIMIT_AS 512MB memory cap, recursion limit 100, nesting depth 100, PEP 578 audit hooks, ReDoS guard, Z3 vacuous truth detector
- **MCTS forced backtracking** — Protocol-level rejection (`isError: true`) when EWMA drops below budget-aware threshold, with hedged rejection zones
- **MCTS Graph** — Arena-allocated PUCT + Adaptive UCT (variance-based c_puct, correlation_lambda) — superior to standard UCB1
- **Graph-of-Thought (GoT)** — DAG topology with Kahn's topological sort + DP for correct longest-path depth, Tarjan SCC cycle detection O(V+E) for circular reasoning
- **Persistent thought sessions** — Cross-call state accumulation: EWMA, novelty, graph, confidence oscillation, depth degradation, root-anchoring, hypothesis drift
- **Epistemological rollback** — Snapshot/rollback of session state when MCTS rejects a thought, preventing hallucinated premises from poisoning future reasoning
- **Mode collapse detection** — OrthogonalityGuard: Jaccard similarity against failed thoughts detects paraphrased rejected ideas
- **Bias detection** — 5 cognitive biases (Anchoring, Confirmation, Availability, Sunk Cost, Bandwagon) with confidence + actionable suggestions
- **Metacognitive analysis** — Filler ratio, content-word ratio (Coh-Metrix), claim density, fallacy detection, dialectical reasoning checks
- **Corrective directives** — 3 severity levels (INFO/WARNING/CORRECTION), prescriptive improvement targeting weak quality dimensions
- **Stage-content alignment** — Validates thought content matches declared cognitive stage + Logical Validity Score (ReasonEval/RECEVAL 2024)
- **Reward Consistency Check** — PRM↔EWMA divergence detection (Z≈1.9, P(false alarm)≈2.9%) for reward gaming prevention
- **Softmax Gated Attention** — Dynamic EWMA weights via softmax with τ=0.5, Bayesian prior initialization (Fin-PRM inspired, Zhou et al. 2025)
- **CUSUM change-point detection** — Lag-free quality collapse detection replacing MACD (μ=0.70, k=0.05, h=0.15)
- **Cross-MCP memory symbiosis** — Bridge to [cuba-memorys](https://github.com/LeandroPG19/cuba-memorys) with anti-repetition guard
- **EWMA reward tracking** — 6-signal composite with adaptive α floor, stage-adaptive weight profiles, CUSUM collapse prediction, Process Advantage Verifier (PAV), stagnation/fatigue detection
- **Contradiction detection** — Direct negation, 30+ antonym pairs, quantifier conflicts with sentence context
- **Novelty tracking** — Shannon information gain per thought step via Jaccard distance on TF vectors
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

**Output includes:** EWMA reward %, trust score, calibrated confidence, quality scores, contradiction warnings, bias alerts, corrective directives, memory instructions, trend indicators, reward consistency warnings.

### 2. `verify_code` — Process Reward Model (PRM)

Executes Python code in a sandboxed environment and evaluates 8 quality signals.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `code` | string | Python code to verify (asserts, functions, computations) |

**8 PRM Signals:**

| Signal | Weight | Scoring |
|--------|:------:|---------|
| E1 Compiles | 0.25 | 1.0 if execution succeeds, 0.0 otherwise |
| E2 Asserts Pass | 0.25 | 1.0 with passing asserts, 0.3 without asserts, 0.0 on failure |
| E3 Complexity | 0.10 | 1.0 if CC ≤ 7, 0.7 if CC ≤ 10, 0.0 otherwise |
| E4 Type Safety | 0.08 | 1.0 with type annotations, 0.3 without |
| E5 Safe Imports | 0.05 | 1.0 clean, 0.0 with security violations |
| E6 Determinism | 0.10 | 1.0 reproducible, 0.5 with random/time |
| E7 Coverage | 0.07 | assert-to-function ratio |
| E8 Diversity | 0.10 | unique assert targets / total asserts (anti-gaming) |

**V9 Veto Gate:** `Gate = (E1×0.7 + E4×0.3).max(0.05)` — uncompilable code gets near-zero score regardless of other signals.

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
| DEFINE | Clarify scope, requirements | 0.30 – 0.60 | Clarity (2×) |
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
| 2 | **Confidence Calibration** | Per-stage expected ranges, Bayesian blend with PRM evidence |
| 3 | **Chain-of-Verification (CoVe)** | Self-verification keywords + question detection (Dhuliawala et al., 2023) |
| 4 | **Evidence Accumulation** | Flags unsupported confidence increases (Wald, 1945) |
| 5 | **Claim Counter** | Verifiable assertions per sentence |
| 6 | **Source Grounding** | Per-claim proximity check — evidence in ±1 adjacent sentences |
| 7 | **EWMA Threshold** | Budget-aware MCTS rejection with hedged rejection zone |
| 8 | **Contradiction Flag** | Internal contradiction detection |
| 9 | **Warmup Guard** | Suppress false alarms for thoughts 1–2 |

**Anti-Overthinking (R10):** Detects stagnation (3+ similar EWMA), fatigue (3+ consecutive drops), and mode collapse (paraphrasing rejected thoughts). Triggers early stopping signals.

**Trust Score:**

```text
trust = quality×0.40 + evidence×0.20 + grounding×0.20 + calibrated×0.10 + ewma_ok×0.10
```

---

## EWMA Step Reward — Softmax Gated Attention (V9)

6-signal composite with dynamic weighting via softmax (τ=0.5) and CUSUM change-point detection:

```text
EWMA_t = α · reward_t + (1 - α) · EWMA_{t-1}
α = max(2/(n+1), α_floor)    — budget-aware floor

Softmax Gated Attention (V9):
  w_i = prior_i × exp(|x_i - μ| / τ)  — signals deviating from mean get higher weight
  reward = Σ(normalize(w_i) × x_i)

CUSUM change-point detection (V9):
  S_t = max(0, S_{t-1} + (x_t - μ) - k)  — cumulative sum, k=0.05, h=0.15
  Change detected when S_t > h (lag-free, replaces MACD)
```

- **Reward history**: Capped at 20 entries (VecDeque ring buffer)
- **Stagnation detection**: 3+ steps with <2% EWMA change
- **Fatigue detection**: 3+ consecutive quality drops
- **CUSUM collapse detection (V9)**: Lag-free quality degradation tracking
- **Process Advantage Verifier (PAV)**: Measures advantage over baseline
- **Reward Consistency (N3)**: PRM↔EWMA divergence >0.4 → warning (Z≈1.9, P≈2.9%)

---

## Graph-of-Thought (GoT)

DAG topology tracking across reasoning chains:

- **Nodes**: Each thought step becomes a node
- **Edges**: Sequential and revision dependencies
- **Longest path**: Kahn's topological sort + dynamic programming (convergence-safe)
- **Cycle detection**: Tarjan's SCC algorithm O(V+E) detects circular reasoning

---

## MCTS Graph — PUCT + Adaptive UCT

Arena-allocated MCTS with PUCT scoring (superior to standard UCB1):

```text
PUCT(s,a) = Q(s,a) + c_puct × P(s,a) × √(N_parent) / (1 + N_child)
c_puct adapts based on variance of child Q-values
```

---

## Persistent Thought Sessions

Sessions maintain state across multiple MCP tool calls sharing the same hypothesis:

- **EWMA accumulation**: Quality tracking persists across calls
- **Novelty tracking**: Vocabulary grows across calls
- **Graph-of-Thought**: DAG builds across calls
- **Trend indicator**: ↗️ Improving, → Stable, ↘️ Declining
- **Hypothesis drift (G11)**: Semantic distance from original hypothesis
- **Root-anchoring**: Combined drift detection (hypothesis + first thought)
- **Confidence oscillation**: Detects rapidly alternating confidence (>3 sign changes in 5 readings)
- **Depth degradation**: Tracks quality.depth history, detects >50% drop vs baseline
- **Epistemological rollback**: Snapshot/rollback when MCTS rejects a thought
- **Mode collapse guard**: Stores rejected thoughts, detects paraphrasing via Jaccard > 0.6
- **Auto-expire**: TTL 600s to prevent memory leaks

---

## Sandbox Security

Multi-layered Python sandbox via PyO3:

| Layer | Protection |
|-------|------------|
| **AST Scanner** | Blocks dangerous imports at parse time (os.system, subprocess, etc.) |
| **Nesting Depth** | Max 100 bracket depth — prevents CPython C-stack exhaustion |
| **Memory Limit** | RLIMIT_AS 512MB — prevents OOM bombs |
| **Recursion Limit** | 100 — prevents stack overflow |
| **Concurrency** | Semaphore max 2 concurrent executions |
| **Timeout** | 5 seconds max |
| **Code Size** | 50KB max input |
| **PEP 578 Audit Hooks** | Blocks OS-level events at runtime |
| **ReDoS Guard** | Backreference length limit on re.compile |
| **Z3 Vacuous Truth** | Detects trivial assertions in Z3 solver outputs |

---

## Stage-Content Alignment (V8)

Validates that thought content actually matches its declared cognitive stage:

- **6 keyword pattern sets**: Bilingual (English/Spanish) for each stage
- **Alignment score**: Ratio of declared vs detected stage patterns
- **Logical Validity Score**: 4 dimensions — Premise reference (30%), Conclusion support (30%), Backward reference (20%), No logic gaps (20%)
- **Warning**: Fires when alignment < 0.5 with >2 detected patterns

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

| Stage | Trigger | Action |
|-------|---------|--------|
| **DEFINE** (thought ≤ 2) | Problem definition | `cuba_faro` search + `cuba_expediente` error check |
| **HYPOTHESIZE** | New hypothesis | `cuba_expediente` anti-repetition guard |
| **VERIFY** | Claim verification | `cuba_faro(mode:"verify")` grounding check |
| **SYNTHESIZE** (final) | Conclusion | `cuba_cronica` lesson consolidation |

Analogous to the cortex-hippocampus consolidation cycle (McClelland et al., 1995).

---

## Budget Modes

| Mode | EWMA α Floor | MCTS Threshold | Quality Gate | Max Thoughts | Length Penalty |
|------|:------------:|:--------------:|:------------:|:------------:|:--------------:|
| ⚡ `fast` | 0.30 | 50% | 0.30 | 5 | 80 words |
| ⚖️ `balanced` | 0.25 | 40% | 0.25 | 10 | 150 words |
| 🔎 `thorough` | 0.20 | 35% | 0.20 | 20 | 250 words |
| 🔬 `exhaustive` | 0.15 | 30% | 0.15 | 50 | 400 words |

---

## Architecture

```text
cuba-thinking/
└── cuba_cognitive_engine/
    ├── Cargo.toml
    └── src/
        ├── main.rs                          # Entry point
        ├── server/
        │   ├── mcp_protocol.rs              # JSON-RPC 2.0, tool dispatch, progress streaming
        │   └── observability.rs             # Prometheus metrics, tool timing
        └── engine/
            ├── mod.rs                       # Module registry
            │
            ├── ── Cognitive Core ──
            ├── stage_engine.rs              # 6-stage state machine (Bloom's Taxonomy)
            ├── quality_metrics.rs           # 6D quality + Shannon entropy + LZ76 + Root-TTR
            ├── ewma_reward.rs               # Softmax Gated Attention + CUSUM + PAV
            ├── budget.rs                    # 4 budget modes with optimal stopping (Wald 1945)
            ├── anti_hallucination.rs         # 9-layer trust verification + Pareto L2 norm
            ├── bias_detector.rs             # 5 cognitive bias detectors (Kahneman & Tversky)
            ├── metacognition.rs             # Filler, CWR (Coh-Metrix), fallacies, dialectics
            ├── thought_graph.rs             # GoT DAG + Kahn DP + Tarjan SCC O(V+E)
            ├── memory_bridge.rs             # Cross-MCP memory symbiosis + anti-repetition
            ├── formatter.rs                 # Output formatting
            │
            ├── ── Semantics ──
            ├── semantic_similarity.rs       # TF-IDF cosine coherence
            ├── contradiction_detector.rs    # Negation, 30+ antonyms, quantifiers
            ├── novelty_tracker.rs           # Shannon information gain (Jaccard distance)
            ├── claim_grounding.rs           # ROSCOE faithfulness + specificity
            │
            ├── ── Deep Reasoning ──
            ├── thought_session.rs           # Persistent sessions + rollback + trends + drift
            ├── corrective_directives.rs     # 3-level prescriptive corrections (SEAL/FS-C)
            ├── stage_validator.rs           # Stage alignment + Logical Validity (ReasonEval 2024)
            │
            ├── ── Execution ──
            ├── micro_prm.rs                 # 8-signal PRM + Veto Gate (V9)
            ├── sandbox.rs                   # PyO3 sandbox + AST + RLIMIT + PEP 578
            ├── mcts_graph.rs                # PUCT + Adaptive UCT (arena-allocated)
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
| `bumpalo` | Arena allocation for MCTS graph |

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
| Circular Reasoning | Tarjan SCC finds cycle in GoT |
| Depth Degradation | quality.depth drops >50% vs baseline |
| Confidence Oscillation | >3 sign changes in 5 readings |
| Hypothesis Drift | Semantic distance from original > threshold |
| Mode Collapse | Jaccard similarity > 0.6 with rejected thought |
| Reward Divergence | PRM↔EWMA divergence > 40pp (Z≈1.9) |
| Kinematic Collapse | CUSUM detects accelerating quality decline |
| Stage Mismatch | Declared vs detected stage disagree |
| Memory Instructions | DEFINE/HYPOTHESIZE/VERIFY/SYNTHESIZE stages |

---

## Mathematical Verification

Every formula verified with unit tests and formal analysis:

| # | Formula | Description | Verified |
|---|---------|-------------|:--------:|
| 1 | EWMA α = max(2/(n+1), α_floor) | Adaptive smoothing (Roberts 1959) | ✅ |
| 2 | Softmax w_i = p_i × exp(\|x_i-μ\|/τ) | Gated Attention (Zhou 2025) | ✅ |
| 3 | CUSUM S_t = max(0, S_{t-1} + x_t - μ - k) | Change-point detection | ✅ |
| 4 | PRM Gate = (E1×0.7 + E4×0.3).max(0.05) | Veto Gate | ✅ |
| 5 | Composite reward (6 signals) | Weights sum to 1.0 (3 stage profiles) | ✅ |
| 6 | PRM composite (8 signals) | Weights sum to 1.0 + Veto Gate | ✅ |
| 7 | Trust score (5 components) | Weights sum to 1.0 | ✅ |
| 8 | TF-IDF cosine similarity | Coherence scoring (Salton 1975) | ✅ |
| 9 | Jaccard \|A∩B\|/\|A∪B\| | Novelty + mode collapse detection | ✅ |
| 10 | TTR / √N | Root-TTR normalization | ✅ |
| 11 | Tarjan SCC O(V+E) | Cycle detection correctness | ✅ |
| 12 | Kahn's DP topological sort | DAG longest path (convergence-safe) | ✅ |
| 13 | Pareto L2 norm d²+c² > 1.65 | Gaming detection (Z≈1.645, 90th pctl) | ✅ |
| 14 | Sigmoid P(reject)=1/(1+e^(20d)) | Hedged MCTS rejection | ✅ |
| 15 | Shannon H(X) + LZ76 | Information density | ✅ |
| 16 | \|PRM-EWMA\| > 0.4 | Reward consistency (Z≈1.9, P≈2.9%) | ✅ |

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
| 13 | Wald (1945). "Sequential Analysis" | Evidence accumulation, Optimal Stopping |
| 14 | Shannon (1948). "Mathematical Theory of Communication" | Information gain, entropy |
| 15 | McClelland et al. (1995). "Complementary Learning Systems" | Memory symbiosis |
| 16 | DeepSeek (2025). "Thoughtology" | Anti-overthinking |
| 17 | Zangari (1994). "EWMA for Risk Management" | Adaptive alpha floor |
| 18 | Salton (1975). "Vector Space Model" | TF-IDF cosine similarity |
| 19 | Tarjan (1972). "Depth-First Search and Linear Graph Algorithms" | SCC cycle detection |
| 20 | Lempel & Ziv (1976). "On the Complexity of Finite Sequences" | LZ76 complexity metric |
| 21 | Press et al. (2022). "Train Short, Test Long" | Depth degradation |
| 22 | PEP 578 (2019). "Python Runtime Audit Hooks" | Sandbox security layer |
| 23 | Cilibrasi & Vitányi (2005). "Clustering by Compression" — IEEE TIT | Mode collapse detection |
| 24 | Kahn (1962). "Topological Sorting of Large Networks" — CACM | DAG longest path |
| 25 | Besta et al. (2024). "Graph of Thoughts" — ETH Zurich | GoT DAG topology |
| 26 | Greenblatt et al. (2024). "Sycophancy to Subterfuge" — arXiv:2406.10162 | Reward gaming detection |
| 27 | Sun et al. (2024). "Diagram of Thought" — arXiv:2409.10038 | DAG formalization |
| 28 | Zhou et al. (2025). "Fin-PRM" | Softmax Gated Attention |
| 29 | Page (1954). "Continuous Inspection Schemes" | CUSUM change-point detection |
| 30 | Rosin & Silver (2011). "PUCT Algorithm" | MCTS exploration policy |

---

## License

[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) — Free to use and modify, **not for commercial use**.

---

## Author

**Leandro Pérez G.**

- GitHub: [@LeandroPG19](https://github.com/LeandroPG19)
- Email: [leandropatodo@gmail.com](mailto:leandropatodo@gmail.com)
