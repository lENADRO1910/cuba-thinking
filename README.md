# 🧠 Cuba-Thinking

**Advanced sequential thinking for AI agents** — A Model Context Protocol (MCP) server that enhances AI reasoning with a 6-stage cognitive engine, semantic embeddings, anti-hallucination, graph-of-thought, NLI contradiction detection, MCTS quality enforcement, and cross-MCP memory symbiosis.

1 tool. Zero configuration. Mathematically verified.

---

## Why Cuba-Thinking?

AI agents think in flat, unstructured sequences. Cuba-Thinking gives them:

- **A cognitive engine** — 6-stage state machine (Bloom's Taxonomy) that guides thinking from DEFINE → SYNTHESIZE
- **Semantic embeddings** — Local BGE-small-en-v1.5 (384d) for thought similarity, stagnation, and contradiction detection
- **6D quality metrics** — TTR clarity, clause depth, structural logic, noun breadth, semantic relevance, and concrete actionability
- **Anti-hallucination** — Assumption tracking, NLI-verified contradiction detection, confidence calibration, Chain-of-Verification
- **NLI Cross-Encoder** — DeBERTa-v3-xsmall (22M params) for semantic contradiction detection that negation-counting misses
- **MCTS Forced Backtracking** — Protocol-level quality enforcement that rejects thoughts when EWMA drops below 40%
- **Memory Symbiosis** — Cross-MCP bridge to cuba-memorys via formatted recall/consolidation instructions
- **Metacognitive analysis** — Filler detection, claim density scoring, fallacy detection, dialectical reasoning checks
- **Bias detection** — Identifies 5 cognitive biases with actionable suggestions
- **Graph-of-Thought** — DAG edge registry with topology analysis (orphan detection, linearity ratio)
- **Anti-overthinking** — EWMA stagnation detection, early stopping signals, fatigue monitoring

| Feature | Cuba-Thinking | Basic Thinking MCPs |
|---------|:------------:|:-------------------:|
| 6-stage cognitive engine (Bloom's) | ✅ | ❌ |
| Semantic embeddings (BGE-384d neural) | ✅ | ❌ |
| 6D quality metrics + EWMA reward | ✅ | 4D or less |
| NLI contradiction detection (DeBERTa) | ✅ | ❌ |
| MCTS forced backtracking (isError) | ✅ | ❌ |
| Cross-MCP memory symbiosis | ✅ | ❌ |
| TTR clarity (Templin 1957) | ✅ | ❌ |
| Clause depth analysis (Hunt 1965) | ✅ | ❌ |
| Structural logic scoring (ROSCOE) | ✅ | ❌ |
| Claim density scoring | ✅ | ❌ |
| Metacognitive filler detection | ✅ | ❌ |
| Fallacy detection (hasty generalization) | ✅ | ❌ |
| Dialectical reasoning check | ✅ | ❌ |
| Confidence variance tracking (Shewhart) | ✅ | ❌ |
| Shannon Entropy stability | ✅ | ❌ |
| Graph-of-Thought with topology analysis | ✅ | ❌ |
| Chain-of-Verification (CoVe) | ✅ | ❌ |
| Anti-overthinking + early stopping | ✅ | ❌ |
| Fatigue monitoring | ✅ | ❌ |
| Assumption tracking + dedup | ✅ | ❌ |
| Confidence calibration per stage | ✅ | ❌ |
| 5-bias detector | ✅ | ❌ |
| Stagnation detection | ✅ | ❌ |
| Reasoning type classification | ✅ | ❌ |
| Graceful degradation | ✅ | ❌ |
| Dependencies | **3** | 13+ |

---

## Quick Start

### 1. Prerequisites

- **Node.js 18+**

### 2. Install

```bash
git clone https://github.com/lENADRO1910/cuba-thinking.git
cd cuba-thinking
npm install
npm run build
```

### 3. Configure your AI editor

Add to your MCP configuration (e.g., `mcp_config.json`):

```json
{
  "mcpServers": {
    "cuba-thinking": {
      "command": "node",
      "args": ["/path/to/cuba-thinking/dist/index.js"]
    }
  }
}
```

Zero environment variables. Zero configuration. It just works.

---

## The Tool

### `cuba_thinking`

**Required parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `thought` | string | Your current thinking step |
| `thoughtNumber` | number | Current thought number in the sequence |
| `totalThoughts` | number | Estimated total thoughts needed (adjustable) |
| `nextThoughtNeeded` | boolean | Whether another thought step is needed |

**Optional parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `thinkingStage` | string | `DEFINE`, `RESEARCH`, `ANALYZE`, `HYPOTHESIZE`, `VERIFY`, `SYNTHESIZE` (auto-detected if omitted) |
| `confidence` | number | 0.0–1.0, calibrated against stage expectations |
| `qualityMetrics` | object | Manual quality overrides (0–5 per dimension) |
| `assumptions` | string[] | Tracked and deduplicated across thoughts |
| `hypothesis` | string | Current hypothesis being tested |
| `isRevision` | boolean | Whether this revises a previous thought |
| `revisesThought` | number | Which thought is being revised |
| `branchFromThought` | number | Branching point for parallel exploration |
| `branchId` | string | Identifier for parallel reasoning paths |
| `parentThoughts` | number[] | Multiple parent thought references for GoT merge operations |
| `needsMoreThoughts` | boolean | Extend beyond initial estimate |
| `budgetMode` | string | `fast`, `balanced`, `thorough`, `exhaustive` |
| `budgetUsed` | number | Budget consumed (0–100%) |
| `biasDetected` | string | Agent-reported bias type |

---

## The 6 Cognitive Stages

Based on Bloom's Revised Taxonomy (Anderson & Krathwohl, 2001):

```
DEFINE ──→ RESEARCH ──→ ANALYZE ──→ HYPOTHESIZE ──→ VERIFY ──→ SYNTHESIZE
  📋          🔍          🔬           💡             ✅          🎯
```

| Stage | Focus | Confidence Range |
|-------|-------|:----------------:|
| DEFINE | Clarify scope, requirements | 0.30 – 0.60 |
| RESEARCH | Explore options, gather data | 0.30 – 0.70 |
| ANALYZE | Evaluate trade-offs, compare | 0.40 – 0.80 |
| HYPOTHESIZE | Propose solutions, predict | 0.50 – 0.85 |
| VERIFY | Test, validate, confirm | 0.60 – 0.90 |
| SYNTHESIZE | Conclude, recommend, summarize | 0.70 – 0.95 |

Each stage boosts different quality dimensions. DEFINE boosts **Clarity** (3×), ANALYZE boosts **Depth** (3×), SYNTHESIZE boosts **Actionability** (3×).

---

## MCTS Forced Backtracking

When the EWMA step reward drops below **40%** (after thought #3), the MCP tool call is **rejected at the protocol level** with `isError: true`. This is not a suggestion — it's an enforcement mechanism that forces the LLM to backtrack.

```
EWMA_reward < 0.40 AND thoughtNumber > 3
  → isError: true
  → Rollback to best historical thought
  → LLM MUST branch from that thought
```

The system identifies the thought with the highest quality score in the session history and instructs the agent to branch from it using a completely different reasoning path:

```
⛔ MCTS BACKTRACK — EWMA Reward 39% < 40% threshold
Thought #12 REJECTED at protocol level.
Rollback to thought #2 (quality: 75%).
You MUST branch with: branchFromThought: 2
```

This is based on Monte Carlo Tree Search (Coulom, 2006) applied to reasoning quality: the system prunes low-reward subtrees and forces exploration of high-reward branches.

---

## NLI Cross-Encoder — DeBERTa-v3-xsmall

A two-stage contradiction detection pipeline that catches implicit semantic contradictions that negation word counting misses:

```
Stage 1: Cosine similarity > 0.6?     (~1ms, embedding-based)
  ↓ Yes
Stage 2: DeBERTa NLI classification   (~200ms, cross-encoder)
  ↓ contradiction score > 0.85
  → NLI-verified contradiction
```

The model is `Xenova/nli-deberta-v3-xsmall` (22M parameters, ONNX quantized q8), trained on SNLI + MultiNLI (~1M sentence pairs). It runs locally with zero API calls.

Example output:

```
🔴 NLI-verified contradiction between thought #8 and #9 (NLI: 86%, semantic: 64%)
🔴 NLI-verified contradiction between thought #8 and #10 (NLI: 91%, semantic: 62%)
```

In both cases, the semantic similarity (64%, 62%) was below the negation detection threshold — only the NLI cross-encoder caught these contradictions.

If the NLI model fails to load, the system falls back to negation polarity detection.

---

## Cortex-Hippocampus Symbiosis

Cross-MCP memory bridge between cuba-thinking and [cuba-memorys](https://github.com/lENADRO1910/cuba-memorys). Since MCPs cannot call each other directly, the symbiosis works through formatted instructions in the tool output that guide the LLM:

| Stage | Trigger | Injected Instruction |
|-------|---------|---------------------|
| **DEFINE** (thought ≤ 2) | Problem definition | `cuba_faro(query:...)` — search past knowledge |
| | | `cuba_expediente(query:...)` — check past errors |
| **SYNTHESIZE** (!nextThought) | Conclusion | `cuba_cronica(action:"add", ...)` — consolidate lesson |

This creates a cognitive loop: **recall before reasoning, consolidate after conclusion** — analogous to the cortex-hippocampus consolidation cycle in neuroscience (McClelland et al., 1995).

---

## 6D Quality Metrics

Each dimension uses empirically validated linguistic measures:

| Dimension | Method | Basis |
|-----------|--------|-------|
| **Clarity** | Type-Token Ratio (unique/total words) + sentence diversity | Templin (1957) |
| **Depth** | Subordinate clause counting + causal keyword density | Hunt (1965) |
| **Breadth** | Unique noun ratio + topic diversity markers | Lexical diversity |
| **Logic** | Connective type diversity + conditional chain depth + conclusion presence | ROSCOE (Golovneva et al., 2023) |
| **Relevance** | Cosine similarity to first thought (embedding) or keyword fallback | Salton (1975) |
| **Actionability** | Imperative verbs + units/measurements + specificity vs. vagueness | GRACE (Guan et al., 2024) |

### EWMA Step Reward — Roberts (1959)

```
EWMA_t = α_n · reward_t + (1 - α_n) · EWMA_{t-1}
α_n = 2 / (n + 1)        — adaptive smoothing
reward = 0.6·quality + 0.3·coherence + 0.1·(1 - contradictions/t)
```

Adaptive α reduces sensitivity to noise as the session progresses while maintaining fast initial responsiveness.

### Shannon Entropy Stability — Shannon (1948)

```
H = -Σ pᵢ·log₂(pᵢ)   where pᵢ = scoreᵢ / Σscores
stability = H / H_max  where H_max = log₂(6)
```

Stability < 0.60 triggers a warning that reasoning is lopsided.

### OLS Trend Analysis — Gauss (1795)

```
slope = (n·Σxy − Σx·Σy) / (n·Σx² − (Σx)²)
```

`slope > 0.02` → 📈 improving, `< -0.02` → 📉 declining.

---

## Anti-Hallucination

Six verification layers that require zero LLM calls:

### 1. Assumption Tracking

Accumulates and deduplicates assumptions across all thoughts. Semantic deduplication when embeddings are available (cosine > 0.85 → duplicate), keyword fallback otherwise.

### 2. Contradiction Detection — Two-Stage Pipeline

**Stage 1: Cosine + Negation** — Compares each new thought against all previous thoughts for semantic similarity combined with negation polarity analysis:

```
contradiction = similarity(A, B) > 0.6
                AND |negations(A) - negations(B)| ≥ 2
```

**Stage 2: NLI Cross-Encoder** — When cosine similarity exceeds 0.6, the DeBERTa-v3-xsmall model classifies the pair. Contradiction score > 0.85 = verified contradiction.

### 3. Confidence Calibration

Flags overconfidence (high confidence in early stages) and underconfidence (low confidence in late stages) with per-stage expected ranges.

### 4. Chain-of-Verification (CoVe) — Dhuliawala et al. (2023)

At critical stage transitions, generates targeted verification questions:
- **Quantitative assumptions** (containing numbers/percentages): "What measurement confirms: ...?"
- **Qualitative assumptions**: "What evidence supports: ...?"

### 5. Claim Density Scoring

Counts verifiable assertions per sentence (percentages, large numbers, absolutes, causal claims). High density signals text that needs more verification.

### 6. MCTS Quality Enforcement

EWMA reward < 40% after 3+ thoughts → tool call rejected with `isError: true` at the MCP protocol level. The LLM is forced to backtrack to the highest-quality historical thought.

---

## Metacognitive Analysis

### Metacognitive Filler Detection — Flavell (1979)

Identifies "thinking about thinking" patterns that consume tokens without substance:

```
filler_ratio = filler_words / total_words
```

Patterns: "let me think", "well", "hmm", "I'm not sure", "maybe I should". Warning triggers at > 30% filler ratio.

### Fallacy Detection

Detects hasty generalization: absolute claims (`always`, `never`, `every`, `all`) near singular evidence (`one`, `single`, `this example`).

### Dialectical Reasoning — Stage-Aware

In VERIFY and SYNTHESIZE stages, checks for counter-argument markers (`however`, `on the other hand`, `admittedly`, `despite`). Absence triggers a warning to consider opposing viewpoints before finalizing conclusions.

### Reasoning Type Classification

Classifies dominant reasoning pattern (deductive, inductive, or abductive) and provides actionable feedback when reasoning is imbalanced.

---

## Graph-of-Thought (GoT-lite) — Besta et al. (2024)

Tracks reasoning structure as a directed acyclic graph:

| Edge Type | Created By | Meaning |
|-----------|-----------|---------|
| `extends` | `branchFromThought` | Thought branches from another |
| `revises` | `revisesThought` | Thought revises a previous one |
| `merges` | `parentThoughts[]` | Thought merges multiple parents |

Graph coherence is computed as the average similarity across all edges:

```
coherence = (1/|E|) · Σ sim(thought_u, thought_v)
```

### Topology Analysis

- **Orphan detection**: Counts thoughts with no incoming or outgoing edges
- **Linearity ratio**: `unique_nodes_with_edges / total_thoughts`. Low ratio indicates unexplored branches

---

## Confidence Variance — Shewhart (1931)

Tracks standard deviation of confidence values across the session:

```
σ = sqrt(Σ(x_i - μ)² / n)
```

σ > 0.25 triggers a stability warning — large confidence swings indicate the agent is oscillating rather than converging.

---

## Anti-Overthinking — DeepSeek (2025)

Based on DeepSeek's "Thoughtology" research: reasoning quality follows an inverted-U curve.

```
stagnation = true if EWMA_diff < 2% for 3+ consecutive thoughts
```

### Early Stopping Signal

When quality > 0.7 and progress > 70%, suggests concluding. In `fast` budgetMode, auto-reduces `totalThoughts`.

---

## Fatigue Detection

Monitors consecutive quality drops:

| Consecutive Drops | Suggested Action |
|:-----------------:|:----------------:|
| < 3 | `continue` |
| 3–4 | `step_back` |
| ≥ 5 | `conclude` |

---

## Bias Detection — Kahneman & Tversky (1974)

Identifies 5 cognitive biases with actionable suggestions:

| Bias | Detection Method | Trigger |
|------|:---------------:|---------| 
| **Confirmation** | History similarity > 0.7 | Repeatedly reinforcing same conclusion |
| **Anchoring** | First quantitative reference dominates | Over-reliance on initial data point |
| **Availability** | Recency weighting of examples | Using recent/memorable examples disproportionately |
| **Overconfidence** | High confidence early in reasoning | Confidence > 0.8 before 50% progress |
| **Sunk Cost** | Late-stage reluctance to change | Keywords like "already invested" after 70% progress |

---

## Silent by Default

All features follow the **silent by default** principle — they only appear when they detect actionable conditions:

| Feature | Only Appears When |
|---------|------------------|
| Shannon Stability | < 60% (unbalanced reasoning) |
| EWMA Reward | Always (core quality metric) |
| Claim Density | Claims detected in text |
| Metacognition Warning | > 30% filler ratio |
| Fallacy Warning | Hasty generalization detected |
| Dialectical Warning | VERIFY/SYNTHESIZE without counter-arguments |
| Confidence Variance | σ > 0.25 |
| Anti-Overthinking | 3+ stagnant thoughts |
| Early Stopping | Quality > 0.7 and progress > 70% |
| CoVe Checkpoint | Stage transitions with open assumptions |
| Fatigue | 3+ consecutive quality drops |
| Graph | When edges exist |
| Topology Orphans | Orphan thoughts detected |
| MCTS Backtracking | EWMA < 40% after thought #3 |
| Memory Recall | DEFINE stage, thought ≤ 2 |
| Memory Consolidation | SYNTHESIZE stage, reasoning complete |
| NLI Contradiction | Cross-encoder score > 0.85 |

---

## Test Results

```
Test Suites: 9 passed, 9 total
Tests:       246 passed, 246 total
Failures:    0
```

### Coverage

| File | Stmts | Branch | Funcs | Lines |
|------|:-----:|:------:|:-----:|:-----:|
| **All files** | **91.47%** | **83.51%** | **96.87%** | **92.9%** |
| formatter.ts | 100% | 100% | 100% | 100% |
| types.ts | 100% | 100% | 100% | 100% |
| stage-engine.service.ts | 100% | 100% | 100% | 100% |
| bias-detector.service.ts | 100% | 100% | 100% | 100% |
| quality-metrics.service.ts | 95.95% | 86.25% | 100% | 99.53% |
| anti-hallucination.service.ts | 97.36% | 92.85% | 100% | 100% |
| cognitive-processor.ts | 86.5% | 64.4% | 80% | 86.4% |
| embedding.service.ts | 66.66% | 62.5% | 93.75% | 69.76% |
| nli.service.ts | 71.42% | 57.14% | 100% | 71.05% |

> **Note**: embedding/nli services depend on ONNX model loading. Uncovered lines are hardware-dependent model inference paths that require the actual model binaries.

### Coverage by Category

| Category | Tests | Coverage |
|----------|:-----:|----------|
| Quality Metrics (6D + EWMA + Shannon) | 42 | TTR, clause depth, structural logic, actionability, EWMA reward, entropy stability |
| Anti-Hallucination (contradictions + CoVe) | 28 | Assumption tracking, negation polarity, confidence calibration, CoVe questions |
| Cognitive Processor (orchestration) | 35 | Full pipeline integration, graph edges, fatigue, overthinking |
| Stage Engine (6-stage FSM) | 24 | Auto-detection, transitions, weights, confidence ranges |
| Embedding Service (BGE + fallback) | 18 | Cosine similarity, keyword fallback, cache, graceful degradation |
| Bias Detector (5 biases) | 12 | Confirmation, anchoring, availability, overconfidence, sunk cost |
| Formatter (response rendering) | 48 | All 14 render functions, stage/trend icons, all sections, bar() edges |
| Coverage Boost (edge cases) | 32 | NLI mocks, model fallbacks, deep branches |

### Nemesis Protocol (Adversarial Testing)

| Level | Tests | Description |
|-------|:-----:|-------------|
| 🟢 Normal | 25 | Valid inputs, happy paths |
| 🟡 Pessimistic | 14 | Empty strings, undefined, single words, no edges |
| 🔴 Extreme | 12 | Unicode attacks, SQL injection, XSS payloads, 5000-repeat strings, path traversal |

**Key invariant**: All quality scores stay in [0, 1] range for ALL inputs including adversarial payloads.

### Live Validation (32/32 Features)

All features validated in a 13-thought live session:

| Feature Category | Validated |
|-----------------|:---------:|
| All 6 cognitive stages | ✅ |
| 6D quality metrics + EWMA + trend | ✅ |
| NLI cross-encoder (86%, 91% confidence) | ✅ |
| MCTS backtracking (EWMA 39% → isError) | ✅ |
| Memory Recall (DEFINE) | ✅ |
| Memory Consolidation (SYNTHESIZE) | ✅ |
| Assumptions (4 tracked, deduplicated) | ✅ |
| Confidence calibration (under/over) | ✅ |
| Bias detection (overconfidence, sunk cost) | ✅ |
| GoT graph (branch, revise, merge) | ✅ |
| Contradictions (negation + NLI) | ✅ |
| CoVe verification checkpoint | ✅ |
| Metacognition (43%–100% filler) | ✅ |
| Stagnation + early stopping | ✅ |
| Dialectical reasoning warning | ✅ |
| Fallacy detection | ✅ |

---

## Architecture

```
cuba-thinking/
├── package.json
├── tsconfig.json
├── jest.config.mjs
└── src/
    ├── index.ts                          # MCP server + MCTS backtracking
    ├── types.ts                          # Zod schemas + TypeScript interfaces
    ├── formatter.ts                      # Response rendering + memory symbiosis (14 renderers)
    ├── __tests__/
    │   ├── anti-hallucination.test.ts     # 28 tests
    │   ├── bias-detector.test.ts          # 12 tests
    │   ├── cognitive-processor.test.ts    # 35 tests
    │   ├── coverage-boost.test.ts         # 32 tests (edge cases)
    │   ├── embedding.test.ts             # 18 tests
    │   ├── formatter.test.ts             # 48 tests
    │   ├── quality-metrics.test.ts        # 42 tests
    │   ├── stage-engine.test.ts           # 24 tests
    │   └── v2-nemesis.test.ts            # 7 tests (adversarial)
    └── services/
        ├── cognitive-processor.ts        # Central orchestrator (6 phases, CC~7)
        ├── embedding.service.ts          # BGE-384d + keyword fallback
        ├── nli.service.ts                # DeBERTa NLI cross-encoder
        ├── stage-engine.service.ts       # 6-stage FSM
        ├── quality-metrics.service.ts    # 6D + EWMA + metacognitive analysis
        ├── anti-hallucination.service.ts # 6-layer verification + NLI pipeline
        ├── bias-detector.service.ts      # 5-bias detection
        └── transformers-loader.ts        # Shared HuggingFace module loader
```

### Dependencies (3 total)

| Package | Purpose |
|---------|---------|
| `@modelcontextprotocol/sdk` | MCP protocol server |
| `@huggingface/transformers` | Local BGE embeddings + NLI cross-encoder (lazy init) |
| `zod` | Input validation |

### Graceful Degradation

If the embedding model fails to load, Cuba-Thinking automatically falls back to keyword-based cosine similarity. If the NLI model fails to load, contradiction detection falls back to negation polarity only. All features continue working — model-dependent features degrade to heuristic equivalents.

---

## Mathematical Verification

Every formula is verified with Wolfram Alpha against analytical solutions:

| Formula | Input | Expected | Result |
|---------|-------|----------|--------|
| Shannon Entropy | `{0.6, 0.55, 0.4, 0.57, 1.0, 0.4}` | `H = 2.5077, stability = 0.9701` | ✅ |
| EWMA decay (α=0.3) | 5-step chain | `60% → 53% → 49% → 42% → 36% → 28%` | ✅ |
| Cosine similarity | `[1,2,3]·[4,5,6]` | `32/√1078 ≈ 0.9746` | ✅ |
| OLS slope | `y = 0.3 + 0.1x` | `slope = 0.1` | ✅ |
| Weighted mean (DEFINE) | `[0.8, 0.5, 0.4, 0.6, 0.7, 0.3]` | `0.6222` | ✅ |

---

## Part of the Cuba Ecosystem

| Project | Purpose |
|---------|---------|
| [Cuba-Memorys](https://github.com/lENADRO1910/cuba-memorys) | Persistent memory — knowledge graph, Hebbian learning, anti-hallucination grounding |
| **Cuba-Thinking** | Sequential reasoning — cognitive engine, quality metrics, NLI contradictions, MCTS enforcement, memory symbiosis |

Together, they give AI agents **memory + reasoning** — the two fundamental capabilities for reliable AI assistance. The Cortex-Hippocampus symbiosis enables bidirectional communication: cuba-thinking recalls from cuba-memorys before reasoning, and consolidates conclusions back after synthesis.

---

## Academic References

| # | Citation | Used For |
|---|----------|----------|
| 1 | Shannon (1948). "A Mathematical Theory of Communication" | Entropy stability |
| 2 | Besta et al. (2024). "Graph of Thoughts" — ETH Zurich | DAG structure + topology |
| 3 | Dhuliawala et al. (2023). "CoVe Reduces Hallucination" — Meta AI | Verification questions |
| 4 | Lightman et al. (2023). "Let's Verify Step by Step" — OpenAI | Step reward (EWMA) |
| 5 | DeepSeek (2025). "Thoughtology" | Anti-overthinking + early stopping |
| 6 | Flavell (1979). "Metacognition and Cognitive Monitoring" | Metacognitive filler detection |
| 7 | Roberts (1959). "EWMA Control Charts" | Adaptive EWMA smoothing |
| 8 | Anderson & Krathwohl (2001). "Revised Bloom's Taxonomy" | Cognitive stages |
| 9 | Salton (1975). "Vector Space Model" | Cosine similarity |
| 10 | Gauss (1795). "Method of Least Squares" | OLS trend analysis |
| 11 | Kahneman & Tversky (1974). "Judgment Under Uncertainty" | Bias detection |
| 12 | Templin (1957). "Certain Language Skills in Children" | TTR clarity metric |
| 13 | Hunt (1965). "Grammatical Structures" | Clause depth analysis |
| 14 | Golovneva et al. (2023). "ROSCOE: Reasoning Scores" | Structural logic evaluation |
| 15 | Guan et al. (2024). "GRACE: Generative Reasoning Assessment" | Actionability scoring |
| 16 | Shewhart (1931). "Economic Control of Quality" | Confidence variance |
| 17 | Coulom (2006). "Efficient Selectivity and Backup Operators in MCTS" | Forced backtracking |
| 18 | He et al. (2021). "DeBERTa: Decoding-enhanced BERT with Disentangled Attention" | NLI cross-encoder |
| 19 | McClelland et al. (1995). "Complementary Learning Systems" | Memory symbiosis |

---

## License

[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) — Free to use and modify, **not for commercial use**.

---

## Author

**Leandro Pérez G.**

- GitHub: [@lENADRO1910](https://github.com/lENADRO1910)
- Email: [leandropatodo@gmail.com](mailto:leandropatodo@gmail.com)
