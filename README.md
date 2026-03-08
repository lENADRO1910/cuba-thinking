# 🧠 Cuba-Thinking

**Advanced sequential thinking for AI agents** — A Model Context Protocol (MCP) server that enhances AI reasoning with a 6-stage cognitive engine, semantic embeddings, anti-hallucination, and bias detection.

1 tool. Zero configuration. Mathematically verified.

---

## Why Cuba-Thinking?

AI agents think in flat, unstructured sequences. Cuba-Thinking gives them:

- **A cognitive engine** — 6-stage state machine (Bloom's Taxonomy) that guides thinking from DEFINE → SYNTHESIZE
- **Semantic embeddings** — Local BGE-small-en-v1.5 (384d) for thought similarity and stagnation detection
- **6D quality metrics** — Clarity, Depth, Breadth, Logic, Relevance, Actionability with OLS trend analysis
- **Anti-hallucination** — Assumption tracking, contradiction detection, confidence calibration
- **Bias detection** — Identifies 5 cognitive biases with actionable suggestions in English

| Feature | Cuba-Thinking | Basic Thinking MCPs |
|---------|:------------:|:-------------------:|
| 6-stage cognitive engine (Bloom's) | ✅ | ❌ |
| Semantic embeddings (BGE-384d neural) | ✅ | ❌ |
| 6D quality metrics | ✅ | 4D or less |
| OLS trend analysis | ✅ | ❌ |
| Assumption tracking + dedup | ✅ | ❌ |
| Contradiction detection | ✅ | ❌ |
| Confidence calibration per stage | ✅ | ❌ |
| 5-bias detector | ✅ | ❌ |
| Stagnation detection | ✅ | ❌ |
| Graceful degradation | ✅ | ❌ |
| Dependencies | **4** | 13+ |

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

Advanced sequential thinking with 6-stage cognitive engine, semantic embeddings, anti-hallucination (assumption tracking, contradiction detection, confidence calibration), 6D quality metrics with trends, and bias detection.

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

## Mathematical Foundations

Every formula is **verified with Wolfram Alpha** against analytical solutions.

### Cosine Similarity — Salton (1975)

```
cos(A, B) = (A · B) / (‖A‖ × ‖B‖)
```

Used for semantic similarity between thought embeddings (384-dimensional BGE vectors) and keyword-based fallback (TF frequency vectors).

**Verification:** `cos([1,2,3], [4,5,6]) = 32/√1078 ≈ 0.9746` ✅ Wolfram Alpha confirmed.

### OLS Linear Regression — Gauss (1795)

```
slope = (n·Σxy − Σx·Σy) / (n·Σx² − (Σx)²)
```

Trend analysis over a sliding window of quality scores. `slope > 0.02` → improving, `< -0.02` → declining.

**Verification:** `y = 0.3 + 0.1x` → slope = `(5·6 − 10·2.5)/(5·30 − 100)` = `5/50` = `0.1` ✅ Wolfram Alpha confirmed.

### Weighted Mean — Quality Overall

```
overall = Σ(score_i × weight_i) / Σ(weight_i)
```

Stage-specific weights boost relevant dimensions. DEFINE: Clarity×3, ANALYZE: Depth×3, SYNTHESIZE: Actionability×3.

**Verification:** DEFINE scores `[0.8, 0.5, 0.4, 0.6, 0.7, 0.3]` with weights `[3,1,1,1,2,1]` → `28/45 ≈ 0.6222` ✅ Wolfram Alpha confirmed.

### Keyword Cosine (Fallback)

```
cos(TF_A, TF_B) where TF = term frequency vector
```

When neural embeddings are unavailable, token-based cosine similarity provides graceful degradation.

**Verification:** `"cat sat mat"` vs `"cat sat"` → `2/√6 ≈ 0.8165` ✅ Wolfram Alpha confirmed.

### Confidence Calibration

Stage-specific expected ranges based on Bloom's cognitive load:

```
DEFINE:      [0.30, 0.60]  — problem exploration, low certainty expected
SYNTHESIZE:  [0.70, 0.95]  — conclusion, high certainty expected
```

Monotonically increasing ranges — confidence expectations grow as reasoning matures.

---

## Anti-Hallucination

Three passive verification layers that require zero LLM calls:

### 1. Assumption Tracking

Accumulates and deduplicates assumptions across all thoughts. Semantic deduplication when embeddings are available (cosine > 0.85 → duplicate), exact match otherwise.

### 2. Contradiction Detection

Compares each new thought against all previous thoughts for semantic similarity + negation patterns:

```
contradiction = similarity(A, B) > 0.7 AND has_negation_difference(A, B)
```

Negation patterns: `not`, `don't`, `never`, `incorrect`, `wrong`, `instead`, `avoid`, `shouldn't`.

### 3. Confidence Calibration

Flags overconfidence (high confidence in early stages) and underconfidence (low confidence in late stages):

```
DEFINE + confidence 0.9  →  ⚠️ overconfident
SYNTHESIZE + confidence 0.3  →  ⚠️ underconfident
```

---

## Bias Detection

Identifies 5 cognitive biases with actionable English suggestions:

| Bias | Detection Method | Trigger |
|------|:---------------:|---------|
| **Confirmation** | History similarity > 0.7 for recent thoughts | Pattern indicates repeatedly reinforcing same conclusion |
| **Anchoring** | First quantitative reference dominates | Over-reliance on initial data point |
| **Availability** | Recency weighting of examples | Using recent/memorable examples disproportionately |
| **Overconfidence** | High confidence early in reasoning | Confidence > 0.8 before 50% progress |
| **Sunk Cost** | Late-stage reluctance to change | Keywords like "already invested" after 70% progress |

---

## Architecture

```
cuba-thinking/
├── package.json        # 4 dependencies
├── tsconfig.json       # TypeScript strict mode
└── src/
    ├── index.ts                     # MCP server entry point
    ├── types.ts                     # Zod schemas + TypeScript interfaces
    ├── formatter.ts                 # Structured response rendering
    └── services/
        ├── cognitive-processor.ts   # Central orchestrator
        ├── embedding.service.ts     # BGE-384d + keyword fallback
        ├── stage-engine.service.ts  # 6-stage FSM
        ├── quality-metrics.service.ts # 6D + OLS trends
        ├── anti-hallucination.service.ts # 3-layer verification
        └── bias-detector.service.ts # 5-bias detection (English)
```

### Dependencies (4 total)

| Package | Purpose |
|---------|---------|
| `@modelcontextprotocol/sdk` | MCP protocol server |
| `@huggingface/transformers` | Local BGE embeddings (lazy init, ~80MB one-time download) |
| `zod` | Input validation |
| `chalk` | Terminal formatting |

### Graceful Degradation

If the embedding model fails to load (no internet, ONNX error, etc.), Cuba-Thinking automatically falls back to keyword-based cosine similarity. All features continue working — embedding-dependent features degrade to heuristic equivalents.

---

## Verification

### NEMESIS Protocol (3-level test suite)

```
Test Suites: 6 passed, 6 total
Tests:       108 passed, 108 total
Time:        3.589s
```

| Suite | Tests | Levels |
|-------|:-----:|--------|
| StageEngine | 15 | 🟢6 🟡6 🔴3 |
| QualityMetrics | 16 | 🟢5 🟡6 🔴5 |
| AntiHallucination | 25 | 🟢7 🟡7 🔴5 + 6 general |
| BiasDetector | 14 | 🟢4 🟡5 🔴5 |
| EmbeddingService | 19 | 🟢9 🟡8 🔴6 |
| CognitiveProcessor | 10 | 🟢3 🟡3 🔴3 |

### Mathematical Verification (Wolfram Alpha)

| Formula | Expected | Computed | Source |
|---------|:--------:|:--------:|:------:|
| `cos([1,2,3],[4,5,6])` | `32/√1078` | `0.97463184619...` | Wolfram Alpha ✅ |
| OLS slope `y=0.3+0.1x` | `1/10` | `0.1` | Wolfram Alpha ✅ |
| Keyword `"cat sat mat"/"cat sat"` | `2/√6` | `0.81649658092...` | Wolfram Alpha ✅ |
| Weighted mean (DEFINE) | `28/45` | `0.62222222222...` | Wolfram Alpha ✅ |

---

## How It Works in Practice

### 1. Structured reasoning with cognitive stages

```
Agent: I need to analyze the caching options...
→ cuba_thinking(thought: "...", stage: "ANALYZE", confidence: 0.6)
← Stage: ANALYZE (50% progress)
  Quality: Clarity=0.72, Depth=0.85 (boosted 3×), Breadth=0.64
  Trend: improving 📈
  Calibration: ✅ calibrated (0.6 is within ANALYZE range [0.4, 0.8])
```

### 2. Anti-hallucination catches contradictions

```
Thought 1: "Use React for the frontend"
Thought 5: "Don't use React, use Vue instead"
→ ⚠️ Potential contradiction between Thought #1 and #5
  Similarity: 0.82, Negation detected: "Don't use"
```

### 3. Bias detection prevents cognitive traps

```
Agent: "I am absolutely certain this is correct" (confidence: 0.95, thought 2/10)
→ ⚠️ Overconfidence bias detected at 20% progress
  Suggestion: "Verify with evidence before proceeding with high confidence"
```

### 4. Quality trends guide improvement

```
Thoughts 1-3: Quality declining 📉
→ suggestedAction: "Quality is declining. Consider revisiting your approach."

Thoughts 4-7: Quality improving 📈
→ On track, continue current approach
```

---

## Part of the Cuba Ecosystem

| Project | Purpose |
|---------|---------|
| [Cuba-Memorys](https://github.com/lENADRO1910/cuba-memorys) | Persistent memory — knowledge graph, Hebbian learning, anti-hallucination grounding |
| **Cuba-Thinking** | Sequential reasoning — cognitive engine, quality metrics, bias detection |

Together, they give AI agents **memory + reasoning** — the two fundamental capabilities for reliable AI assistance.

---

## License

[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) — Free to use and modify, **not for commercial use**.

---

## Author

**Leandro Pérez G.**

- GitHub: [@lENADRO1910](https://github.com/lENADRO1910)
- Email: [leandropatodo@gmail.com](mailto:leandropatodo@gmail.com)

## Credits

Mathematical foundations: Bloom (1956, Taxonomy), Anderson & Krathwohl (2001, Revised Taxonomy), Salton (1975, Cosine Similarity), Gauss (1795, OLS Regression), Xiao et al. (2023, BGE Embeddings), Kahneman & Tversky (1974, Cognitive Biases).
