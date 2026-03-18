import type { QualityScores, QualityTrend, ThinkingStage, BudgetMode } from '../types.js';
import { STAGE_QUALITY_WEIGHTS } from './stage-engine.service.js';

const MIN_TREND_POINTS = 3;
const TREND_WINDOW = 10;
const STAGNATION_EPSILON = 0.02;
const STAGNATION_CONSECUTIVE = 3;

// Word-boundary regex helper
const WB = (word: string): RegExp => new RegExp(`\\b${word}\\b`, 'i');

// Claim density patterns — ordered exclusive to avoid double-counting
// (B4 fix: \d+% matched both percentage AND 2-digit pattern)
const CLAIM_PATTERNS = [
  /\b\d+(\.\d+)?%/g,                           // Percentages (must be first)
  /\b\d{2,}(?!\s*%)\b/g,                        // Numbers ≥ 2 digits (exclude % suffix)
  /\b(always|never|all|none|every|must)\b/gi,   // Absolutes
  /\b(proves?|confirms?|demonstrates?|shows?)\b/gi, // Causal claims
];

// Metacognitive filler patterns — /i only, NO /g
// (B1 fix: global flag causes .test() to mutate lastIndex per ECMAScript §22.2.5.11)
const METACOG_PATTERNS = [
  /\b(let me think|let me consider|I need to think|I should think)\b/i,
  /\b(hmm|well|okay so|alright)\b/i,
  /\b(on second thought|wait|actually no)\b/i,
  /\b(I'm not sure|I think maybe|perhaps I should)\b/i,
];

// Fallacy patterns
const HASTY_ABSOLUTES = /\b(always|never|all|every|none|no one)\b/i;
const HASTY_SINGULAR = /\b(one|single|this example|this case|anecdot)\b/i;

// Logic connective types for diversity
const LOGIC_CONNECTIVES = [
  'because', 'therefore', 'thus', 'hence', 'since',
  'implies', 'if', 'then', 'consequently', 'so',
  'given that', 'it follows', 'as a result',
] as const;
const CONCLUSION_MARKERS = /\b(therefore|thus|in conclusion|finally|the answer is|hence|to summarize)\b/i;

// Actionability patterns
const IMPERATIVE_VERBS = /\b(use|implement|create|build|apply|test|verify|ensure|add|remove|check|configure|install|run|execute|avoid|consider)\b/gi;
const VAGUE_PHRASES = /\b(somehow|maybe|some kind of|in some way|things|stuff|might work|could be)\b/gi;
const SPECIFICITY_PATTERNS = [
  /\b\d+(\.\d+)?\s*(ms|mb|gb|kb|s|%|px|rem|em)\b/gi,  // Units
  /[A-Za-z]+\.[a-z]{2,4}\b/g,                           // File extensions
  /\/[a-z_/]+/gi,                                        // File paths
];

export class QualityMetricsService {
  private history: number[] = [];
  private ewma: number | null = null;
  private ewmaCount = 0;
  private consecutiveStagnant = 0;
  private confidenceHistory: number[] = [];
  private firstThoughtTokens: Set<string> | null = null;
  private cumulativeVocab = new Set<string>();
  private chainRefCount = 0; // V8

  
  calculate(
    thought: string,
    stage: ThinkingStage,
    manualMetrics?: Partial<Record<string, number>>,
    relevanceScore?: number,
  ): QualityScores {
    const scores: Record<string, number> = {
      clarity: this.evalClarity(thought),
      depth: this.evalDepth(thought),
      breadth: this.evalBreadth(thought),
      logic: this.evalLogic(thought),
      relevance: relevanceScore ?? this.evalRelevance(thought),
      actionability: this.evalActionability(thought),
    };
    if (manualMetrics) {
      for (const [key, value] of Object.entries(manualMetrics)) {
        if (value !== undefined && value >= 0 && value <= 5 && key in scores) {
          scores[key] = value / 5;
        }
      }
    }
    const weights = STAGE_QUALITY_WEIGHTS[stage];
    let weightedSum = 0;
    let totalWeight = 0;
    for (const [dim, score] of Object.entries(scores)) {
      const w = weights[dim] ?? 1;
      weightedSum += score * w;
      totalWeight += w;
    }
    const overall = totalWeight > 0 ? weightedSum / totalWeight : 0;
    this.history.push(round(overall));

    return {
      clarity: round(scores.clarity),
      depth: round(scores.depth),
      breadth: round(scores.breadth),
      logic: round(scores.logic),
      relevance: round(scores.relevance),
      actionability: round(scores.actionability),
      overall: round(overall),
    };
  }

  getTrend(): QualityTrend {
    const data = this.history.slice(-TREND_WINDOW);
    if (data.length < MIN_TREND_POINTS) return 'stable';

    const slope = linearRegressionSlope(data);
    if (data.length >= 4) {
      let directionChanges = 0;
      for (let i = 2; i < data.length; i++) {
        const prev = data[i - 1] - data[i - 2];
        const curr = data[i] - data[i - 1];
        if ((prev > 0 && curr < 0) || (prev < 0 && curr > 0)) {
          directionChanges++;
        }
      }
      if (directionChanges >= data.length * 0.6) return 'unstable';
    }

    if (slope > 0.02) return 'improving';
    if (slope < -0.02) return 'declining';
    return 'stable';
  }

  stabilityScore(quality: QualityScores): number {
    const dims = [
      quality.clarity, quality.depth, quality.breadth,
      quality.logic, quality.relevance, quality.actionability,
    ];
    const H = shannonEntropy(dims);
    const Hmax = Math.log2(dims.length);
    return Hmax > 0 ? round(H / Hmax) : 0;
  }

  // V4: Adaptive EWMA decay (Roberts 1959, Zangari 1994)
  // Budget-aware α floor prevents EWMA becoming sluggish in long chains
  updateEwma(
    quality: number, coherence: number, contradictionRatio: number,
    faithfulness?: number, informationGain?: number, grounding?: number,
    budgetMode?: BudgetMode,
  ): number {
    this.ewmaCount++;
    const baseAlpha = 2 / (this.ewmaCount + 1);
    // V4: Alpha floor by budget mode
    const alphaFloor = budgetMode === 'fast' ? 0.3
      : budgetMode === 'exhaustive' ? 0.15
      : budgetMode === 'thorough' ? 0.20
      : 0; // balanced: no floor
    const alpha = Math.max(baseAlpha, alphaFloor);
    // V10: Include E1/E4/E6 in composite reward (Shannon DPI 1948)
    const f = faithfulness ?? 1;
    const ig = informationGain ?? 0.5;
    const g = grounding ?? 1;
    const reward = 0.40 * quality + 0.20 * coherence + 0.10 * (1 - contradictionRatio)
      + 0.10 * f + 0.10 * ig + 0.10 * g;
    this.ewma = this.ewma === null
      ? reward
      : alpha * reward + (1 - alpha) * this.ewma;
    return round(this.ewma);
  }

  // Confidence variance tracking (Shewhart)
  trackConfidence(confidence: number | undefined): number | undefined {
    if (confidence === undefined) return undefined;
    this.confidenceHistory.push(confidence);
    if (this.confidenceHistory.length < 3) return undefined;
    const mean = this.confidenceHistory.reduce((a, b) => a + b, 0) / this.confidenceHistory.length;
    const variance = this.confidenceHistory.reduce((s, v) => s + (v - mean) ** 2, 0) / this.confidenceHistory.length;
    const stdDev = Math.sqrt(variance);
    return stdDev > 0.25 ? round(stdDev) : undefined;
  }

  // Early stopping signal
  checkEarlyStopping(
    thoughtNumber: number,
    totalThoughts: number,
    stageProgress: number,
    qualityOverall: number,
  ): string | undefined {
    if (thoughtNumber < 3) return undefined;
    const ewmaConverged = this.ewma !== null && this.history.length >= 3 &&
      Math.abs(this.history[this.history.length - 1] - this.history[this.history.length - 2]) < 0.01;
    const stageNearEnd = stageProgress >= 0.83;
    const qualityHigh = qualityOverall >= 0.75;
    if (ewmaConverged && stageNearEnd && qualityHigh && thoughtNumber >= totalThoughts - 1) {
      return `Early stopping recommended: quality converged at ${(qualityOverall * 100).toFixed(0)}%, stage ${(stageProgress * 100).toFixed(0)}% complete. Consider concluding.`;
    }
    return undefined;
  }

  checkOverthinking(_thoughtNumber: number): string | undefined {
    if (this.history.length < STAGNATION_CONSECUTIVE + 1) return undefined;
    const recent = this.history.slice(-2);
    const diff = Math.abs(recent[1] - recent[0]);
    if (diff < STAGNATION_EPSILON) {
      this.consecutiveStagnant++;
    } else {
      this.consecutiveStagnant = 0;
    }
    if (this.consecutiveStagnant >= STAGNATION_CONSECUTIVE) {
      return `Stagnation detected: ${this.consecutiveStagnant} consecutive thoughts with <${(STAGNATION_EPSILON * 100).toFixed(0)}% quality improvement. Consider concluding.`;
    }
    return undefined;
  }

  // Claim density scoring
  measureClaimDensity(thought: string): { density: number; claimCount: number } {
    const sentences = thought.split(/[.!?]+/).filter(s => s.trim().length > 0);
    if (sentences.length === 0) return { density: 0, claimCount: 0 };
    let claimCount = 0;
    for (const pattern of CLAIM_PATTERNS) {
      const matches = thought.match(pattern);
      if (matches) claimCount += matches.length;
    }
    const density = round(claimCount / sentences.length);
    return { density, claimCount };
  }

  // Metacognitive signal detection
  measureMetacognition(thought: string): { ratio: number; warning?: string } {
    const sentences = thought.split(/[.!?]+/).filter(s => s.trim().length > 0);
    if (sentences.length === 0) return { ratio: 0 };
    let metacogSentences = 0;
    for (const s of sentences) {
      if (METACOG_PATTERNS.some(p => p.test(s))) {
        metacogSentences++;
      }
      // B1 fix: lastIndex reset no longer needed — /g flag removed from METACOG_PATTERNS
    }
    const ratio = round(metacogSentences / sentences.length);
    if (ratio > 0.3) {
      return {
        ratio,
        warning: `High metacognition (${(ratio * 100).toFixed(0)}% filler). Focus on substance over "thinking about thinking".`,
      };
    }
    return { ratio };
  }

  // Fallacy detection
  detectFallacies(thought: string): string | undefined {
    // Hasty generalization detection
    if (HASTY_ABSOLUTES.test(thought) && HASTY_SINGULAR.test(thought)) {
      return 'Possible hasty generalization: absolute claim near singular evidence. Consider qualifying the scope.';
    }
    return undefined;
  }

  // Dialectical score for VERIFY/SYNTHESIZE
  measureDialectical(thought: string, stage: ThinkingStage): { score: number; warning?: string } {
    if (stage !== 'VERIFY' && stage !== 'SYNTHESIZE') return { score: 1 };
    const hasCounter = /\b(however|but|on the other hand|alternatively|counterpoint|downside|drawback)\b/i.test(thought);
    const hasConcession = /\b(admittedly|granted|while|although|despite|nevertheless)\b/i.test(thought);
    const hasSynthesis = /\b(therefore|overall|balancing|considering both|net effect|in summary)\b/i.test(thought);
    const score = round(
      0.33 * (hasCounter ? 1 : 0) +
      0.33 * (hasConcession ? 1 : 0) +
      0.34 * (hasSynthesis ? 1 : 0),
    );
    if (score < 0.33) {
      return {
        score,
        warning: `Low dialectical reasoning in ${stage} — consider arguing against your conclusion before finalizing.`,
      };
    }
    return { score };
  }

  // Reasoning type detection
  detectReasoningType(thought: string): { dominant: string; feedback?: string } {
    const lower = thought.toLowerCase();
    const counts = {
      deductive: 0,
      inductive: 0,
      abductive: 0,
      analogical: 0,
    };
    if (WB('therefore').test(lower) || WB('thus').test(lower) || WB('it follows').test(lower) || WB('must be').test(lower)) counts.deductive++;
    if (WB('pattern').test(lower) || WB('data shows').test(lower) || WB('evidence indicates').test(lower) || WB('trend').test(lower)) counts.inductive++;
    if (/\b(best explanation|most likely|hypothesis|plausible)\b/i.test(lower)) counts.abductive++;
    if (/\b(similar to|like|analogous|reminds of|compared to)\b/i.test(lower)) counts.analogical++;
    const entries = Object.entries(counts) as [string, number][];
    const total = entries.reduce((s, [, v]) => s + v, 0);
    if (total === 0) return { dominant: 'mixed' };
    const [dominant, dominantCount] = entries.reduce((max, e) => e[1] > max[1] ? e : max, entries[0]);
    const ratio = dominantCount / total;
    if (ratio > 0.8 && total >= 2) {
      return {
        dominant,
        feedback: `Reasoning is ${(ratio * 100).toFixed(0)}% ${dominant} — consider incorporating ${dominant === 'deductive' ? 'inductive' : 'deductive'} reasoning for balance.`,
      };
    }
    return { dominant };
  }

  // GoT topology analysis
  analyzeTopology(edges: Array<{ from: number; to: number }>, thoughtCount: number): { orphanCount: number; linearRatio: number } {
    if (thoughtCount <= 1 || edges.length === 0) return { orphanCount: 0, linearRatio: 1 };
    const connected = new Set<number>();
    for (const e of edges) { connected.add(e.from); connected.add(e.to); }
    const orphanCount = Math.max(0, thoughtCount - connected.size);
    const linearRatio = round(edges.length / Math.max(thoughtCount - 1, 1));
    return { orphanCount, linearRatio };
  }

  // E1: ROSCOE Faithfulness — how faithful is thought_n to prior thoughts (Golovneva et al., ICLR 2023)
  measureFaithfulness(embeddings: { isAvailable: boolean; similarity: (a: number, b: number) => number }, thoughtNumber: number): number | undefined {
    if (!embeddings.isAvailable || thoughtNumber <= 1) return undefined;
    let maxSimSum = 0;
    for (let i = 1; i < thoughtNumber; i++) {
      maxSimSum += embeddings.similarity(i, thoughtNumber);
    }
    return round(maxSimSum / (thoughtNumber - 1));
  }

  // E4: Information Gain (Shannon 1948) — unique new concepts introduced
  measureInformationGain(thought: string): number {
    const nouns = (thought.match(/\b[A-Z][a-z]{3,}\b/g) || []);
    const techTerms = (thought.match(/\b[a-z]{2,}(?:_[a-z]+)+\b/g) || []);
    const concepts = new Set([...nouns, ...techTerms]);
    let newConcepts = 0;
    for (const c of concepts) {
      if (!this.cumulativeVocab.has(c)) {
        newConcepts++;
        this.cumulativeVocab.add(c);
      }
    }
    return concepts.size > 0 ? round(newConcepts / concepts.size) : 0;
  }

  // E6: Source Grounding — fraction of claims backed by evidence vs ungrounded
  measureGrounding(thought: string): { score: number; warning?: string } {
    const sentences = thought.split(/[.!?]+/).filter(s => s.trim().length > 0);
    if (sentences.length === 0) return { score: 1 };
    const GROUNDED = /\b(according to|research shows|data from|per rfc|studies show|evidence suggests|source:|reference:|based on|documented in|as described in|paper by|measured at|benchmark shows)\b/i;
    const UNGROUNDED = /\b(always|never|obviously|clearly|everyone knows|it's well known|of course)\b/i;
    let grounded = 0;
    let ungrounded = 0;
    for (const s of sentences) {
      if (GROUNDED.test(s)) grounded++;
      if (UNGROUNDED.test(s) && !GROUNDED.test(s)) ungrounded++;
    }
    const total = grounded + ungrounded;
    if (total === 0) return { score: 1 };
    const score = round(grounded / total);
    if (score < 0.3 && total >= 2) {
      return { score, warning: `Low grounding (${(score * 100).toFixed(0)}%): ${ungrounded} ungrounded claims vs ${grounded} grounded. Add sources/references.` };
    }
    return { score };
  }

  // V1: Step Transition Coherence (Golovneva et al., ICLR 2023 — ROSCOE §3.2)
  measureStepCoherence(
    embeddings: { isAvailable: boolean; similarity: (a: number, b: number) => number },
    thoughtNumber: number,
  ): { score: number; warning?: string } | undefined {
    if (!embeddings.isAvailable || thoughtNumber <= 2) return undefined;
    const sim = embeddings.similarity(thoughtNumber - 1, thoughtNumber);
    const score = round(sim);
    if (score < 0.3) {
      return {
        score,
        warning: `Low step coherence (${(score * 100).toFixed(0)}%): thought #${thoughtNumber} may have jumped topics without explicit branching.`,
      };
    }
    return { score };
  }

  // V2: Evidence Accumulation Score (Wald 1945 — Sequential Analysis)
  measureEvidenceAccumulation(
    confidence: number | undefined,
    qualityOverall: number,
    groundingScore: number,
  ): { warning?: string } | undefined {
    if (confidence === undefined || this.confidenceHistory.length < 2) return undefined;
    const prevConfidence = this.confidenceHistory[this.confidenceHistory.length - 2];
    const actualDelta = confidence - prevConfidence;
    const expectedDelta = 0.1 * qualityOverall * groundingScore;
    if (actualDelta > 0.1 && expectedDelta < 0.03) {
      return {
        warning: `Unsupported confidence increase: +${(actualDelta * 100).toFixed(0)}% confidence but evidence strength only ${(expectedDelta * 100).toFixed(1)}%. Verify assumptions.`,
      };
    }
    if (actualDelta < 0.02 && expectedDelta > 0.05) {
      return {
        warning: `Confidence plateau: evidence supports +${(expectedDelta * 100).toFixed(0)}% but confidence only changed +${(actualDelta * 100).toFixed(0)}%. Consider updating your assessment.`,
      };
    }
    return undefined;
  }

  // V3: Verbosity Detection via Content-Word Ratio (Graesser 2004 — Coh-Metrix)
  measureVerbosity(thought: string): { ratio: number; warning?: string } {
    const words = thought.toLowerCase().split(/\s+/).filter(w => w.length > 1);
    if (words.length === 0) return { ratio: 0 };
    const CONTENT_STOPWORDS = new Set([
      'the', 'is', 'a', 'an', 'in', 'on', 'of', 'and', 'to', 'for',
      'it', 'that', 'this', 'with', 'as', 'be', 'was', 'are', 'or',
      'by', 'at', 'from', 'they', 'we', 'my', 'your', 'its', 'not',
      'but', 'if', 'so', 'do', 'has', 'had', 'have', 'been', 'would',
      'could', 'should', 'will', 'can', 'may', 'about', 'up', 'out',
      'let', 'me', 'think', 'well', 'okay', 'now', 'just', 'also',
    ]);
    const contentWords = words.filter(w => !CONTENT_STOPWORDS.has(w));
    const ratio = round(contentWords.length / words.length);
    if (ratio < 0.4) {
      return {
        ratio,
        warning: `High verbosity: only ${(ratio * 100).toFixed(0)}% content words. Be more concise.`,
      };
    }
    return { ratio };
  }

  // V7: Semantic Novelty (Guilford 1967 — Divergent Thinking Originality)
  measureSemanticNovelty(
    embeddings: { isAvailable: boolean; similarity: (a: number, b: number) => number },
    thoughtNumber: number,
  ): { score: number; warning?: string } | undefined {
    if (!embeddings.isAvailable || thoughtNumber <= 1) return undefined;
    let maxSim = 0;
    for (let i = 1; i < thoughtNumber; i++) {
      maxSim = Math.max(maxSim, embeddings.similarity(i, thoughtNumber));
    }
    const novelty = round(1 - maxSim);
    if (novelty < 0.15 && thoughtNumber > 2) {
      return {
        score: novelty,
        warning: `Semantic redundancy: thought #${thoughtNumber} is ${((1 - novelty) * 100).toFixed(0)}% similar to a prior thought. Explore a different angle.`,
      };
    }
    return { score: novelty };
  }

  // V8: Cross-Thought Reasoning Depth (Bloom's Taxonomy, Anderson & Krathwohl 2001)
  measureReasoningChain(thought: string, thoughtNumber: number): { score: number; warning?: string } {
    if (thoughtNumber <= 1) return { score: 1 };
    const BACKWARD_REFS = /\b(building on|given the above|from the previous|therefore since|as established|as shown earlier|this extends|following from|based on this|continuing from|as noted|per the earlier|given that we)\b/i;
    const hasRef = BACKWARD_REFS.test(thought) ? 1 : 0;
    this.chainRefCount = (this.chainRefCount ?? 0) + hasRef;
    const score = round(this.chainRefCount / (thoughtNumber - 1));
    if (score < 0.1 && thoughtNumber > 3) {
      return {
        score,
        warning: 'Low cross-thought depth: thoughts are not building on prior conclusions. Reference earlier findings.',
      };
    }
    return { score };
  }

  reset(): void {
    this.history = [];
    this.ewma = null;
    this.ewmaCount = 0;
    this.consecutiveStagnant = 0;
    this.confidenceHistory = [];
    this.firstThoughtTokens = null;
    this.cumulativeVocab.clear();
    this.chainRefCount = 0;
  }

  
  getHistory(): number[] {
    return [...this.history];
  }

  // TTR-based clarity (Templin 1957)
  private evalClarity(thought: string): number {
    if (!thought.trim()) return 0;
    const words = thought.toLowerCase().split(/\s+/).filter(w => w.length > 1);
    if (words.length === 0) return 0;
    const uniqueWords = new Set(words);
    // TTR = unique/total
    const ttr = uniqueWords.size / words.length;
    // Sentence diversity bonus
    const sentences = thought.split(/[.!?]+/).filter(s => s.trim().length > 0);
    const structureBonus = sentences.length >= 2 ? 0.1 : 0;
    const formatBonus = (thought.includes('```') || thought.includes('- ') || thought.includes('1.')) ? 0.1 : 0;
    return clamp(ttr + structureBonus + formatBonus);
  }

  // Clause-based depth (Hunt 1965)
  private evalDepth(thought: string): number {
    if (!thought.trim()) return 0;
    let score = 0.3;
    const lower = thought.toLowerCase();
    // Clause indicators: commas, semicolons, colons within sentences
    const clauseIndicators = (thought.match(/[,;:]/g) || []).length;
    const sentences = thought.split(/[.!?]+/).filter(s => s.trim().length > 0);
    const avgClauses = sentences.length > 0 ? clauseIndicators / sentences.length : 0;
    score += Math.min(0.3, avgClauses * 0.1);

    const depthWords = ['because', 'therefore', 'since', 'implies', 'consequence', 'specifically'];
    score += Math.min(0.3, depthWords.filter(w => WB(w).test(lower)).length * 0.1);
    // Length bonus
    if (thought.length > 200) score += 0.1;
    return clamp(score);
  }

  // Breadth with unique noun ratio
  private evalBreadth(thought: string): number {
    if (!thought.trim()) return 0;
    let score = 0.3;
    const lower = thought.toLowerCase();
    const breadthWords = ['also', 'another', 'additionally', 'furthermore', 'alternative', 'option', 'versus', 'compared'];
    score += Math.min(0.3, breadthWords.filter(w => WB(w).test(lower)).length * 0.1);
    const listItems = (thought.match(/^[-•*]\s/gm) || []).length;
    if (listItems >= 2) score += 0.15;
    if (listItems >= 4) score += 0.1;
    // Unique noun-like words (capitalized, >3 chars) as topic proxy
    const nouns = thought.match(/\b[A-Z][a-z]{3,}\b/g) || [];
    const uniqueNouns = new Set(nouns);
    if (uniqueNouns.size >= 3) score += 0.1;
    if (uniqueNouns.size >= 6) score += 0.05;
    return clamp(score);
  }

  // Structural logic scoring (ROSCOE-inspired)
  private evalLogic(thought: string): number {
    if (!thought.trim()) return 0;
    const lower = thought.toLowerCase();
    // Connective diversity: how many DIFFERENT connective types are used
    const usedConnectives = LOGIC_CONNECTIVES.filter(c => WB(c).test(lower));
    const connectiveDiversity = usedConnectives.length / LOGIC_CONNECTIVES.length;
    // Conditional chain depth: A because B, therefore C
    let chainDepth = 0;
    const causalPairs = [['because', 'therefore'], ['since', 'thus'], ['given that', 'hence'], ['if', 'then']];
    for (const [premise, conclusion] of causalPairs) {
      if (WB(premise).test(lower) && WB(conclusion).test(lower)) chainDepth++;
    }
    // Conclusion presence
    const hasConclusion = CONCLUSION_MARKERS.test(thought) ? 1 : 0;
    return clamp(
      0.4 * connectiveDiversity +
      0.3 * Math.min(chainDepth / 3, 1) +
      0.3 * hasConclusion,
    );
  }

  // B3 fix: compute real fallback relevance via keyword overlap with first thought
  private evalRelevance(thought: string): number {
    if (!thought.trim()) return 0;
    const tokens = thought.toLowerCase().split(/\s+/).filter((w: string) => w.length > 1);
    if (!this.firstThoughtTokens) {
      this.firstThoughtTokens = new Set(tokens);
      return 0.8; // First thought is highly relevant to itself
    }
    const overlap = tokens.filter((t: string) => this.firstThoughtTokens!.has(t)).length;
    return clamp(overlap / Math.max(tokens.length, 1));
  }

  // Actionability (GRACE-inspired)
  private evalActionability(thought: string): number {
    if (!thought.trim()) return 0;
    const sentences = thought.split(/[.!?]+/).filter(s => s.trim().length > 0);
    if (sentences.length === 0) return 0;
    // Imperative ratio — .match() with /g returns all matches, does NOT mutate lastIndex
    const imperatives = (thought.match(IMPERATIVE_VERBS) || []).length;
    const imperativeRatio = Math.min(imperatives / sentences.length, 1);
    // Specificity: numbers with units, file paths, extensions
    let specificityCount = 0;
    for (const pattern of SPECIFICITY_PATTERNS) {
      specificityCount += (thought.match(pattern) || []).length;
    }
    const specificity = Math.min(specificityCount / sentences.length, 1);
    // Concreteness: absence of vague phrases
    const vagueCount = (thought.match(VAGUE_PHRASES) || []).length;
    const concreteness = clamp(1 - vagueCount / Math.max(sentences.length, 1));
    return clamp(
      0.4 * imperativeRatio +
      0.3 * specificity +
      0.3 * concreteness,
    );
  }

}

function shannonEntropy(scores: number[]): number {
  const sum = scores.reduce((a, b) => a + b, 0);
  if (sum < 1e-10) return 0;
  const probs = scores.map(s => s / sum);
  return -probs.reduce((h, p) => p > 0 ? h + p * Math.log2(p) : h, 0);
}

function linearRegressionSlope(values: number[]): number {
  const n = values.length;
  if (n < 2) return 0;

  let sumX = 0;
  let sumY = 0;
  let sumXY = 0;
  let sumX2 = 0;

  for (let i = 0; i < n; i++) {
    sumX += i;
    sumY += values[i];
    sumXY += i * values[i];
    sumX2 += i * i;
  }

  const denominator = n * sumX2 - sumX * sumX;
  if (Math.abs(denominator) < 1e-10) return 0;

  return (n * sumXY - sumX * sumY) / denominator;
}

function clamp(value: number, min = 0, max = 1): number {
  return Math.max(min, Math.min(max, value));
}

function round(value: number, decimals = 2): number {
  const factor = 10 ** decimals;
  return Math.round(value * factor) / factor;
}
