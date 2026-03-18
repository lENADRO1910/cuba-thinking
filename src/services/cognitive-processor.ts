import type {
  CubaThinkingInput,
  CubaThinkingOutput,
  ThoughtEdge,
  EdgeType,
  VerificationCheckpoint,
  FatigueReport,
} from '../types.js';
import { EmbeddingService, keywordSimilarity } from './embedding.service.js';
import { NLIService } from './nli.service.js';
import { StageEngine } from './stage-engine.service.js';
import { QualityMetricsService } from './quality-metrics.service.js';
import { AntiHallucinationService } from './anti-hallucination.service.js';
import { BiasDetectorService } from './bias-detector.service.js';
import type { StageInfo } from '../types.js';

const FATIGUE_THRESHOLD = 3;
const FATIGUE_CRITICAL = 5;

// TD6: Memory bounds — prevent unbounded growth in long sessions
const MAX_HISTORY = 500;
const MAX_EDGES = 1000;

export class CognitiveProcessor {
  private readonly embeddings: EmbeddingService;
  private readonly nli: NLIService;
  private readonly stage: StageEngine;
  private readonly quality: QualityMetricsService;
  private readonly antiHallucination: AntiHallucinationService;
  private readonly bias: BiasDetectorService;

  private thoughtHistory: string[] = [];
  private edges: ThoughtEdge[] = [];
  private previousStage: string | null = null;
  private consecutiveQualityDrops = 0;
  private lastQuality = 0;
  private qualityByThought = new Map<number, number>();

  constructor() {
    this.embeddings = new EmbeddingService();
    this.nli = new NLIService();
    this.stage = new StageEngine();
    this.quality = new QualityMetricsService();
    this.antiHallucination = new AntiHallucinationService();
    this.bias = new BiasDetectorService();
  }

  /**
   * TD1: Main orchestrator decomposed into phases.
   * Original CC ~25 → now CC ~7 (delegates to sub-methods).
   */
  async process(input: CubaThinkingInput): Promise<CubaThinkingOutput> {
    const { thought, thoughtNumber, totalThoughts, nextThoughtNeeded } = input;

    // Phase 1: Initialize models on first thought
    if (thoughtNumber === 1) {
      this.embeddings.ensureReady().catch(() => {});
      this.nli.ensureReady().catch(() => {});
    }

    // Phase 2: Embed + stage + quality
    await this.embeddings.embed(thought, thoughtNumber);
    const stageInfo = this.stage.processStage(
      thought, input.thinkingStage, thoughtNumber, totalThoughts,
    );
    const relevanceScore = this.embeddings.isAvailable
      ? this.embeddings.relevance(thoughtNumber) : undefined;
    const qualityScores = this.quality.calculate(
      thought, stageInfo.current,
      input.qualityMetrics as Partial<Record<string, number>> | undefined,
      relevanceScore,
    );
    const qualityTrend = this.quality.getTrend();
    const stability = this.quality.stabilityScore(qualityScores);

    // Phase 3: Anti-hallucination
    const hallucinationResults = await this.processAntiHallucination(
      input, thought, thoughtNumber, stageInfo,
    );

    // Phase 4: EWMA + graph + fatigue + bias + E1/E4/E6
    const graphResults = this.processGraph(
      input, thought, thoughtNumber, totalThoughts, qualityScores.overall,
      hallucinationResults.contradictions.length,
    );

    // Phase 5: Advanced metrics
    const advancedMetrics = this.computeAdvancedMetrics(
      thought, thoughtNumber, totalThoughts, stageInfo, qualityScores.overall,
      input, graphResults.overthinkingWarning,
    );

    // Phase 6: Build output
    return this.buildOutput(
      input, thought, thoughtNumber, totalThoughts, nextThoughtNeeded,
      stageInfo, qualityScores, qualityTrend, stability, relevanceScore,
      hallucinationResults, graphResults, advancedMetrics,
    );
  }

  /**
   * Phase 3: Anti-hallucination checks.
   */
  private async processAntiHallucination(
    input: CubaThinkingInput,
    thought: string,
    thoughtNumber: number,
    stageInfo: StageInfo,
  ) {
    const contradictions = await this.antiHallucination.detectContradictions(
      thought, thoughtNumber, this.embeddings, this.nli, stageInfo.current, // V5: pass stage
    );
    const assumptions = await this.antiHallucination.trackAssumptions(
      input.assumptions, thoughtNumber,
    );
    const confidenceCalibration = this.antiHallucination.calibrateConfidence(
      input.confidence, stageInfo.current,
    );
    const stagnationWarning = this.antiHallucination.detectStagnation(
      thoughtNumber, this.embeddings,
    );
    let verificationCheckpoint: VerificationCheckpoint | undefined;
    if (this.previousStage && this.previousStage !== stageInfo.current) {
      verificationCheckpoint = this.antiHallucination.generateVerificationCheckpoint(
        thoughtNumber, this.previousStage, stageInfo.current,
      ) as VerificationCheckpoint | undefined;
    }
    this.previousStage = stageInfo.current;
    return {
      contradictions, assumptions, confidenceCalibration,
      stagnationWarning, verificationCheckpoint,
    };
  }

  /**
   * Phase 4: Graph coherence, EWMA, fatigue, bias.
   */
  private processGraph(
    input: CubaThinkingInput,
    thought: string,
    thoughtNumber: number,
    totalThoughts: number,
    qualityOverall: number,
    contradictionCount: number,
  ) {
    let coherence = 0.5;
    if (thoughtNumber > 1) {
      if (this.embeddings.isAvailable) {
        coherence = this.embeddings.similarity(thoughtNumber - 1, thoughtNumber);
      } else {
        const prevText = this.thoughtHistory[thoughtNumber - 2];
        if (prevText) {
          const freqPrev = this.embeddings.getFrequencyMap(thoughtNumber - 1, prevText);
          const freqCurr = this.embeddings.getFrequencyMap(thoughtNumber, thought);
          coherence = keywordSimilarity(prevText, thought, freqPrev, freqCurr);
        }
      }
    }
    // B2 fix: clamp to [0,1] — Roberts (1959)
    const contradictionRatio = thoughtNumber > 0
      ? Math.min(1, contradictionCount / thoughtNumber) : 0;
    const ewmaReward = this.quality.updateEwma(
      qualityOverall, coherence, contradictionRatio,
    ); // V4/V10: will update with E1/E4/E6 in Phase 5
    const overthinkingWarning = this.quality.checkOverthinking(thoughtNumber);
    this.registerEdges(input, thoughtNumber);
    const graphCoherence = this.computeGraphCoherence();
    const fatigue = this.computeFatigue(qualityOverall);
    const biasResult = this.bias.detect(
      thought, thoughtNumber, totalThoughts,
      input.confidence, this.thoughtHistory, this.embeddings, input.biasDetected,
    );
    this.updateHistory(thought, thoughtNumber, qualityOverall);
    // MCTS: historical quality for rollback
    let bestHistoricalQuality: { thoughtNumber: number; quality: number } | undefined;
    if (ewmaReward < 0.40 && thoughtNumber > 3) {
      bestHistoricalQuality = this.findBestHistoricalThought(thoughtNumber);
    }
    return {
      ewmaReward, overthinkingWarning, graphCoherence, fatigue,
      biasResult, bestHistoricalQuality,
    };
  }

  /**
   * Phase 5: Advanced metrics (claims, metacog, fallacies, etc.)
   */
  private computeAdvancedMetrics(
    thought: string,
    thoughtNumber: number,
    totalThoughts: number,
    stageInfo: StageInfo,
    qualityOverall: number,
    input: CubaThinkingInput,
    overthinkingWarning: string | undefined,
  ) {
    const claimDensity = this.quality.measureClaimDensity(thought);
    const metacog = this.quality.measureMetacognition(thought);
    const fallacyWarning = this.quality.detectFallacies(thought);
    const dialectical = this.quality.measureDialectical(thought, stageInfo.current);
    const reasoning = this.quality.detectReasoningType(thought);
    const earlyStop = this.quality.checkEarlyStopping(
      thoughtNumber, totalThoughts, stageInfo.progress, qualityOverall,
    );
    const confidenceVariance = this.quality.trackConfidence(input.confidence);
    const topology = this.quality.analyzeTopology(this.edges, thoughtNumber);
    // E1: ROSCOE Faithfulness
    const faithfulness = this.quality.measureFaithfulness(this.embeddings, thoughtNumber);
    // E4: Information Gain (Shannon)
    const informationGain = this.quality.measureInformationGain(thought);
    // E6: Source Grounding
    const grounding = this.quality.measureGrounding(thought);

    // V1: Step Transition Coherence (Golovneva ICLR 2023)
    const stepCoherence = this.quality.measureStepCoherence(this.embeddings, thoughtNumber);
    // V2: Evidence Accumulation (Wald 1945)
    const evidenceAccum = this.quality.measureEvidenceAccumulation(
      input.confidence, qualityOverall, grounding.score,
    );
    // V3: Verbosity (Graesser 2004)
    const verbosity = this.quality.measureVerbosity(thought);
    // V7: Semantic Novelty (Guilford 1967)
    const semanticNovelty = this.quality.measureSemanticNovelty(this.embeddings, thoughtNumber);
    // V8: Reasoning Chain Depth (Bloom 2001)
    const reasoningChain = this.quality.measureReasoningChain(thought, thoughtNumber);

    const adjustedTotal = this.adjustBudget(
      totalThoughts, thoughtNumber, input, overthinkingWarning, earlyStop, qualityOverall,
    );
    return {
      claimDensity, metacog, fallacyWarning, dialectical,
      reasoning, earlyStop, confidenceVariance, topology, adjustedTotal,
      faithfulness, informationGain, grounding,
      stepCoherence, evidenceAccum, verbosity, semanticNovelty, reasoningChain,
    };
  }

  /**
   * Adjusts totalThoughts based on budget mode signals.
   */
  private adjustBudget(
    totalThoughts: number,
    thoughtNumber: number,
    input: CubaThinkingInput,
    overthinkingWarning: string | undefined,
    earlyStop: string | undefined,
    qualityOverall?: number,
  ): number {
    let adjusted = totalThoughts;
    if (input.needsMoreThoughts) {
      adjusted = Math.max(totalThoughts, thoughtNumber + 2);
    }
    if (overthinkingWarning && input.budgetMode === 'fast') {
      adjusted = Math.min(adjusted, thoughtNumber + 1);
    }
    if (earlyStop && input.budgetMode === 'fast') {
      adjusted = Math.min(adjusted, thoughtNumber + 1);
    }
    // V6: Budget-aware quality gate (Wald 1947 — Optimal Stopping)
    if (qualityOverall !== undefined && thoughtNumber > 2) {
      const qualityFloor = input.budgetMode === 'fast' ? 0.30
        : input.budgetMode === 'balanced' ? 0.25
        : input.budgetMode === 'thorough' ? 0.20
        : 0.15; // exhaustive
      if (qualityOverall < qualityFloor) {
        adjusted = Math.min(adjusted, thoughtNumber + 1);
      }
    }
    return adjusted;
  }

  /**
   * Phase 6: Assemble output (data mapping, not logic — ternaries are acceptable).
   */
  private buildOutput(
    _input: CubaThinkingInput,
    thought: string,
    thoughtNumber: number,
    _totalThoughts: number,
    nextThoughtNeeded: boolean,
    stageInfo: StageInfo,
    qualityScores: import('../types.js').QualityScores,
    qualityTrend: import('../types.js').QualityTrend,
    stability: number,
    relevanceScore: number | undefined,
    hall: Awaited<ReturnType<typeof this.processAntiHallucination>>,
    graph: ReturnType<typeof this.processGraph>,
    adv: ReturnType<typeof this.computeAdvancedMetrics>,
  ): CubaThinkingOutput {
    return {
      thought,
      thoughtNumber,
      totalThoughts: adv.adjustedTotal,
      nextThoughtNeeded,
      stage: stageInfo,
      quality: qualityScores,
      qualityTrend,
      stability: stability < 0.6 ? stability : undefined,
      ewmaReward: graph.ewmaReward,
      assumptions: hall.assumptions,
      contradictions: hall.contradictions,
      confidenceCalibration: hall.confidenceCalibration,
      stagnationWarning: hall.stagnationWarning,
      overthinkingWarning: graph.overthinkingWarning,
      verificationCheckpoint: hall.verificationCheckpoint,
      fatigue: graph.fatigue.fatigueDetected ? graph.fatigue : undefined,
      edges: this.edges,
      graphCoherence: graph.graphCoherence,
      biasDetected: graph.biasResult?.type,
      biasSuggestion: graph.biasResult?.suggestion,
      relevanceScore: relevanceScore !== undefined
        ? Math.round(relevanceScore * 100) / 100 : undefined,
      claimDensity: adv.claimDensity.density > 0 ? adv.claimDensity.density : undefined,
      claimCount: adv.claimDensity.claimCount > 0 ? adv.claimDensity.claimCount : undefined,
      metacogRatio: adv.metacog.ratio > 0 ? adv.metacog.ratio : undefined,
      metacogWarning: adv.metacog.warning,
      fallacyWarning: adv.fallacyWarning,
      dialecticalScore: adv.dialectical.score < 1 ? adv.dialectical.score : undefined,
      dialecticalWarning: adv.dialectical.warning,
      reasoningType: adv.reasoning.dominant !== 'mixed' ? adv.reasoning.dominant : undefined,
      reasoningFeedback: adv.reasoning.feedback,
      earlyStopSuggestion: adv.earlyStop,
      confidenceVariance: adv.confidenceVariance,
      topologyOrphanCount: adv.topology.orphanCount > 0 ? adv.topology.orphanCount : undefined,
      topologyLinearRatio: adv.topology.linearRatio !== 1 ? adv.topology.linearRatio : undefined,
      bestHistoricalQuality: graph.bestHistoricalQuality,
      // E1: ROSCOE Faithfulness
      faithfulnessScore: adv.faithfulness !== undefined && adv.faithfulness < 0.95
        ? adv.faithfulness : undefined,
      // E4: Information Gain
      informationGain: adv.informationGain > 0 ? adv.informationGain : undefined,
      // E6: Source Grounding
      groundingScore: adv.grounding.score < 1 && adv.grounding.warning
        ? adv.grounding.score : undefined,
      groundingWarning: adv.grounding.warning,

      // V1: Step Transition Coherence
      stepCoherenceScore: adv.stepCoherence?.score,
      stepCoherenceWarning: adv.stepCoherence?.warning,
      // V2: Evidence Accumulation
      evidenceWarning: adv.evidenceAccum?.warning,
      // V3: Verbosity
      verbosityRatio: adv.verbosity.ratio < 0.4 ? adv.verbosity.ratio : undefined,
      verbosityWarning: adv.verbosity.warning,
      // V7: Semantic Novelty
      semanticNovelty: adv.semanticNovelty?.score,
      semanticNoveltyWarning: adv.semanticNovelty?.warning,
      // V8: Reasoning Chain Depth
      reasoningChainScore: adv.reasoningChain.score,
      reasoningChainWarning: adv.reasoningChain.warning,
      // V15: Warmup noise guard (Shewhart 1931 — min sample size)
      ...(thoughtNumber <= 2 ? {
        stagnationWarning: undefined,
        overthinkingWarning: undefined,
      } : {}),
    };
  }

  private registerEdges(input: CubaThinkingInput, thoughtNumber: number): void {
    if (input.branchFromThought) {
      this.edges.push({
        from: input.branchFromThought,
        to: thoughtNumber,
        type: 'extends' as EdgeType,
      });
    }
    if (input.revisesThought) {
      this.edges.push({
        from: input.revisesThought,
        to: thoughtNumber,
        type: 'revises' as EdgeType,
      });
    }
    if (input.parentThoughts) {
      for (const parent of input.parentThoughts) {
        this.edges.push({
          from: parent,
          to: thoughtNumber,
          type: 'merges' as EdgeType,
        });
      }
    }
    // TD6: trim edges if exceeding MAX_EDGES
    if (this.edges.length > MAX_EDGES) {
      this.edges = this.edges.slice(-MAX_EDGES);
    }
  }

  /**
   * Compute average graph coherence from edge embeddings.
   */
  private computeGraphCoherence(): number | undefined {
    if (this.edges.length === 0 || !this.embeddings.isAvailable) return undefined;
    let totalSim = 0;
    for (const edge of this.edges) {
      totalSim += this.embeddings.similarity(edge.from, edge.to);
    }
    return Math.round((totalSim / this.edges.length) * 100) / 100;
  }

  /**
   * Update thought history with TD6 memory bounds.
   */
  private updateHistory(thought: string, thoughtNumber: number, qualityOverall: number): void {
    while (this.thoughtHistory.length < thoughtNumber) {
      this.thoughtHistory.push('');
    }
    this.thoughtHistory[thoughtNumber - 1] = thought;
    this.qualityByThought.set(thoughtNumber, qualityOverall);
    // TD6: trim history if exceeding MAX_HISTORY
    if (this.thoughtHistory.length > MAX_HISTORY) {
      this.thoughtHistory = this.thoughtHistory.slice(-MAX_HISTORY);
    }
  }

  private computeFatigue(currentQuality: number): FatigueReport {
    if (currentQuality < this.lastQuality) {
      this.consecutiveQualityDrops++;
    } else if (currentQuality > this.lastQuality) {
      this.consecutiveQualityDrops = 0;
    }
    this.lastQuality = currentQuality;

    let suggestedAction: 'continue' | 'conclude' | 'step_back' = 'continue';
    if (this.consecutiveQualityDrops >= FATIGUE_CRITICAL) {
      suggestedAction = 'conclude';
    } else if (this.consecutiveQualityDrops >= FATIGUE_THRESHOLD) {
      suggestedAction = 'step_back';
    }

    return {
      fatigueDetected: this.consecutiveQualityDrops >= FATIGUE_THRESHOLD,
      consecutiveDrops: this.consecutiveQualityDrops,
      suggestedAction,
    };
  }

  reset(): void {
    this.embeddings.clearCache();
    this.nli.reset();
    this.stage.reset();
    this.quality.reset();
    this.antiHallucination.reset();
    this.thoughtHistory = [];
    this.edges = [];
    this.previousStage = null;
    this.consecutiveQualityDrops = 0;
    this.lastQuality = 0;
    this.qualityByThought.clear();
  }

  private findBestHistoricalThought(
    currentThought: number,
  ): { thoughtNumber: number; quality: number } {
    let bestNum = 1;
    let bestQuality = 0;
    for (const [num, quality] of this.qualityByThought.entries()) {
      if (num < currentThought && quality > bestQuality) {
        bestNum = num;
        bestQuality = quality;
      }
    }
    return { thoughtNumber: bestNum, quality: Math.round(bestQuality * 100) / 100 };
  }
}
