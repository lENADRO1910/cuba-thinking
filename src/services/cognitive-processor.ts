import type {
  CubaThinkingInput,
  CubaThinkingOutput,
  ThoughtEdge,
  EdgeType,
  VerificationCheckpoint,
  FatigueReport,
} from '../types.js';
import { EmbeddingService } from './embedding.service.js';
import { NLIService } from './nli.service.js';
import { StageEngine } from './stage-engine.service.js';
import { QualityMetricsService } from './quality-metrics.service.js';
import { AntiHallucinationService } from './anti-hallucination.service.js';
import { BiasDetectorService } from './bias-detector.service.js';

const FATIGUE_THRESHOLD = 3;
const FATIGUE_CRITICAL = 5;

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

  async process(input: CubaThinkingInput): Promise<CubaThinkingOutput> {
    const { thought, thoughtNumber, totalThoughts, nextThoughtNeeded } = input;
    if (thoughtNumber === 1) {
      this.embeddings.ensureReady().catch(() => {});
      this.nli.ensureReady().catch(() => {});
    }
    await this.embeddings.embed(thought, thoughtNumber);
    const stageInfo = this.stage.processStage(
      thought,
      input.thinkingStage,
      thoughtNumber,
      totalThoughts,
    );
    const relevanceScore = this.embeddings.isAvailable
      ? this.embeddings.relevance(thoughtNumber)
      : undefined;
    const qualityScores = this.quality.calculate(
      thought,
      stageInfo.current,
      input.qualityMetrics as Partial<Record<string, number>> | undefined,
      relevanceScore,
    );
    const qualityTrend = this.quality.getTrend();
    const stability = this.quality.stabilityScore(qualityScores);
    const contradictions = await this.antiHallucination.detectContradictions(
      thought, thoughtNumber, this.embeddings, this.nli,
    );
    const coherence = this.embeddings.isAvailable && thoughtNumber > 1
      ? this.embeddings.similarity(thoughtNumber - 1, thoughtNumber)
      : 0.5;
    const contradictionRatio = thoughtNumber > 0 ? contradictions.length / thoughtNumber : 0;
    const ewmaReward = this.quality.updateEwma(
      qualityScores.overall, coherence, contradictionRatio,
    );
    const overthinkingWarning = this.quality.checkOverthinking(thoughtNumber);
    const assumptions = await this.antiHallucination.trackAssumptions(
      input.assumptions,
      thoughtNumber,
      this.embeddings,
    );
    const confidenceCalibration = this.antiHallucination.calibrateConfidence(
      input.confidence,
      stageInfo.current,
    );
    const stagnationWarning = this.antiHallucination.detectStagnation(
      thoughtNumber,
      this.embeddings,
    );
    let verificationCheckpoint: VerificationCheckpoint | undefined;
    if (this.previousStage && this.previousStage !== stageInfo.current) {
      verificationCheckpoint = this.antiHallucination.generateVerificationCheckpoint(
        thoughtNumber,
        this.previousStage,
        stageInfo.current,
      ) as VerificationCheckpoint | undefined;
    }
    this.previousStage = stageInfo.current;
    this.registerEdges(input, thoughtNumber);
    let graphCoherence: number | undefined;
    if (this.edges.length > 0 && this.embeddings.isAvailable) {
      let totalSim = 0;
      for (const edge of this.edges) {
        totalSim += this.embeddings.similarity(edge.from, edge.to);
      }
      graphCoherence = Math.round((totalSim / this.edges.length) * 100) / 100;
    }
    const fatigue = this.computeFatigue(qualityScores.overall);
    const biasResult = this.bias.detect(
      thought,
      thoughtNumber,
      totalThoughts,
      input.confidence,
      this.thoughtHistory,
      input.biasDetected,
    );
    while (this.thoughtHistory.length < thoughtNumber) {
      this.thoughtHistory.push('');
    }
    this.thoughtHistory[thoughtNumber - 1] = thought;
    this.qualityByThought.set(thoughtNumber, qualityScores.overall);

    // MCTS: Find best historical thought for potential rollback
    let bestHistoricalQuality: { thoughtNumber: number; quality: number } | undefined;
    if (ewmaReward < 0.40 && thoughtNumber > 3) {
      bestHistoricalQuality = this.findBestHistoricalThought(thoughtNumber);
    }


    const claimDensityResult = this.quality.measureClaimDensity(thought);
    const metacogResult = this.quality.measureMetacognition(thought);
    const fallacyWarning = this.quality.detectFallacies(thought);
    const dialecticalResult = this.quality.measureDialectical(thought, stageInfo.current);
    const reasoningResult = this.quality.detectReasoningType(thought);
    const earlyStopSuggestion = this.quality.checkEarlyStopping(
      thoughtNumber, totalThoughts, stageInfo.progress, qualityScores.overall,
    );
    const confidenceVariance = this.quality.trackConfidence(input.confidence);
    const topologyResult = this.quality.analyzeTopology(this.edges, thoughtNumber);

    let adjustedTotal = totalThoughts;
    if (input.needsMoreThoughts) {
      adjustedTotal = Math.max(totalThoughts, thoughtNumber + 2);
    }
    if (overthinkingWarning && input.budgetMode === 'fast') {
      adjustedTotal = Math.min(adjustedTotal, thoughtNumber + 1);
    }
    if (earlyStopSuggestion && input.budgetMode === 'fast') {
      adjustedTotal = Math.min(adjustedTotal, thoughtNumber + 1);
    }

    return {
      thought,
      thoughtNumber,
      totalThoughts: adjustedTotal,
      nextThoughtNeeded,
      stage: stageInfo,
      quality: qualityScores,
      qualityTrend,
      stability: stability < 0.6 ? stability : undefined,
      ewmaReward,
      assumptions,
      contradictions,
      confidenceCalibration,
      stagnationWarning,
      overthinkingWarning,
      verificationCheckpoint,
      fatigue: fatigue.fatigueDetected ? fatigue : undefined,
      edges: this.edges,
      graphCoherence,
      biasDetected: biasResult?.type,
      biasSuggestion: biasResult?.suggestion,
      relevanceScore: relevanceScore !== undefined
        ? Math.round(relevanceScore * 100) / 100
        : undefined,
  
      claimDensity: claimDensityResult.density > 0 ? claimDensityResult.density : undefined,
      claimCount: claimDensityResult.claimCount > 0 ? claimDensityResult.claimCount : undefined,
      metacogRatio: metacogResult.ratio > 0 ? metacogResult.ratio : undefined,
      metacogWarning: metacogResult.warning,
      fallacyWarning,
      dialecticalScore: dialecticalResult.score < 1 ? dialecticalResult.score : undefined,
      dialecticalWarning: dialecticalResult.warning,
      reasoningType: reasoningResult.dominant !== 'mixed' ? reasoningResult.dominant : undefined,
      reasoningFeedback: reasoningResult.feedback,
      earlyStopSuggestion,
      confidenceVariance,
      topologyOrphanCount: topologyResult.orphanCount > 0 ? topologyResult.orphanCount : undefined,
      topologyLinearRatio: topologyResult.linearRatio !== 1 ? topologyResult.linearRatio : undefined,
      bestHistoricalQuality,
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
