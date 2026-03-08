import type {
  CubaThinkingInput,
  CubaThinkingOutput,
} from '../types.js';
import { EmbeddingService } from './embedding.service.js';
import { StageEngine } from './stage-engine.service.js';
import { QualityMetricsService } from './quality-metrics.service.js';
import { AntiHallucinationService } from './anti-hallucination.service.js';
import { BiasDetectorService } from './bias-detector.service.js';

export class CognitiveProcessor {
  private readonly embeddings: EmbeddingService;
  private readonly stage: StageEngine;
  private readonly quality: QualityMetricsService;
  private readonly antiHallucination: AntiHallucinationService;
  private readonly bias: BiasDetectorService;

  
  private thoughtHistory: string[] = [];

  constructor() {
    this.embeddings = new EmbeddingService();
    this.stage = new StageEngine();
    this.quality = new QualityMetricsService();
    this.antiHallucination = new AntiHallucinationService();
    this.bias = new BiasDetectorService();
  }

  
  async process(input: CubaThinkingInput): Promise<CubaThinkingOutput> {
    const { thought, thoughtNumber, totalThoughts, nextThoughtNeeded } = input;
    if (thoughtNumber === 1) {
      this.embeddings.ensureReady().catch(() => {
      });
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
    const assumptions = await this.antiHallucination.trackAssumptions(
      input.assumptions,
      thoughtNumber,
      this.embeddings,
    );

    const contradictions = await this.antiHallucination.detectContradictions(
      thought,
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
    return {
      thought,
      thoughtNumber,
      totalThoughts: input.needsMoreThoughts
        ? Math.max(totalThoughts, thoughtNumber + 2)
        : totalThoughts,
      nextThoughtNeeded,
      stage: stageInfo,
      quality: qualityScores,
      qualityTrend,
      assumptions,
      contradictions,
      confidenceCalibration,
      stagnationWarning,
      biasDetected: biasResult?.type,
      biasSuggestion: biasResult?.suggestion,
      relevanceScore: relevanceScore !== undefined
        ? Math.round(relevanceScore * 100) / 100
        : undefined,
    };
  }

  
  reset(): void {
    this.embeddings.clearCache();
    this.stage.reset();
    this.quality.reset();
    this.antiHallucination.reset();
    this.thoughtHistory = [];
  }
}
