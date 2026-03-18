import type { ThinkingStage, StageInfo } from '../types.js';
import { THINKING_STAGES } from '../types.js';

export const CONFIDENCE_RANGES: Record<ThinkingStage, { min: number; max: number }> = {
  DEFINE:      { min: 0.3, max: 0.6 },
  RESEARCH:    { min: 0.3, max: 0.7 },
  ANALYZE:     { min: 0.4, max: 0.8 },
  HYPOTHESIZE: { min: 0.5, max: 0.85 },
  VERIFY:      { min: 0.6, max: 0.95 },
  SYNTHESIZE:  { min: 0.7, max: 1.0 },
};

export const STAGE_QUALITY_WEIGHTS: Record<ThinkingStage, Record<string, number>> = {
  DEFINE:      { clarity: 3, depth: 1, breadth: 1, logic: 1, relevance: 1, actionability: 1 },
  RESEARCH:    { clarity: 1, depth: 1, breadth: 3, logic: 1, relevance: 1, actionability: 1 },
  ANALYZE:     { clarity: 1, depth: 3, breadth: 1, logic: 1, relevance: 1, actionability: 1 },
  HYPOTHESIZE: { clarity: 1, depth: 1, breadth: 1, logic: 3, relevance: 1, actionability: 1 },
  VERIFY:      { clarity: 1, depth: 1, breadth: 1, logic: 1, relevance: 3, actionability: 1 },
  SYNTHESIZE:  { clarity: 1, depth: 1, breadth: 1, logic: 1, relevance: 1, actionability: 3 },
};

const STAGE_KEYWORDS: Record<ThinkingStage, string[]> = {
  DEFINE: [
    'problem', 'define', 'scope', 'requirement', 'goal', 'objective',
    'constraint', 'boundary', 'what', 'clarify', 'understand',
  ],
  RESEARCH: [
    'research', 'explore', 'option', 'alternative', 'approach',
    'investigate', 'compare', 'survey', 'find', 'search', 'look into',
  ],
  ANALYZE: [
    'analyze', 'trade-off', 'tradeoff', 'pro', 'con', 'advantage',
    'disadvantage', 'decompose', 'break down', 'evaluate', 'assess',
  ],
  HYPOTHESIZE: [
    'hypothesi', 'propose', 'suggest', 'recommend', 'solution',
    'approach would be', 'plan', 'design', 'architect', 'implement',
  ],
  VERIFY: [
    'verify', 'validate', 'test', 'check', 'confirm', 'ensure',
    'prove', 'evidence', 'assumption', 'correct', 'actually',
  ],
  SYNTHESIZE: [
    'synthesize', 'conclude', 'summary', 'final', 'overall',
    'recommendation', 'decision', 'therefore', 'in conclusion',
  ],
};

const MAX_SAME_STAGE = 8;

export class StageEngine {
  private stageHistory: ThinkingStage[] = [];
  private consecutiveSameStage = 0;
  private lastStage: ThinkingStage | null = null;


  processStage(
    thought: string,
    requestedStage: ThinkingStage | undefined,
    thoughtNumber: number,
    totalThoughts: number,
  ): StageInfo {
    const stage = requestedStage
      ?? this.autoDetectStage(thought)
      ?? this.inferFromProgress(thoughtNumber, totalThoughts);
    if (stage === this.lastStage) {
      this.consecutiveSameStage++;
    } else {
      this.consecutiveSameStage = 1;
    }
    this.lastStage = stage;
    this.stageHistory.push(stage);
    const stageIndex = THINKING_STAGES.indexOf(stage);
    const progress = Math.min(1.0, (stageIndex + 1) / THINKING_STAGES.length);
    const suggestedAction = this.getSuggestedAction(stage, thoughtNumber, totalThoughts);

    return { current: stage, progress, suggestedAction };
  }


  autoDetectStage(thought: string): ThinkingStage | null {
    const lower = thought.toLowerCase();
    let bestStage: ThinkingStage | null = null;
    let bestScore = 0;

    for (const stage of THINKING_STAGES) {
      const keywords = STAGE_KEYWORDS[stage];
      const score = keywords.reduce((acc, kw) => {
        return acc + (lower.includes(kw) ? 1 : 0);
      }, 0);
      if (score > bestScore) {
        bestScore = score;
        bestStage = stage;
      }
    }
    return bestScore >= 2 ? bestStage : null;
  }


  inferFromProgress(thoughtNumber: number, totalThoughts: number): ThinkingStage {
    const ratio = thoughtNumber / totalThoughts;
    if (ratio <= 0.15) return 'DEFINE';
    if (ratio <= 0.30) return 'RESEARCH';
    if (ratio <= 0.50) return 'ANALYZE';
    if (ratio <= 0.70) return 'HYPOTHESIZE';
    if (ratio <= 0.85) return 'VERIFY';
    return 'SYNTHESIZE';
  }


  private getSuggestedAction(
    stage: ThinkingStage,
    thoughtNumber: number,
    totalThoughts: number,
  ): string | undefined {
    if (this.consecutiveSameStage >= MAX_SAME_STAGE) {
      const nextStageIndex = THINKING_STAGES.indexOf(stage) + 1;
      if (nextStageIndex < THINKING_STAGES.length) {
        return `Consider advancing to ${THINKING_STAGES[nextStageIndex]} stage — ${this.consecutiveSameStage} consecutive thoughts in ${stage}`;
      }
    }
    const ratio = thoughtNumber / totalThoughts;
    if (ratio >= 0.9 && stage !== 'SYNTHESIZE') {
      return 'Approaching final thoughts — consider moving to SYNTHESIZE stage';
    }
    if (stage === 'SYNTHESIZE' && ratio < 0.5) {
      return 'Synthesizing early — ensure sufficient ANALYZE and VERIFY before concluding';
    }

    return undefined;
  }


  getHistory(): ThinkingStage[] {
    return [...this.stageHistory];
  }


  reset(): void {
    this.stageHistory = [];
    this.consecutiveSameStage = 0;
    this.lastStage = null;
  }
}
