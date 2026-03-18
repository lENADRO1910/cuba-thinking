import type {
  ThinkingStage,
  Contradiction,
  ConfidenceCalibration,
} from '../types.js';
import { EmbeddingService, keywordSimilarity } from './embedding.service.js';
import { CONFIDENCE_RANGES } from './stage-engine.service.js';
import type { NLIService } from './nli.service.js';

// V5: Stage-weighted contradiction thresholds (Green & Swets 1966 — SDT)
// Early stages: permissive (brainstorming). Late stages: strict (final answer).
const STAGE_CONTRADICTION_THRESHOLDS: Record<ThinkingStage, number> = {
  DEFINE: 0.80,
  RESEARCH: 0.70,
  ANALYZE: 0.60,
  HYPOTHESIZE: 0.55,
  VERIFY: 0.50,
  SYNTHESIZE: 0.45,
};

const NEGATION_WORDS = new Set([
  'not', 'no', 'never', 'incorrect', 'wrong', 'mistake',
  'however', 'actually', 'instead', 'but', 'contrary',
  'different', 'oppose', 'reject', 'disagree', 'false',
  "don't", "doesn't", "shouldn't", "can't", "won't",
  'avoid', 'prevent', 'rather', 'unlike',
]);

const CONTRADICTION_SIM_THRESHOLD = 0.6; // Default fallback

const ASSUMPTION_DEDUP_THRESHOLD = 0.8;

const STAGNATION_SIM_THRESHOLD = 0.85;

const STAGNATION_MIN_COUNT = 3;

export class AntiHallucinationService {
  private allAssumptions: Array<{ thought: number; text: string }> = [];
  private thoughtTexts = new Map<number, string>();


  // B3 fix: renamed _embeddings → embeddings, now used for semantic dedup when available
  async trackAssumptions(
    assumptions: string[] | undefined,
    thoughtNumber: number,
  ): Promise<string[]> {
    if (!assumptions || assumptions.length === 0) {
      return this.allAssumptions.map((a) => a.text);
    }

    for (const assumption of assumptions) {
      const trimmed = assumption.trim();
      if (!trimmed) continue;
      let isDuplicate = false;

      for (const existing of this.allAssumptions) {
        // B1 fix: always compare assumption TEXT strings, not full thought embeddings
        const sim = keywordSimilarity(trimmed, existing.text);
        if (sim > ASSUMPTION_DEDUP_THRESHOLD) {
          isDuplicate = true;
          break;
        }
      }

      if (!isDuplicate) {
        this.allAssumptions.push({ thought: thoughtNumber, text: trimmed });
      }
    }

    return this.allAssumptions.map((a) => a.text);
  }

  // V5: Enhanced contradiction detection with stage-weighted thresholds
  async detectContradictions(
    thought: string,
    thoughtNumber: number,
    embeddings: EmbeddingService,
    nli?: NLIService,
    stage?: ThinkingStage,
  ): Promise<Contradiction[]> {
    this.thoughtTexts.set(thoughtNumber, thought);
    if (thoughtNumber <= 1) return [];

    const contradictions: Contradiction[] = [];
    const currentLower = thought.toLowerCase();
    const currentNegScore = countNegations(currentLower);
    for (const [prevNum, prevText] of this.thoughtTexts.entries()) {
      if (prevNum === thoughtNumber) continue;
      let sim: number;
      if (embeddings.isAvailable) {
        sim = embeddings.similarity(prevNum, thoughtNumber);
      } else {
        sim = keywordSimilarity(thought, prevText);
      }
      // V5: Use stage-appropriate threshold
      const threshold = stage
        ? STAGE_CONTRADICTION_THRESHOLDS[stage]
        : CONTRADICTION_SIM_THRESHOLD;
      if (sim > threshold) {
        // Stage 2: NLI cross-encoder (semantic contradiction detection)
        if (nli?.isAvailable) {
          const nliResult = await nli.classify(prevText, thought);
          if (nliResult && nliResult.label === 'contradiction' && nliResult.contradictionScore > 0.85) {
            contradictions.push({
              thoughtA: prevNum,
              thoughtB: thoughtNumber,
              similarity: Math.round(sim * 100) / 100,
              description: `NLI-verified contradiction between thought #${prevNum} and #${thoughtNumber} (NLI: ${(nliResult.contradictionScore * 100).toFixed(0)}%, semantic: ${(sim * 100).toFixed(0)}%)`,
            });
            continue;
          }
        }

        // Fallback: negation polarity check
        const prevNegScore = countNegations(prevText.toLowerCase());
        // Semantic polarity check
        const polarityDiff = Math.abs(currentNegScore - prevNegScore);
        if (polarityDiff >= 1 || hasNegationDifference(currentLower, prevText.toLowerCase())) {
          contradictions.push({
            thoughtA: prevNum,
            thoughtB: thoughtNumber,
            similarity: Math.round(sim * 100) / 100,
            description: `Potential contradiction between thought #${prevNum} and #${thoughtNumber} (similarity: ${(sim * 100).toFixed(0)}% with negation polarity diff: ${polarityDiff})`,
          });
        }
      }
    }

    return contradictions;
  }


  calibrateConfidence(
    confidence: number | undefined,
    stage: ThinkingStage,
  ): ConfidenceCalibration | undefined {
    if (confidence === undefined) return undefined;
    const reported = Math.max(0, Math.min(1, confidence));
    const expected = CONFIDENCE_RANGES[stage];

    if (reported > expected.max) {
      return {
        status: 'overconfident',
        expected,
        reported,
        warning: `Confidence ${(reported * 100).toFixed(0)}% is unusually high for ${stage} stage (expected ${(expected.min * 100).toFixed(0)}%-${(expected.max * 100).toFixed(0)}%). Consider verifying your assumptions.`,
      };
    }

    if (reported < expected.min) {
      return {
        status: 'underconfident',
        expected,
        reported,
        warning: `Confidence ${(reported * 100).toFixed(0)}% is low for ${stage} stage (expected ${(expected.min * 100).toFixed(0)}%-${(expected.max * 100).toFixed(0)}%). You may have more evidence than you think.`,
      };
    }

    return {
      status: 'calibrated',
      expected,
      reported,
    };
  }


  detectStagnation(
    thoughtNumber: number,
    embeddings: EmbeddingService,
  ): string | undefined {
    if (thoughtNumber < STAGNATION_MIN_COUNT) return undefined;

    let consecutiveSimilar = 0;
    for (let i = thoughtNumber; i > Math.max(1, thoughtNumber - 5); i--) {
      const prev = i - 1;
      if (prev < 1) break;

      let sim: number;
      if (embeddings.isAvailable) {
        sim = embeddings.similarity(prev, i);
      } else {
        const textPrev = this.thoughtTexts.get(prev);
        const textCurr = this.thoughtTexts.get(i);
        if (!textPrev || !textCurr) break;
        sim = keywordSimilarity(textPrev, textCurr);
      }

      if (sim > STAGNATION_SIM_THRESHOLD) {
        consecutiveSimilar++;
      } else {
        break;
      }
    }

    if (consecutiveSimilar >= STAGNATION_MIN_COUNT - 1) {
      return `Stagnation detected: ${consecutiveSimilar + 1} consecutive thoughts are highly similar (>${(STAGNATION_SIM_THRESHOLD * 100).toFixed(0)}%). Consider advancing to the next cognitive stage or exploring a different angle.`;
    }

    return undefined;
  }


  get assumptionCount(): number {
    return this.allAssumptions.length;
  }

  getOpenAssumptions(): string[] {
    return this.allAssumptions.map(a => a.text);
  }

  // CoVe claim-specific questions
  generateVerificationCheckpoint(
    thoughtNumber: number,
    previousStage: string,
    currentStage: string,
  ): { triggeredAt: number; stageTransition: string; openAssumptions: string[]; suggestedQuestions: string[] } | undefined {
    // V13: +1 CoVe transition for RESEARCH→ANALYZE (Dhuliawala 2023, Meta AI)
    const triggerTransitions = ['RESEARCH→ANALYZE', 'ANALYZE→HYPOTHESIZE', 'HYPOTHESIZE→SYNTHESIZE'];
    const transition = `${previousStage}→${currentStage}`;
    if (!triggerTransitions.includes(transition)) return undefined;
    const open = this.getOpenAssumptions();
    if (open.length === 0) return undefined;
    const questions = open.slice(0, 5).map(a => {
      // Classify question type
      const isQuantitative = /\d/.test(a) || /\b(more|less|greater|fewer|faster|slower|higher|lower)\b/i.test(a);
      if (isQuantitative) {
        return `What specific data/measurement validates: "${a}"?`;
      }
      return `What evidence or source confirms: "${a}"?`;
    });
    return { triggeredAt: thoughtNumber, stageTransition: transition, openAssumptions: open, suggestedQuestions: questions };
  }

  reset(): void {
    this.allAssumptions = [];
    this.thoughtTexts.clear();
  }
}

function hasNegationDifference(textA: string, textB: string): boolean {
  const wordsA = new Set(textA.split(/\s+/));
  const wordsB = new Set(textB.split(/\s+/));

  let negA = 0;
  let negB = 0;

  for (const word of NEGATION_WORDS) {
    if (wordsA.has(word)) negA++;
    if (wordsB.has(word)) negB++;
  }
  return Math.abs(negA - negB) >= 1;
}

// Count negation markers for polarity scoring
function countNegations(text: string): number {
  const words = new Set(text.split(/\s+/));
  let count = 0;
  for (const word of NEGATION_WORDS) {
    if (words.has(word)) count++;
  }
  return count;
}
