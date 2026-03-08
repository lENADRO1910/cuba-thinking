import type {
  ThinkingStage,
  Contradiction,
  ConfidenceCalibration,
} from '../types.js';
import { EmbeddingService, keywordSimilarity } from './embedding.service.js';
import { CONFIDENCE_RANGES } from './stage-engine.service.js';

const NEGATION_WORDS = new Set([
  'not', 'no', 'never', 'incorrect', 'wrong', 'mistake',
  'however', 'actually', 'instead', 'but', 'contrary',
  'different', 'oppose', 'reject', 'disagree', 'false',
  "don't", "doesn't", "shouldn't", "can't", "won't",
  'avoid', 'prevent', 'rather', 'unlike',
]);

const CONTRADICTION_SIM_THRESHOLD = 0.6;

const ASSUMPTION_DEDUP_THRESHOLD = 0.8;

const STAGNATION_SIM_THRESHOLD = 0.85;

const STAGNATION_MIN_COUNT = 3;

export class AntiHallucinationService {
  private allAssumptions: Array<{ thought: number; text: string }> = [];
  private thoughtTexts = new Map<number, string>();

  
  async trackAssumptions(
    assumptions: string[] | undefined,
    thoughtNumber: number,
    embeddings: EmbeddingService,
  ): Promise<string[]> {
    if (!assumptions || assumptions.length === 0) {
      return this.allAssumptions.map((a) => a.text);
    }

    for (const assumption of assumptions) {
      const trimmed = assumption.trim();
      if (!trimmed) continue;
      let isDuplicate = false;
      if (embeddings.isAvailable) {
        for (const existing of this.allAssumptions) {
          const sim = keywordSimilarity(trimmed, existing.text);
          if (sim > ASSUMPTION_DEDUP_THRESHOLD) {
            isDuplicate = true;
            break;
          }
        }
      } else {
        isDuplicate = this.allAssumptions.some(
          (a) => a.text.toLowerCase() === trimmed.toLowerCase(),
        );
      }

      if (!isDuplicate) {
        this.allAssumptions.push({ thought: thoughtNumber, text: trimmed });
      }
    }

    return this.allAssumptions.map((a) => a.text);
  }

  
  async detectContradictions(
    thought: string,
    thoughtNumber: number,
    embeddings: EmbeddingService,
  ): Promise<Contradiction[]> {
    this.thoughtTexts.set(thoughtNumber, thought);
    if (thoughtNumber <= 1) return [];

    const contradictions: Contradiction[] = [];
    const currentLower = thought.toLowerCase();
    for (const [prevNum, prevText] of this.thoughtTexts.entries()) {
      if (prevNum === thoughtNumber) continue;
      let sim: number;
      if (embeddings.isAvailable) {
        sim = embeddings.similarity(prevNum, thoughtNumber);
      } else {
        sim = keywordSimilarity(thought, prevText);
      }
      if (sim > CONTRADICTION_SIM_THRESHOLD) {
        if (hasNegationDifference(currentLower, prevText.toLowerCase())) {
          contradictions.push({
            thoughtA: prevNum,
            thoughtB: thoughtNumber,
            similarity: Math.round(sim * 100) / 100,
            description: `Potential contradiction between thought #${prevNum} and #${thoughtNumber} (similarity: ${(sim * 100).toFixed(0)}% with negation detected)`,
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
