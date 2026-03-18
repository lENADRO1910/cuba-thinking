import { EmbeddingService, keywordSimilarity } from './embedding.service.js';

export type BiasType =
  | 'confirmation_bias'
  | 'anchoring_bias'
  | 'availability_heuristic'
  | 'overconfidence_bias'
  | 'sunk_cost_fallacy';

interface BiasResult {
  type: BiasType;
  suggestion: string;
}

const SUGGESTIONS: Record<BiasType, string> = {
  confirmation_bias:
    'Consider alternative viewpoints or counterarguments. What evidence would disprove your conclusion?',
  anchoring_bias:
    'Re-evaluate without reference to your initial assumption. Consider the problem from scratch.',
  availability_heuristic:
    'Consider the full context, not just recent or memorable information. Are there older data points worth revisiting?',
  overconfidence_bias:
    'Verify your assumptions. What unknowns remain? Consider a wider confidence interval.',
  sunk_cost_fallacy:
    'Focus on future value, not past investment. Would you start this approach from scratch today?',
};

export class BiasDetectorService {
  
  detect(
    thought: string,
    thoughtNumber: number,
    totalThoughts: number,
    confidence: number | undefined,
    history: string[],
    embeddings: EmbeddingService,
    explicitBias?: string,
  ): BiasResult | null {
    if (explicitBias) {
      const type = this.normalizeBiasType(explicitBias);
      if (type) return { type, suggestion: SUGGESTIONS[type] };
    }

    const lower = thought.toLowerCase();
    const progress = thoughtNumber / totalThoughts;
    if (history.length >= 3) {
      const recent = history.slice(-3);
      const currentFreq = embeddings.getFrequencyMap(thoughtNumber, thought);
      const similarities = recent.map((h, i) => {
        const histNum = history.length - 2 + i;
        const histFreq = embeddings.getFrequencyMap(histNum, h);
        return keywordSimilarity(h, thought, histFreq, currentFreq);
      });
      const avgSim = similarities.reduce((a, b) => a + b, 0) / similarities.length;
      if (avgSim > 0.7) {
        return {
          type: 'confirmation_bias',
          suggestion: SUGGESTIONS.confirmation_bias,
        };
      }
    }
    if (thoughtNumber > 5 && history.length > 0) {
      const anchoringWords = ['initially', 'originally', 'first thought', 'started with', 'began'];
      const hasAnchoring = anchoringWords.some((w) => lower.includes(w));
      if (hasAnchoring) {
        const currentFreq = embeddings.getFrequencyMap(thoughtNumber, thought);
        const firstFreq = embeddings.getFrequencyMap(1, history[0]);
        const simToFirst = keywordSimilarity(thought, history[0], currentFreq, firstFreq);
        if (simToFirst > 0.6) {
          return {
            type: 'anchoring_bias',
            suggestion: SUGGESTIONS.anchoring_bias,
          };
        }
      }
    }
    if (confidence !== undefined && confidence > 0.9 && progress < 0.5) {
      return {
        type: 'overconfidence_bias',
        suggestion: SUGGESTIONS.overconfidence_bias,
      };
    }
    if (thoughtNumber > 10) {
      const recentWords = ['recent', 'just', 'last', 'latest', 'just saw'];
      const count = recentWords.filter((w) => lower.includes(w)).length;
      if (count >= 2) {
        return {
          type: 'availability_heuristic',
          suggestion: SUGGESTIONS.availability_heuristic,
        };
      }
    }
    if (progress > 0.7) {
      const sunkWords = ['already', 'so far', 'invested', 'spent'];
      const contWords = ['continue', 'proceed', 'keep going', 'press on'];
      const hasSunk = sunkWords.some((w) => lower.includes(w));
      const hasCont = contWords.some((w) => lower.includes(w));
      if (hasSunk && hasCont) {
        return {
          type: 'sunk_cost_fallacy',
          suggestion: SUGGESTIONS.sunk_cost_fallacy,
        };
      }
    }

    return null;
  }

  
  private normalizeBiasType(bias: string): BiasType | null {
    const lower = bias.toLowerCase().replace(/[^a-z_]/g, '_');
    const mapping: Record<string, BiasType> = {
      confirmation: 'confirmation_bias',
      confirmation_bias: 'confirmation_bias',
      anchoring: 'anchoring_bias',
      anchoring_bias: 'anchoring_bias',
      availability: 'availability_heuristic',
      availability_heuristic: 'availability_heuristic',
      overconfidence: 'overconfidence_bias',
      overconfidence_bias: 'overconfidence_bias',
      sunk_cost: 'sunk_cost_fallacy',
      sunk_cost_fallacy: 'sunk_cost_fallacy',
    };
    return mapping[lower] ?? null;
  }
}
