import type { QualityScores, QualityTrend, ThinkingStage } from '../types.js';
import { STAGE_QUALITY_WEIGHTS } from './stage-engine.service.js';

const MIN_TREND_POINTS = 3;

const TREND_WINDOW = 10;

export class QualityMetricsService {
  private history: number[] = [];

  
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

  
  reset(): void {
    this.history = [];
  }

  
  getHistory(): number[] {
    return [...this.history];
  }

  private evalClarity(thought: string): number {
    if (!thought.trim()) return 0;
    const sentences = thought.split(/[.!?]+/).filter((s) => s.trim().length > 0);
    const avgLength = thought.length / Math.max(sentences.length, 1);

    let score = 0.5;
    if (avgLength < 100) score += 0.2;
    if (avgLength < 50) score += 0.1;
    if (sentences.length >= 2) score += 0.1;
    if (thought.includes('```') || thought.includes('- ') || thought.includes('1.')) score += 0.1;

    return clamp(score);
  }

  private evalDepth(thought: string): number {
    if (!thought.trim()) return 0;
    let score = 0.4;
    const lower = thought.toLowerCase();

    if (thought.length > 200) score += 0.15;
    if (thought.length > 500) score += 0.1;
    const depthWords = ['because', 'therefore', 'since', 'implies', 'consequence', 'root cause', 'specifically'];
    score += Math.min(0.3, depthWords.filter((w) => lower.includes(w)).length * 0.1);

    return clamp(score);
  }

  private evalBreadth(thought: string): number {
    if (!thought.trim()) return 0;
    let score = 0.4;
    const lower = thought.toLowerCase();
    const breadthWords = ['also', 'another', 'additionally', 'furthermore', 'alternative', 'option', 'versus', 'compared'];
    score += Math.min(0.3, breadthWords.filter((w) => lower.includes(w)).length * 0.1);
    const listItems = (thought.match(/^[-•*]\s/gm) || []).length;
    if (listItems >= 2) score += 0.15;
    if (listItems >= 4) score += 0.1;

    return clamp(score);
  }

  private evalLogic(thought: string): number {
    if (!thought.trim()) return 0;
    let score = 0.5;
    const lower = thought.toLowerCase();

    const logicWords = ['if', 'then', 'because', 'therefore', 'however', 'but', 'although', 'thus', 'hence'];
    score += Math.min(0.3, logicWords.filter((w) => lower.includes(w)).length * 0.075);
    if (lower.includes('on the other hand') || lower.includes('conversely')) score += 0.1;

    return clamp(score);
  }

  private evalRelevance(thought: string): number {
    if (!thought.trim()) return 0;
    return 0.6;
  }

  private evalActionability(thought: string): number {
    if (!thought.trim()) return 0;
    let score = 0.4;
    const lower = thought.toLowerCase();

    const actionWords = ['implement', 'create', 'build', 'use', 'apply', 'configure', 'install', 'run', 'execute', 'add'];
    score += Math.min(0.3, actionWords.filter((w) => lower.includes(w)).length * 0.075);
    if (thought.includes('```')) score += 0.15;
    if (thought.includes('/') || thought.includes('.ts') || thought.includes('.py')) score += 0.1;

    return clamp(score);
  }
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
