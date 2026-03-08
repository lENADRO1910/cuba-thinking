import { z } from 'zod';
export const THINKING_STAGES = [
  'DEFINE',
  'RESEARCH',
  'ANALYZE',
  'HYPOTHESIZE',
  'VERIFY',
  'SYNTHESIZE',
] as const;

export type ThinkingStage = (typeof THINKING_STAGES)[number];
export const CubaThinkingInputSchema = z.object({
  thought: z.string().min(1).max(50_000)
    .describe('Your current thinking step'),
  thoughtNumber: z.number().int().min(1).max(1000)
    .describe('Current thought number in the sequence'),
  totalThoughts: z.number().int().min(1).max(1000)
    .describe('Estimated total thoughts needed (dynamically adjustable)'),
  nextThoughtNeeded: z.boolean()
    .describe('Whether another thought step is needed'),
  thinkingStage: z.enum(THINKING_STAGES).optional()
    .describe('Current cognitive stage: DEFINE → RESEARCH → ANALYZE → HYPOTHESIZE → VERIFY → SYNTHESIZE'),
  confidence: z.number().min(0).max(1).optional()
    .describe('Confidence level in current reasoning (0.0-1.0)'),
  qualityMetrics: z.object({
    clarity: z.number().min(0).max(5).optional(),
    depth: z.number().min(0).max(5).optional(),
    breadth: z.number().min(0).max(5).optional(),
    logic: z.number().min(0).max(5).optional(),
    relevance: z.number().min(0).max(5).optional(),
    actionability: z.number().min(0).max(5).optional(),
  }).optional().describe('Quality ratings 0-5 across 6 dimensions'),
  assumptions: z.array(z.string().max(500)).max(50).optional()
    .describe('Explicit assumptions made in this thought'),
  hypothesis: z.string().max(2000).optional()
    .describe('Current hypothesis being tested'),
  isRevision: z.boolean().optional()
    .describe('Whether this revises a previous thought'),
  revisesThought: z.number().int().min(1).optional()
    .describe('Which thought number is being revised'),
  branchFromThought: z.number().int().min(1).optional()
    .describe('Branching point for parallel exploration'),
  branchId: z.string().max(100).optional()
    .describe('Identifier for parallel reasoning paths'),
  needsMoreThoughts: z.boolean().optional()
    .describe('Analysis needs to continue beyond initial estimate'),
  budgetMode: z.enum(['fast', 'balanced', 'thorough', 'exhaustive']).optional()
    .describe('Efficiency mode affecting thinking depth'),
  budgetUsed: z.number().min(0).max(100).optional()
    .describe('Percentage of thinking budget consumed (0-100)'),
  biasDetected: z.string().optional()
    .describe('Detected cognitive bias: confirmation, anchoring, availability, overconfidence, sunk_cost'),
});

export type CubaThinkingInput = z.infer<typeof CubaThinkingInputSchema>;
export interface QualityScores {
  clarity: number;
  depth: number;
  breadth: number;
  logic: number;
  relevance: number;
  actionability: number;
  overall: number;
}

export type QualityTrend = 'improving' | 'stable' | 'declining' | 'unstable';

export interface Contradiction {
  thoughtA: number;
  thoughtB: number;
  similarity: number;
  description: string;
}

export interface ConfidenceCalibration {
  status: 'calibrated' | 'overconfident' | 'underconfident';
  expected: { min: number; max: number };
  reported: number;
  warning?: string;
}

export interface StageInfo {
  current: ThinkingStage;
  progress: number;
  suggestedAction?: string;
}

export interface CubaThinkingOutput {
  thought: string;
  thoughtNumber: number;
  totalThoughts: number;
  nextThoughtNeeded: boolean;
  stage: StageInfo;
  quality: QualityScores;
  qualityTrend: QualityTrend;
  assumptions: string[];
  contradictions: Contradiction[];
  confidenceCalibration?: ConfidenceCalibration;
  stagnationWarning?: string;
  biasDetected?: string;
  biasSuggestion?: string;
  relevanceScore?: number;
}
