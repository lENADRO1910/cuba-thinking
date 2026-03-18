#!/usr/bin/env node

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  ListToolsRequestSchema,
  CallToolRequestSchema,
  type Tool,
} from '@modelcontextprotocol/sdk/types.js';

import { CubaThinkingInputSchema } from './types.js';
import { CognitiveProcessor } from './services/cognitive-processor.js';
import { formatResponse } from './formatter.js';

const SERVER_NAME = 'cuba-thinking';
const SERVER_VERSION = '1.2.0';
const processor = new CognitiveProcessor();
const cubaTool: Tool = {
  name: 'cuba_thinking',
  description:
    'Advanced sequential thinking with 6-stage cognitive engine, semantic embeddings, anti-hallucination (assumption tracking, contradiction detection, confidence calibration), 6D quality metrics with trends, and bias detection.',
  inputSchema: {
    type: 'object',
    properties: {
      thought: {
        type: 'string',
        description: 'Your current thinking step',
      },
      thoughtNumber: {
        type: 'number',
        description: 'Current thought number in the sequence',
      },
      totalThoughts: {
        type: 'number',
        description: 'Estimated total thoughts needed (dynamically adjustable)',
      },
      nextThoughtNeeded: {
        type: 'boolean',
        description: 'Whether another thought step is needed',
      },
      thinkingStage: {
        type: 'string',
        enum: ['DEFINE', 'RESEARCH', 'ANALYZE', 'HYPOTHESIZE', 'VERIFY', 'SYNTHESIZE'],
        description:
          'Current cognitive stage (auto-detected if omitted)',
      },
      confidence: {
        type: 'number',
        description: 'Confidence level (0.0-1.0) — calibrated against stage expectations',
      },
      qualityMetrics: {
        type: 'object',
        description: 'Quality ratings 0-5: clarity, depth, breadth, logic, relevance, actionability',
        properties: {
          clarity: { type: 'number', description: 'Clarity 0-5' },
          depth: { type: 'number', description: 'Depth 0-5' },
          breadth: { type: 'number', description: 'Breadth 0-5' },
          logic: { type: 'number', description: 'Logic 0-5' },
          relevance: { type: 'number', description: 'Relevance 0-5' },
          actionability: { type: 'number', description: 'Actionability 0-5' },
        },
      },
      assumptions: {
        type: 'array',
        items: { type: 'string' },
        description: 'Explicit assumptions made — tracked and deduplicated across thoughts',
      },
      hypothesis: {
        type: 'string',
        description: 'Current hypothesis being tested',
      },
      isRevision: {
        type: 'boolean',
        description: 'Whether this revises a previous thought',
      },
      revisesThought: {
        type: 'number',
        description: 'Which thought number is being revised',
      },
      branchFromThought: {
        type: 'number',
        description: 'Branching point for parallel exploration',
      },
      branchId: {
        type: 'string',
        description: 'Identifier for parallel reasoning paths',
      },
      needsMoreThoughts: {
        type: 'boolean',
        description: 'Analysis needs to continue beyond initial estimate',
      },
      budgetMode: {
        type: 'string',
        enum: ['fast', 'balanced', 'thorough', 'exhaustive'],
        description: 'Efficiency mode',
      },
      budgetUsed: {
        type: 'number',
        description: 'Percentage of thinking budget consumed (0-100)',
      },
      biasDetected: {
        type: 'string',
        description:
          'Detected cognitive bias: confirmation, anchoring, availability, overconfidence, sunk_cost',
      },
      parentThoughts: {
        type: 'array',
        items: { type: 'number' },
        description: 'Multiple parent thought references for GoT merge operations',
      },
    },
    required: ['thought', 'thoughtNumber', 'totalThoughts', 'nextThoughtNeeded'],
  },
};
const server = new Server(
  { name: SERVER_NAME, version: SERVER_VERSION },
  { capabilities: { tools: {} } },
);
server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [cubaTool],
}));
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  if (request.params.name !== 'cuba_thinking') {
    return {
      content: [{ type: 'text' as const, text: `Unknown tool: ${request.params.name}` }],
      isError: true,
    };
  }

  try {
    const rawArgs = request.params.arguments ?? {};
    const coerced = coerceInput(rawArgs);
    const parsed = CubaThinkingInputSchema.parse(coerced);
    const result = await processor.process(parsed);

    // V12: Budget-aware MCTS threshold (Kocsis & Szepesvári 2006 — UCB1)
    const mctsThresholds: Record<string, number> = {
      fast: 0.50, balanced: 0.40, thorough: 0.35, exhaustive: 0.30,
    };
    const mctsThreshold = mctsThresholds[parsed.budgetMode ?? 'balanced'] ?? 0.40;

    // MCTS Forced Backtracking: reject thought when EWMA critically low
    if (
      result.ewmaReward !== undefined &&
      result.ewmaReward < mctsThreshold &&
      result.thoughtNumber > 3 &&
      result.bestHistoricalQuality
    ) {
      const best = result.bestHistoricalQuality;
      return {
        content: [{
          type: 'text' as const,
          text:
            `⛔ MCTS BACKTRACK — EWMA Reward ${(result.ewmaReward * 100).toFixed(0)}% < ${(mctsThreshold * 100).toFixed(0)}% threshold\n` +
            `Thought #${result.thoughtNumber} REJECTED at protocol level.\n` +
            `Rollback to thought #${best.thoughtNumber} (quality: ${(best.quality * 100).toFixed(0)}%).\n` +
            `You MUST branch with: branchFromThought: ${best.thoughtNumber}\n` +
            `Explore a COMPLETELY DIFFERENT reasoning path.\n\n` +
            formatResponse(result),
        }],
        isError: true,
      };
    }

    return {
      content: [{ type: 'text' as const, text: formatResponse(result) }],
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Processing failed';
    console.error(`[cuba-thinking] Error: ${message}`);
    return {
      content: [{ type: 'text' as const, text: `Error: ${message}` }],
      isError: true,
    };
  }
});
async function startServer(): Promise<void> {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error(`[cuba-thinking] v${SERVER_VERSION} started`);
  console.error('[cuba-thinking] Embedding model: BGE-small-en-v1.5 (lazy init)');
}

startServer().catch((error) => {
  console.error('[cuba-thinking] Fatal error:', error);
  process.exit(1);
});
process.on('SIGINT', () => {
  console.error('[cuba-thinking] Shutting down (SIGINT)...');
  process.exit(0);
});

process.on('SIGTERM', () => {
  console.error('[cuba-thinking] Shutting down (SIGTERM)...');
  process.exit(0);
});

process.on('unhandledRejection', (reason) => {
  console.error('[cuba-thinking] Unhandled rejection:', reason);
});
function coerceInput(raw: Record<string, unknown>): Record<string, unknown> {
  const result = { ...raw };
  const numericFields = [
    'thoughtNumber', 'totalThoughts', 'confidence',
    'revisesThought', 'branchFromThought', 'budgetUsed',
  ];
  for (const field of numericFields) {
    if (typeof result[field] === 'string') {
      const num = Number(result[field]);
      if (!isNaN(num)) result[field] = num;
    }
  }
  const booleanFields = ['nextThoughtNeeded', 'isRevision', 'needsMoreThoughts'];
  for (const field of booleanFields) {
    if (typeof result[field] === 'string') {
      result[field] = result[field] === 'true';
    }
  }

  return result;
}
