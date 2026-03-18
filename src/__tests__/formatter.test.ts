import { formatResponse } from '../formatter';
import type { CubaThinkingOutput } from '../types';

describe('formatter', () => {
  const baseOutput: CubaThinkingOutput = {
    thought: 'This is a test thought.',
    thoughtNumber: 1,
    totalThoughts: 5,
    nextThoughtNeeded: true,
    stage: {
      current: 'DEFINE',
      progress: 0.1,
    },
    quality: {
      clarity: 0.8,
      depth: 0.5,
      breadth: 0.5,
      logic: 0.9,
      relevance: 1.0,
      actionability: 0.2,
      overall: 0.7,
    },
    qualityTrend: 'stable',
    assumptions: [],
    contradictions: [],
    edges: [],
  };

  it('formats a basic output correctly', () => {
    const result = formatResponse(baseOutput);

    expect(result).toContain('🎯 Cuba-Thinking — Thought #1/5');
    expect(result).toContain('Stage: DEFINE (10% cognitive progress)');
    expect(result).toContain('This is a test thought.');
    expect(result).toContain('── Quality ──────────────────────────────────────');
    expect(result).toContain('Overall: [███████░░░] 70% ➡️ stable');
    expect(result).toContain('→ Next thought needed');
  });

  it('formats quality dimensions with clamping correctly', () => {
    const outputWithExtremeQuality: CubaThinkingOutput = {
      ...baseOutput,
      quality: {
        clarity: 1.5, // Should clamp to 1
        depth: -0.5, // Should clamp to 0
        breadth: 0.55, // Should round to 6
        logic: 0.5,
        relevance: 0.5,
        actionability: 0.5,
        overall: 0.5,
      }
    };

    const result = formatResponse(outputWithExtremeQuality);

    expect(result).toContain('Clarity:       [██████████] 150%'); // the value display uses * 100 before clamp
    expect(result).toContain('Depth:         [░░░░░░░░░░] -50%');
    // Note: The bar itself uses clamped value for display, but the text might show raw * 100
  });

  it('formats advanced metrics and warnings', () => {
    const richOutput: CubaThinkingOutput = {
      ...baseOutput,
      nextThoughtNeeded: false,
      stage: {
        current: 'SYNTHESIZE',
        progress: 1.0,
        suggestedAction: 'Conclude findings'
      },
      qualityTrend: 'improving',
      assumptions: ['Assumption 1', 'Assumption 2', 'Assumption 3', 'Assumption 4', 'Assumption 5', 'Assumption 6'],
      contradictions: [{ thoughtA: 1, thoughtB: 2, similarity: 0.9, description: 'Opposing views' }],
      confidenceCalibration: { status: 'overconfident', expected: { min: 0.4, max: 0.6 }, reported: 0.9, warning: 'Too certain' },
      verificationCheckpoint: {
        triggeredAt: 1,
        stageTransition: 'DEFINE->RESEARCH',
        openAssumptions: ['A1'],
        suggestedQuestions: ['What next?']
      },
      stagnationWarning: 'Looping thoughts',
      overthinkingWarning: 'Too deep',
      fatigue: { fatigueDetected: true, consecutiveDrops: 3, suggestedAction: 'step_back' },
      biasDetected: 'Confirmation bias',
      biasSuggestion: 'Look for opposing evidence',
      edges: [{ from: 1, to: 2, type: 'extends' }],
      graphCoherence: 0.85,
      topologyOrphanCount: 1,
      claimDensity: 2.5,
      claimCount: 5,
      metacogWarning: 'Meta warning',
      fallacyWarning: 'Fallacy detected',
      dialecticalWarning: 'One sided',
      reasoningFeedback: 'Good job',
      earlyStopSuggestion: 'Stop now',
      confidenceVariance: 0.15,
      faithfulnessScore: 0.92,
      informationGain: 0.45,
      groundingWarning: 'Not grounded',
      stepCoherenceWarning: 'Jump in logic',
      evidenceWarning: 'Weak evidence',
      verbosityWarning: 'Too wordy',
      semanticNoveltyWarning: 'Repetitive',
      reasoningChainWarning: 'Shallow chain',
      relevanceScore: 0.88,
      ewmaReward: 0.75,
      stability: 0.4,
    };

    const result = formatResponse(richOutput);

    // Advanced Metrics & Quality Extras
    expect(result).toContain('Semantic Relevance: 88% (embedding)');
    expect(result).toContain('EWMA Reward:   [████████░░] 75%');
    expect(result).toContain('⚠️ Stability:  [████░░░░░░] 40% — reasoning is unbalanced');

    // Assumptions
    expect(result).toContain('── Assumptions (6) ──────────────────────');
    expect(result).toContain('... and 1 more');

    // Contradictions
    expect(result).toContain('── Contradictions ────────────────────────────────');
    expect(result).toContain('🔴 Opposing views');

    // Calibration
    expect(result).toContain('── Confidence Calibration ─────────────────────────');
    expect(result).toContain('⬆️ Too certain');

    // Verification
    expect(result).toContain('── Verification Checkpoint (DEFINE->RESEARCH) ──────');
    expect(result).toContain('Open assumptions: 1');
    expect(result).toContain('❓ What next?');

    // Warnings
    expect(result).toContain('⚠️ Looping thoughts');
    expect(result).toContain('⚡ Too deep');

    // Fatigue & Bias
    expect(result).toContain('🧠 Fatigue: 3 consecutive quality drops → step_back');
    expect(result).toContain('🧠 Bias detected: Confirmation bias');
    expect(result).toContain('💡 Look for opposing evidence');

    // Graph
    expect(result).toContain('── Graph (1 edges) ──────────────────────');
    expect(result).toContain('1 →[extends]→ 2');
    expect(result).toContain('Coherence: 85%');
    expect(result).toContain('⚠️ Orphan thoughts: 1');

    // Additional Metrics
    expect(result).toContain('📊 Claim density: 2.50 claims/sentence (5 verifiable assertions)');
    expect(result).toContain('💭 Meta warning');
    expect(result).toContain('⚠️ Fallacy detected');
    expect(result).toContain('⚖️ One sided');
    expect(result).toContain('🧩 Good job');
    expect(result).toContain('🏁 Stop now');
    expect(result).toContain('📏 Confidence variance: σ=0.15 — reasoning stability is low');
    expect(result).toContain('🔗 Faithfulness: 92% — semantic alignment with prior thoughts');
    expect(result).toContain('📡 Information gain: 45% new concepts introduced');
    expect(result).toContain('📎 Not grounded');
    expect(result).toContain('🔗 Jump in logic');
    expect(result).toContain('⚖️ Weak evidence');
    expect(result).toContain('📝 Too wordy');
    expect(result).toContain('🔄 Repetitive');
    expect(result).toContain('🧱 Shallow chain');

    // Stage & Completion
    expect(result).toContain('💡 Conclude findings');
    expect(result).toContain('✓ Reasoning complete');
  });

  it('formats memory hints appropriately for DEFINE stage early on', () => {
    const defineOutput: CubaThinkingOutput = {
      ...baseOutput,
      thoughtNumber: 1,
      stage: { current: 'DEFINE', progress: 0.1 }
    };

    const result = formatResponse(defineOutput);
    expect(result).toContain('── Memory Recall ─────────────────────────────────');
    expect(result).toContain('cuba_faro(query:"[topic]")');
  });

  it('formats memory consolidation appropriately for SYNTHESIZE stage at conclusion', () => {
    const synthesizeOutput: CubaThinkingOutput = {
      ...baseOutput,
      nextThoughtNeeded: false,
      stage: { current: 'SYNTHESIZE', progress: 1.0 }
    };

    const result = formatResponse(synthesizeOutput);
    expect(result).toContain('── Memory Consolidation ──────────────────────────');
    expect(result).toContain('cuba_cronica(action:"add", entity_name:"[topic]"');
  });
});
