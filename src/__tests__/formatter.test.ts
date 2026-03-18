import { formatResponse } from '../formatter';
import { CubaThinkingOutput, QualityScores } from '../types';

describe('formatResponse', () => {
  const defaultQuality: QualityScores = {
    clarity: 0.8,
    depth: 0.8,
    breadth: 0.8,
    logic: 0.8,
    relevance: 0.8,
    actionability: 0.8,
    overall: 0.8,
  };

  const createBaseOutput = (): CubaThinkingOutput => ({
    thought: 'This is a test thought.',
    thoughtNumber: 1,
    totalThoughts: 5,
    nextThoughtNeeded: true,
    stage: { current: 'DEFINE', progress: 0.2 },
    quality: defaultQuality,
    qualityTrend: 'stable',
    assumptions: [],
    contradictions: [],
    edges: [],
  });

  it('formats basic response header and quality correctly', () => {
    const output = createBaseOutput();
    const result = formatResponse(output);

    expect(result).toContain('🎯 Cuba-Thinking — Thought #1/5');
    expect(result).toContain('Stage: DEFINE (20% cognitive progress)');
    expect(result).toContain('This is a test thought.');
    expect(result).toContain('── Quality ──────────────────────────────────────');
    expect(result).toContain('Overall: [████████░░] 80% ➡️ stable');
    expect(result).toContain('Clarity:       [████████░░] 80%');
    expect(result).toContain('→ Next thought needed');
  });

  it('formats conclusion line correctly when no next thought is needed', () => {
    const output = createBaseOutput();
    output.nextThoughtNeeded = false;
    const result = formatResponse(output);

    expect(result).toContain('✓ Reasoning complete');
  });

  it('formats quality extras when provided', () => {
    const output = createBaseOutput();
    output.relevanceScore = 0.95;
    output.ewmaReward = 0.85;
    output.stability = 0.4;
    const result = formatResponse(output);

    expect(result).toContain('Semantic Relevance: 95% (embedding)');
    expect(result).toContain('EWMA Reward:   [█████████░] 85%');
    expect(result).toContain('⚠️ Stability:  [████░░░░░░] 40% — reasoning is unbalanced');
  });

  it('formats assumptions correctly', () => {
    const output = createBaseOutput();
    output.assumptions = [
      'Assumption 1',
      'Assumption 2',
      'Assumption 3',
      'Assumption 4',
      'Assumption 5',
      'Assumption 6',
    ];
    const result = formatResponse(output);

    expect(result).toContain('── Assumptions (6) ──────────────────────');
    expect(result).toContain('⚠️ Assumption 2');
    expect(result).toContain('⚠️ Assumption 6');
    expect(result).toContain('... and 1 more');
  });

  it('formats contradictions correctly', () => {
    const output = createBaseOutput();
    output.contradictions = [
      { thoughtA: 1, thoughtB: 2, similarity: 0.9, description: 'Direct contradiction' }
    ];
    const result = formatResponse(output);

    expect(result).toContain('── Contradictions ────────────────────────────────');
    expect(result).toContain('🔴 Direct contradiction');
  });

  it('formats confidence calibration correctly', () => {
    const output = createBaseOutput();
    output.confidenceCalibration = {
      status: 'overconfident',
      expected: { min: 0.4, max: 0.6 },
      reported: 0.9,
      warning: 'Confidence is too high',
    };
    const result = formatResponse(output);

    expect(result).toContain('── Confidence Calibration ─────────────────────────');
    expect(result).toContain('⬆️ Confidence is too high');
  });

  it('does not format confidence calibration if calibrated', () => {
    const output = createBaseOutput();
    output.confidenceCalibration = {
      status: 'calibrated',
      expected: { min: 0.8, max: 0.9 },
      reported: 0.85,
    };
    const result = formatResponse(output);

    expect(result).not.toContain('Confidence Calibration');
  });

  it('formats verification checkpoint correctly', () => {
    const output = createBaseOutput();
    output.verificationCheckpoint = {
      triggeredAt: 1,
      stageTransition: 'DEFINE → RESEARCH',
      openAssumptions: ['Assum 1'],
      suggestedQuestions: ['What is the meaning of life?'],
    };
    const result = formatResponse(output);

    expect(result).toContain('── Verification Checkpoint (DEFINE → RESEARCH) ──────');
    expect(result).toContain('Open assumptions: 1');
    expect(result).toContain('❓ What is the meaning of life?');
  });

  it('formats warnings correctly', () => {
    const output = createBaseOutput();
    output.stagnationWarning = 'No progress';
    output.overthinkingWarning = 'Too many thoughts';
    const result = formatResponse(output);

    expect(result).toContain('⚠️ No progress');
    expect(result).toContain('⚡ Too many thoughts');
  });

  it('formats fatigue correctly', () => {
    const output = createBaseOutput();
    output.fatigue = {
      fatigueDetected: true,
      consecutiveDrops: 3,
      suggestedAction: 'step_back',
    };
    const result = formatResponse(output);

    expect(result).toContain('🧠 Fatigue: 3 consecutive quality drops → step_back');
  });

  it('formats bias correctly', () => {
    const output = createBaseOutput();
    output.biasDetected = 'confirmation_bias';
    output.biasSuggestion = 'Look for disconfirming evidence';
    const result = formatResponse(output);

    expect(result).toContain('🧠 Bias detected: confirmation_bias');
    expect(result).toContain('💡 Look for disconfirming evidence');
  });

  it('formats graph edges correctly', () => {
    const output = createBaseOutput();
    output.edges = [
      { from: 1, to: 2, type: 'extends' },
      { from: 2, to: 3, type: 'revises' }
    ];
    output.graphCoherence = 0.85;
    output.topologyOrphanCount = 1;
    const result = formatResponse(output);

    expect(result).toContain('── Graph (2 edges) ──────────────────────');
    expect(result).toContain('1 →[extends]→ 2');
    expect(result).toContain('2 →[revises]→ 3');
    expect(result).toContain('Coherence: 85%');
    expect(result).toContain('⚠️ Orphan thoughts: 1');
  });

  it('formats advanced metrics correctly', () => {
    const output = createBaseOutput();
    output.claimDensity = 2.5;
    output.claimCount = 5;
    output.metacogWarning = 'Poor metacognition';
    output.fallacyWarning = 'Slippery slope';
    output.dialecticalWarning = 'Needs counter-arguments';
    output.reasoningFeedback = 'Good logic';
    output.earlyStopSuggestion = 'Stop now';
    output.confidenceVariance = 0.15;
    output.faithfulnessScore = 0.9;
    output.informationGain = 0.6;
    output.groundingWarning = 'Unverified source';
    output.stepCoherenceWarning = 'Abrupt transition';
    output.evidenceWarning = 'Weak evidence';
    output.verbosityWarning = 'Too wordy';
    output.semanticNoveltyWarning = 'Repetitive';
    output.reasoningChainWarning = 'Shallow reasoning';

    const result = formatResponse(output);

    expect(result).toContain('📊 Claim density: 2.50 claims/sentence (5 verifiable assertions)');
    expect(result).toContain('💭 Poor metacognition');
    expect(result).toContain('⚠️ Slippery slope');
    expect(result).toContain('⚖️ Needs counter-arguments');
    expect(result).toContain('🧩 Good logic');
    expect(result).toContain('🏁 Stop now');
    expect(result).toContain('📏 Confidence variance: σ=0.15 — reasoning stability is low');
    expect(result).toContain('🔗 Faithfulness: 90% — semantic alignment with prior thoughts');
    expect(result).toContain('📡 Information gain: 60% new concepts introduced');
    expect(result).toContain('📎 Unverified source');
    expect(result).toContain('🔗 Abrupt transition');
    expect(result).toContain('⚖️ Weak evidence');
    expect(result).toContain('📝 Too wordy');
    expect(result).toContain('🔄 Repetitive');
    expect(result).toContain('🧱 Shallow reasoning');
  });

  it('formats memory recall hints in DEFINE stage', () => {
    const output = createBaseOutput();
    output.stage.current = 'DEFINE';
    output.thoughtNumber = 1;
    const result = formatResponse(output);

    expect(result).toContain('── Memory Recall ─────────────────────────────────');
    expect(result).toContain('💾 Check long-term memory before proceeding');
  });

  it('does not format memory recall hints outside DEFINE stage or after thought 2', () => {
    const output1 = createBaseOutput();
    output1.stage.current = 'RESEARCH';
    output1.thoughtNumber = 1;
    expect(formatResponse(output1)).not.toContain('Memory Recall');

    const output2 = createBaseOutput();
    output2.stage.current = 'DEFINE';
    output2.thoughtNumber = 3;
    expect(formatResponse(output2)).not.toContain('Memory Recall');
  });

  it('formats memory consolidation hints in SYNTHESIZE stage when reasoning is complete', () => {
    const output = createBaseOutput();
    output.stage.current = 'SYNTHESIZE';
    output.nextThoughtNeeded = false;
    const result = formatResponse(output);

    expect(result).toContain('── Memory Consolidation ──────────────────────────');
    expect(result).toContain('💾 Consolidate conclusion into long-term memory');
  });

  it('does not format memory consolidation hints outside SYNTHESIZE stage or if reasoning not complete', () => {
    const output1 = createBaseOutput();
    output1.stage.current = 'VERIFY';
    output1.nextThoughtNeeded = false;
    expect(formatResponse(output1)).not.toContain('Memory Consolidation');

    const output2 = createBaseOutput();
    output2.stage.current = 'SYNTHESIZE';
    output2.nextThoughtNeeded = true;
    expect(formatResponse(output2)).not.toContain('Memory Consolidation');
  });

  it('formats suggested action correctly', () => {
    const output = createBaseOutput();
    output.stage.suggestedAction = 'Review assumptions';
    const result = formatResponse(output);

    expect(result).toContain('💡 Review assumptions');
  });

  it('clamps and formats bars correctly', () => {
    const output = createBaseOutput();
    // Test bar function via overall quality and ewmaReward
    output.quality.overall = 1.5; // Should clamp to 1
    output.ewmaReward = -0.5; // Should clamp to 0
    const result = formatResponse(output);

    expect(result).toContain('Overall: [██████████] 150%'); // Percentage doesn't clamp but bar does
    expect(result).toContain('EWMA Reward:   [░░░░░░░░░░] -50%');
  });
});
