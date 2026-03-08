import type {
  CubaThinkingOutput,
  QualityTrend,
  QualityScores,
  ConfidenceCalibration,
  FatigueReport,
  StageInfo,
  Contradiction,
} from './types.js';

const TREND_ICONS: Record<QualityTrend, string> = {
  improving: '📈',
  stable: '➡️',
  declining: '📉',
  unstable: '🔀',
};

const STAGE_ICONS: Record<string, string> = {
  DEFINE: '🎯',
  RESEARCH: '🔍',
  ANALYZE: '🔬',
  HYPOTHESIZE: '💡',
  VERIFY: '✅',
  SYNTHESIZE: '📋',
};

/**
 * TD2: formatResponse() decomposed from CC=34 into section renderers.
 * Each sub-function handles one output section independently.
 */
export function formatResponse(output: CubaThinkingOutput): string {
  const lines: string[] = [];
  formatHeader(output, lines);
  formatQuality(output, lines);
  formatAssumptions(output.assumptions, lines);
  formatContradictions(output.contradictions, lines);
  formatCalibration(output.confidenceCalibration, lines);
  formatVerification(output, lines);
  formatWarnings(output, lines);
  formatFatigue(output.fatigue, lines);
  formatBias(output, lines);
  formatGraph(output, lines);
  formatAdvancedMetrics(output, lines);
  formatMemoryHints(output, lines);
  formatSuggestedAction(output.stage, lines);
  lines.push(output.nextThoughtNeeded ? '→ Next thought needed' : '✓ Reasoning complete');
  return lines.join('\n');
}

function formatHeader(output: CubaThinkingOutput, lines: string[]): void {
  const stageIcon = STAGE_ICONS[output.stage.current] ?? '🧠';
  lines.push(`${stageIcon} Cuba-Thinking — Thought #${output.thoughtNumber}/${output.totalThoughts}`);
  lines.push(`Stage: ${output.stage.current} (${(output.stage.progress * 100).toFixed(0)}% cognitive progress)`);
  lines.push('');
  lines.push(output.thought);
  lines.push('');
}

function formatQuality(output: CubaThinkingOutput, lines: string[]): void {
  const q = output.quality;
  const trend = TREND_ICONS[output.qualityTrend];
  lines.push('── Quality ──────────────────────────────────────');
  lines.push(`Overall: ${bar(q.overall)} ${(q.overall * 100).toFixed(0)}% ${trend} ${output.qualityTrend}`);
  formatQualityDimensions(q, lines);
  formatQualityExtras(output, lines);
  lines.push('');
}

function formatQualityDimensions(q: QualityScores, lines: string[]): void {
  lines.push(`  Clarity:       ${bar(q.clarity)} ${(q.clarity * 100).toFixed(0)}%`);
  lines.push(`  Depth:         ${bar(q.depth)} ${(q.depth * 100).toFixed(0)}%`);
  lines.push(`  Breadth:       ${bar(q.breadth)} ${(q.breadth * 100).toFixed(0)}%`);
  lines.push(`  Logic:         ${bar(q.logic)} ${(q.logic * 100).toFixed(0)}%`);
  lines.push(`  Relevance:     ${bar(q.relevance)} ${(q.relevance * 100).toFixed(0)}%`);
  lines.push(`  Actionability: ${bar(q.actionability)} ${(q.actionability * 100).toFixed(0)}%`);
}

function formatQualityExtras(output: CubaThinkingOutput, lines: string[]): void {
  if (output.relevanceScore !== undefined) {
    lines.push(`  Semantic Relevance: ${(output.relevanceScore * 100).toFixed(0)}% (embedding)`);
  }
  if (output.ewmaReward !== undefined) {
    lines.push(`  EWMA Reward:   ${bar(output.ewmaReward)} ${(output.ewmaReward * 100).toFixed(0)}%`);
  }
  if (output.stability !== undefined) {
    lines.push(`  ⚠️ Stability:  ${bar(output.stability)} ${(output.stability * 100).toFixed(0)}% — reasoning is unbalanced`);
  }
}

function formatAssumptions(assumptions: string[], lines: string[]): void {
  if (assumptions.length === 0) return;
  lines.push(`── Assumptions (${assumptions.length}) ──────────────────────`);
  for (const a of assumptions.slice(-5)) {
    lines.push(`  ⚠️ ${a}`);
  }
  if (assumptions.length > 5) {
    lines.push(`  ... and ${assumptions.length - 5} more`);
  }
  lines.push('');
}

function formatContradictions(contradictions: Contradiction[], lines: string[]): void {
  if (contradictions.length === 0) return;
  lines.push('── Contradictions ────────────────────────────────');
  for (const c of contradictions) {
    lines.push(`  🔴 ${c.description}`);
  }
  lines.push('');
}

function formatCalibration(cal: ConfidenceCalibration | undefined, lines: string[]): void {
  if (!cal || cal.status === 'calibrated') return;
  const icon = cal.status === 'overconfident' ? '⬆️' : '⬇️';
  lines.push(`── Confidence Calibration ─────────────────────────`);
  lines.push(`  ${icon} ${cal.warning}`);
  lines.push('');
}

function formatVerification(output: CubaThinkingOutput, lines: string[]): void {
  if (!output.verificationCheckpoint) return;
  const vc = output.verificationCheckpoint;
  lines.push(`── Verification Checkpoint (${vc.stageTransition}) ──────`);
  lines.push(`  Open assumptions: ${vc.openAssumptions.length}`);
  for (const q of vc.suggestedQuestions) {
    lines.push(`  ❓ ${q}`);
  }
  lines.push('');
}

function formatWarnings(output: CubaThinkingOutput, lines: string[]): void {
  if (output.stagnationWarning) {
    lines.push(`⚠️ ${output.stagnationWarning}`);
    lines.push('');
  }
  if (output.overthinkingWarning) {
    lines.push(`⚡ ${output.overthinkingWarning}`);
    lines.push('');
  }
}

function formatFatigue(fatigue: FatigueReport | undefined, lines: string[]): void {
  if (!fatigue) return;
  lines.push(`🧠 Fatigue: ${fatigue.consecutiveDrops} consecutive quality drops → ${fatigue.suggestedAction}`);
  lines.push('');
}

function formatBias(output: CubaThinkingOutput, lines: string[]): void {
  if (!output.biasDetected) return;
  lines.push(`🧠 Bias detected: ${output.biasDetected}`);
  if (output.biasSuggestion) {
    lines.push(`   💡 ${output.biasSuggestion}`);
  }
  lines.push('');
}

function formatGraph(output: CubaThinkingOutput, lines: string[]): void {
  if (output.edges.length === 0) return;
  lines.push(`── Graph (${output.edges.length} edges) ──────────────────────`);
  for (const e of output.edges.slice(-5)) {
    lines.push(`  ${e.from} →[${e.type}]→ ${e.to}`);
  }
  if (output.graphCoherence !== undefined) {
    lines.push(`  Coherence: ${(output.graphCoherence * 100).toFixed(0)}%`);
  }
  if (output.topologyOrphanCount !== undefined) {
    lines.push(`  ⚠️ Orphan thoughts: ${output.topologyOrphanCount}`);
  }
  lines.push('');
}

function formatAdvancedMetrics(output: CubaThinkingOutput, lines: string[]): void {
  if (output.claimDensity !== undefined) {
    lines.push(`📊 Claim density: ${output.claimDensity.toFixed(2)} claims/sentence (${output.claimCount ?? 0} verifiable assertions)`);
  }
  if (output.metacogWarning) {
    lines.push(`💭 ${output.metacogWarning}`);
  }
  if (output.fallacyWarning) {
    lines.push(`⚠️ ${output.fallacyWarning}`);
  }
  if (output.dialecticalWarning) {
    lines.push(`⚖️ ${output.dialecticalWarning}`);
  }
  if (output.reasoningFeedback) {
    lines.push(`🧩 ${output.reasoningFeedback}`);
  }
  if (output.earlyStopSuggestion) {
    lines.push(`🏁 ${output.earlyStopSuggestion}`);
  }
  if (output.confidenceVariance !== undefined) {
    lines.push(`📏 Confidence variance: σ=${output.confidenceVariance.toFixed(2)} — reasoning stability is low`);
  }
}

function formatMemoryHints(output: CubaThinkingOutput, lines: string[]): void {
  // Cortex-Hippocampus Symbiosis: Memory Recall at problem definition
  if (output.stage.current === 'DEFINE' && output.thoughtNumber <= 2) {
    lines.push('');
    lines.push('── Memory Recall ─────────────────────────────────');
    lines.push('💾 Check long-term memory before proceeding:');
    lines.push('  → cuba_faro(query:"[topic]") — search past knowledge');
    lines.push('  → cuba_expediente(query:"[topic]") — check past errors/solutions');
  }

  // Cortex-Hippocampus Symbiosis: Memory Consolidation at conclusion
  if (output.stage.current === 'SYNTHESIZE' && !output.nextThoughtNeeded) {
    lines.push('');
    lines.push('── Memory Consolidation ──────────────────────────');
    lines.push('✓ Reasoning complete and validated.');
    lines.push('💾 Consolidate conclusion into long-term memory:');
    lines.push('  → cuba_cronica(action:"add", entity_name:"[topic]", content:"[conclusion]", observation_type:"lesson", source:"agent")');
  }
}

function formatSuggestedAction(stage: StageInfo, lines: string[]): void {
  if (!stage.suggestedAction) return;
  lines.push(`💡 ${stage.suggestedAction}`);
  lines.push('');
}

function bar(value: number): string {
  const clamped = Math.max(0, Math.min(1, value));
  const filled = Math.round(clamped * 10);
  const empty = 10 - filled;
  return `[${'█'.repeat(filled)}${'░'.repeat(empty)}]`;
}
