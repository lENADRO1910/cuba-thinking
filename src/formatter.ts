import type { CubaThinkingOutput, QualityTrend, ConfidenceCalibration } from './types.js';

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

export function formatResponse(output: CubaThinkingOutput): string {
  const lines: string[] = [];
  const stageIcon = STAGE_ICONS[output.stage.current] ?? '🧠';
  lines.push(`${stageIcon} Cuba-Thinking — Thought #${output.thoughtNumber}/${output.totalThoughts}`);
  lines.push(`Stage: ${output.stage.current} (${(output.stage.progress * 100).toFixed(0)}% cognitive progress)`);
  lines.push('');
  lines.push(output.thought);
  lines.push('');
  const q = output.quality;
  const trend = TREND_ICONS[output.qualityTrend];
  lines.push('── Quality ──────────────────────────────────────');
  lines.push(`Overall: ${bar(q.overall)} ${(q.overall * 100).toFixed(0)}% ${trend} ${output.qualityTrend}`);
  lines.push(`  Clarity:       ${bar(q.clarity)} ${(q.clarity * 100).toFixed(0)}%`);
  lines.push(`  Depth:         ${bar(q.depth)} ${(q.depth * 100).toFixed(0)}%`);
  lines.push(`  Breadth:       ${bar(q.breadth)} ${(q.breadth * 100).toFixed(0)}%`);
  lines.push(`  Logic:         ${bar(q.logic)} ${(q.logic * 100).toFixed(0)}%`);
  lines.push(`  Relevance:     ${bar(q.relevance)} ${(q.relevance * 100).toFixed(0)}%`);
  lines.push(`  Actionability: ${bar(q.actionability)} ${(q.actionability * 100).toFixed(0)}%`);

  if (output.relevanceScore !== undefined) {
    lines.push(`  Semantic Relevance: ${(output.relevanceScore * 100).toFixed(0)}% (embedding)`);
  }

  if (output.ewmaReward !== undefined) {
    lines.push(`  EWMA Reward:   ${bar(output.ewmaReward)} ${(output.ewmaReward * 100).toFixed(0)}%`);
  }

  if (output.stability !== undefined) {
    lines.push(`  ⚠️ Stability:  ${bar(output.stability)} ${(output.stability * 100).toFixed(0)}% — reasoning is unbalanced`);
  }

  lines.push('');

  if (output.assumptions.length > 0) {
    lines.push(`── Assumptions (${output.assumptions.length}) ──────────────────────`);
    for (const a of output.assumptions.slice(-5)) {
      lines.push(`  ⚠️ ${a}`);
    }
    if (output.assumptions.length > 5) {
      lines.push(`  ... and ${output.assumptions.length - 5} more`);
    }
    lines.push('');
  }

  if (output.contradictions.length > 0) {
    lines.push('── Contradictions ────────────────────────────────');
    for (const c of output.contradictions) {
      lines.push(`  🔴 ${c.description}`);
    }
    lines.push('');
  }

  if (output.confidenceCalibration) {
    formatCalibration(output.confidenceCalibration, lines);
  }

  if (output.verificationCheckpoint) {
    const vc = output.verificationCheckpoint;
    lines.push(`── Verification Checkpoint (${vc.stageTransition}) ──────`);
    lines.push(`  Open assumptions: ${vc.openAssumptions.length}`);
    for (const q of vc.suggestedQuestions) {
      lines.push(`  ❓ ${q}`);
    }
    lines.push('');
  }

  if (output.stagnationWarning) {
    lines.push(`⚠️ ${output.stagnationWarning}`);
    lines.push('');
  }

  if (output.overthinkingWarning) {
    lines.push(`⚡ ${output.overthinkingWarning}`);
    lines.push('');
  }

  if (output.fatigue) {
    const f = output.fatigue;
    lines.push(`🧠 Fatigue: ${f.consecutiveDrops} consecutive quality drops → ${f.suggestedAction}`);
    lines.push('');
  }

  if (output.biasDetected) {
    lines.push(`🧠 Bias detected: ${output.biasDetected}`);
    if (output.biasSuggestion) {
      lines.push(`   💡 ${output.biasSuggestion}`);
    }
    lines.push('');
  }

  if (output.edges.length > 0) {
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

  if (output.stage.suggestedAction) {
    lines.push(`💡 ${output.stage.suggestedAction}`);
    lines.push('');
  }
  lines.push(output.nextThoughtNeeded ? '→ Next thought needed' : '✓ Reasoning complete');

  return lines.join('\n');
}

function bar(value: number): string {
  const clamped = Math.max(0, Math.min(1, value));
  const filled = Math.round(clamped * 10);
  const empty = 10 - filled;
  return `[${'█'.repeat(filled)}${'░'.repeat(empty)}]`;
}

function formatCalibration(cal: ConfidenceCalibration, lines: string[]): void {
  if (cal.status === 'calibrated') return;

  const icon = cal.status === 'overconfident' ? '⬆️' : '⬇️';
  lines.push(`── Confidence Calibration ─────────────────────────`);
  lines.push(`  ${icon} ${cal.warning}`);
  lines.push('');
}
