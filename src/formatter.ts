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
  if (output.stagnationWarning) {
    lines.push(`⚠️ ${output.stagnationWarning}`);
    lines.push('');
  }

  if (output.biasDetected) {
    lines.push(`🧠 Bias detected: ${output.biasDetected}`);
    if (output.biasSuggestion) {
      lines.push(`   💡 ${output.biasSuggestion}`);
    }
    lines.push('');
  }
  if (output.stage.suggestedAction) {
    lines.push(`💡 ${output.stage.suggestedAction}`);
    lines.push('');
  }
  lines.push(output.nextThoughtNeeded ? '→ Next thought needed' : '✓ Reasoning complete');

  return lines.join('\n');
}

function bar(value: number): string {
  const filled = Math.round(value * 10);
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
