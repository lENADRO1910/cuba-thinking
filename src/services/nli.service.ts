type ZeroShotPipeline = (
  text: string,
  labels: string[],
  options?: Record<string, unknown>,
) => Promise<{ labels: string[]; scores: number[] }>;

let _pipeline: ((task: string, model: string, opts?: Record<string, unknown>) => Promise<unknown>) | null = null;

async function loadTransformers(): Promise<boolean> {
  try {
    const mod = await import('@huggingface/transformers');
    _pipeline = mod.pipeline as unknown as typeof _pipeline;
    return true;
  } catch {
    return false;
  }
}

export interface NLIResult {
  label: 'contradiction' | 'entailment' | 'neutral';
  contradictionScore: number;
}

/**
 * NLI Cross-Encoder service using DeBERTa-v3-xsmall for semantic contradiction detection.
 *
 * Two-stage pipeline:
 *   Stage 1 (caller): Cosine similarity > threshold filters topic overlap (fast, ~1ms)
 *   Stage 2 (this):   NLI cross-encoder classifies contradiction vs entailment (~200ms)
 *
 * Model: Xenova/nli-deberta-v3-xsmall (22M params, ONNX, trained SNLI+MultiNLI)
 */
export class NLIService {
  private classifier: ZeroShotPipeline | null = null;
  private initPromise: Promise<boolean> | null = null;
  private _available = false;
  private _initAttempted = false;

  get isAvailable(): boolean {
    return this._available;
  }

  async ensureReady(): Promise<boolean> {
    if (this._available) return true;
    if (this._initAttempted) return false;

    if (!this.initPromise) {
      this.initPromise = this.init();
    }
    return this.initPromise;
  }

  private async init(): Promise<boolean> {
    this._initAttempted = true;

    const loaded = await loadTransformers();
    if (!loaded || !_pipeline) {
      console.error('[cuba-thinking] NLI: @huggingface/transformers not available');
      return false;
    }

    try {
      this.classifier = await _pipeline(
        'zero-shot-classification',
        'Xenova/nli-deberta-v3-xsmall',
        { dtype: 'q8' },
      ) as unknown as ZeroShotPipeline;

      this._available = true;
      console.error('[cuba-thinking] NLI model loaded: DeBERTa-v3-xsmall (22M params)');
      return true;
    } catch (err) {
      console.error('[cuba-thinking] NLI model load failed:', err);
      return false;
    }
  }

  /**
   * Classify the NLI relationship between two texts.
   *
   * Args:
   *   premise: The reference text (earlier thought)
   *   hypothesis: The text to test against (current thought)
   *
   * Returns:
   *   NLIResult with dominant label and contradiction score
   */
  async classify(premise: string, hypothesis: string): Promise<NLIResult | null> {
    if (!this._available || !this.classifier) return null;

    try {
      // NLI: premise is context, hypothesis is tested against labels
      const combined = `${premise} [SEP] ${hypothesis}`;
      const result = await this.classifier(
        combined,
        ['This is a contradiction', 'This is consistent', 'This is unrelated'],
        { hypothesis_template: 'Based on the context, {}' },
      );

      const contradictionIdx = result.labels.indexOf('This is a contradiction');
      const contradictionScore = contradictionIdx >= 0 ? result.scores[contradictionIdx] : 0;

      const labelMap: Record<string, NLIResult['label']> = {
        'This is a contradiction': 'contradiction',
        'This is consistent': 'entailment',
        'This is unrelated': 'neutral',
      };

      return {
        label: labelMap[result.labels[0]] ?? 'neutral',
        contradictionScore: Math.round(contradictionScore * 100) / 100,
      };
    } catch {
      return null;
    }
  }
}
