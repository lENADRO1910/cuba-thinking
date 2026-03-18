import { loadTransformersModule } from './transformers-loader.js';

// B2 fix: text-classification pipeline — single forward pass (not ZSC's 3)
type TextClassPipeline = (
  text: string,
  options?: Record<string, unknown>,
) => Promise<Array<{ label: string; score: number }>>;

let _pipeline: ((task: string, model: string, opts?: Record<string, unknown>) => Promise<unknown>) | null = null;

async function loadTransformers(): Promise<boolean> {
  const mod = await loadTransformersModule();
  if (!mod) return false;
  _pipeline = mod.pipeline as unknown as typeof _pipeline;
  return true;
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
  private classifier: TextClassPipeline | null = null;
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
      // B2 fix: text-classification — 1 forward pass instead of ZSC's 3
      this.classifier = await _pipeline(
        'text-classification',
        'Xenova/nli-deberta-v3-xsmall',
        { dtype: 'q8', top_k: 3 },
      ) as unknown as TextClassPipeline;

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
   * B2 fix: Direct text-classification (1 forward pass, ~200ms)
   * instead of zero-shot-classification (3 passes, ~600ms).
   */
  async classify(premise: string, hypothesis: string): Promise<NLIResult | null> {
    if (!this._available || !this.classifier) return null;

    try {
      const combined = `${premise} [SEP] ${hypothesis}`;
      const results = await this.classifier(combined);

      // Map SNLI/MultiNLI output labels to our standard
      const labelMap: Record<string, NLIResult['label']> = {
        contradiction: 'contradiction',
        entailment: 'entailment',
        neutral: 'neutral',
        CONTRADICTION: 'contradiction',
        ENTAILMENT: 'entailment',
        NEUTRAL: 'neutral',
      };

      let contradictionScore = 0;
      let dominantLabel: NLIResult['label'] = 'neutral';
      let dominantScore = 0;

      for (const r of results) {
        const mapped = labelMap[r.label];
        if (!mapped) continue;
        if (mapped === 'contradiction') contradictionScore = r.score;
        if (r.score > dominantScore) {
          dominantScore = r.score;
          dominantLabel = mapped;
        }
      }

      return {
        label: dominantLabel,
        contradictionScore: Math.round(contradictionScore * 100) / 100,
      };
    } catch {
      return null;
    }
  }

  /**
   * Reset the service state.
   * NLI is stateless per-session (no conversation history or caching).
   * The classifier model remains loaded across resets for performance.
   */
  reset(): void {

  }
}
