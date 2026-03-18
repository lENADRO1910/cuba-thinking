import { loadTransformersModule } from './transformers-loader.js';

type Pipeline = (text: string, options?: Record<string, unknown>) => Promise<{ data: ArrayLike<number> }>;
type CosSim = (a: ArrayLike<number>, b: ArrayLike<number>) => number;

let _pipeline: ((task: string, model: string, opts?: Record<string, unknown>) => Promise<Pipeline>) | null = null;
let _cos_sim: CosSim | null = null;

async function loadTransformers(): Promise<boolean> {
  const mod = await loadTransformersModule();
  if (!mod) return false;
  _pipeline = mod.pipeline as unknown as typeof _pipeline;
  _cos_sim = mod.cos_sim as unknown as CosSim;
  return true;
}

export class EmbeddingService {
  private extractor: Pipeline | null = null;
  private cache = new Map<number, Float32Array>();
  private freqCache = new Map<number, { freq: Map<string, number>; norm: number }>();
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
      console.error('[cuba-thinking] @huggingface/transformers not available — using keyword fallback');
      return false;
    }

    try {
      this.extractor = await _pipeline(
        'feature-extraction',
        'Xenova/bge-small-en-v1.5',
        { dtype: 'q8' }
      ) as unknown as Pipeline;

      this._available = true;
      console.error('[cuba-thinking] Embedding model loaded: BGE-small-en-v1.5 (384d)');
      return true;
    } catch (err) {
      console.error('[cuba-thinking] Embedding model load failed:', err);
      return false;
    }
  }

  
  async embed(text: string, thoughtNumber: number): Promise<Float32Array | null> {
    if (!this._available || !this.extractor) return null;
    if (!text.trim()) return null;
    const cached = this.cache.get(thoughtNumber);
    if (cached) return cached;

    try {
      const output = await this.extractor(text, {
        pooling: 'mean',
        normalize: true,
      });
      const embedding = Float32Array.from(output.data as unknown as number[]);
      this.cache.set(thoughtNumber, embedding);
      return embedding;
    } catch {
      return null;
    }
  }

  
  similarity(a: number, b: number): number {
    const embA = this.cache.get(a);
    const embB = this.cache.get(b);
    if (!embA || !embB) return 0;

    if (_cos_sim) {
      return _cos_sim(embA, embB);
    }
    return manualCosineSim(embA, embB);
  }

  // V11: Sliding-window relevance (Mitra 1998 — query drift mitigation)
  relevance(thoughtNumber: number): number {
    if (thoughtNumber <= 3) return this.similarity(1, thoughtNumber);
    const window = [thoughtNumber - 3, thoughtNumber - 2, thoughtNumber - 1];
    const sliding = window.reduce((s, i) => s + this.similarity(i, thoughtNumber), 0) / 3;
    return Math.max(sliding, this.similarity(1, thoughtNumber));
  }

  
  clearCache(): void {
    this.cache.clear();
    this.freqCache.clear();
  }

  
  get cacheSize(): number {
    return this.cache.size;
  }

  getFrequencyMap(thoughtNumber: number, text: string): { freq: Map<string, number>; norm: number } {
    const cached = this.freqCache.get(thoughtNumber);
    if (cached) return cached;

    const tokens = tokenize(text);
    const freq = frequencyMap(tokens);
    let norm = 0;
    freq.forEach((count) => {
      norm += count * count;
    });
    const entry = { freq, norm };
    this.freqCache.set(thoughtNumber, entry);
    return entry;
  }
}

export function keywordSimilarity(
  textA: string,
  textB: string,
  freqA?: { freq: Map<string, number>; norm: number },
  freqB?: { freq: Map<string, number>; norm: number },
): number {
  if (textA === textB) return 1.0;
  if (!textA.trim() || !textB.trim()) return 0.0;

  const fA = freqA || { freq: frequencyMap(tokenize(textA)), norm: 0 };
  const fB = freqB || { freq: frequencyMap(tokenize(textB)), norm: 0 };

  if (fA.freq.size === 0 || fB.freq.size === 0) return 0.0;

  let dotProduct = 0;
  fA.freq.forEach((countA, token) => {
    const countB = fB.freq.get(token);
    if (countB !== undefined) {
      dotProduct += countA * countB;
    }
  });

  let normA = fA.norm;
  if (normA === 0) {
    fA.freq.forEach((countA) => { normA += countA * countA; });
  }

  let normB = fB.norm;
  if (normB === 0) {
    fB.freq.forEach((countB) => { normB += countB * countB; });
  }

  if (normA === 0 || normB === 0) return 0;
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

// V9: Stopword filtering (Kaur & Buttar 2023 — +16% MAP improvement)
const STOPWORDS = new Set([
  'the', 'is', 'a', 'an', 'in', 'on', 'of', 'and', 'to', 'for',
  'it', 'that', 'this', 'with', 'as', 'be', 'was', 'are', 'or',
  'by', 'at', 'from', 'they', 'we', 'my', 'your', 'its', 'not',
  'but', 'if', 'so', 'do', 'has', 'had', 'have', 'been', 'would',
  'could', 'should', 'will', 'can', 'may', 'about', 'up', 'out',
]);

function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^\w\s]/g, '')
    .split(/\s+/)
    .filter((t) => t.length > 1 && !STOPWORDS.has(t));
}

function frequencyMap(tokens: string[]): Map<string, number> {
  const freq = new Map<string, number>();
  for (const token of tokens) {
    freq.set(token, (freq.get(token) ?? 0) + 1);
  }
  return freq;
}

function manualCosineSim(a: Float32Array, b: Float32Array): number {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  if (normA < 1e-9 || normB < 1e-9) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}
