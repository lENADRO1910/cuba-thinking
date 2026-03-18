/**
 * Shared loader for @huggingface/transformers.
 * Consolidates TD4: both EmbeddingService and NLIService used
 * independent loadTransformers() functions.
 */

// Lazy-loaded module reference
let _mod: Record<string, unknown> | null = null;
let _loadAttempted = false;

/**
 * Dynamically import @huggingface/transformers.
 * Returns the module or null if not available.
 * Module is loaded at most once (idempotent).
 */
export async function loadTransformersModule(): Promise<Record<string, unknown> | null> {
  if (_mod) return _mod;
  if (_loadAttempted) return null;
  _loadAttempted = true;

  try {
    _mod = await import('@huggingface/transformers') as Record<string, unknown>;
    return _mod;
  } catch {
    return null;
  }
}
