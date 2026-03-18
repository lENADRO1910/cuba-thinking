import { jest } from '@jest/globals';

const mockLoadTransformersModule = jest.fn();

jest.unstable_mockModule('./transformers-loader.js', () => {
  return {
    loadTransformersModule: mockLoadTransformersModule
  };
});

// Since top-level await is supported in ESM, we must await the import
const { keywordSimilarity, EmbeddingService } = await import('./embedding.service.js');

describe('keywordSimilarity', () => {
  it('should return 1.0 for identical strings', () => {
    expect(keywordSimilarity('hello world', 'hello world')).toBe(1.0);
  });

  it('should return 0.0 for completely different strings', () => {
    expect(keywordSimilarity('hello world', 'goodbye moon')).toBe(0.0);
  });

  it('should return 0.0 for empty strings', () => {
    expect(keywordSimilarity('', '')).toBe(1.0); // handled by if (textA === textB) return 1.0;
    expect(keywordSimilarity('hello', '')).toBe(0.0);
    expect(keywordSimilarity('', 'hello')).toBe(0.0);
    expect(keywordSimilarity('   ', '   ')).toBe(1.0); // handled by if (textA === textB) return 1.0;
  });

  it('should filter out stopwords appropriately and ignore punctuation', () => {
    const textA = "The quick brown fox jumps over the lazy dog!";
    const textB = "A quick brown fox jumped over a lazy dog.";
    const similarity = keywordSimilarity(textA, textB);
    expect(similarity).toBeGreaterThan(0.7);
    expect(similarity).toBeLessThan(1.0);
  });

  it('should handle strings that become empty after stopword filtering', () => {
    const similarity = keywordSimilarity('the is a', 'it that this');
    expect(similarity).toBe(0.0); // Tokens are empty after filter
  });

  it('should be case-insensitive', () => {
    expect(keywordSimilarity('HELLO WORLD', 'hello world')).toBeCloseTo(1.0, 5);
  });
});

describe('EmbeddingService', () => {
  let service: any; // Using any because of dynamic import type resolution issues in test
  let mockConsoleError: any;

  beforeEach(() => {
    service = new EmbeddingService();
    // clear all mocks between tests
    jest.clearAllMocks();
    mockLoadTransformersModule.mockReset();

    // suppress console.error for tests
    mockConsoleError = jest.spyOn(console, 'error').mockImplementation(() => {});
  });

  afterEach(() => {
    if (mockConsoleError) mockConsoleError.mockRestore();
  });

  it('should initialize successfully when transformers are available', async () => {
    // Mock successful pipeline load
    const mockExtractor = jest.fn().mockResolvedValue({ data: new Float32Array([0.1, 0.2, 0.3]) } as never);
    const mockPipeline = jest.fn().mockResolvedValue(mockExtractor as never);
    const mockCosSim = jest.fn().mockReturnValue(0.9);

    mockLoadTransformersModule.mockResolvedValue({
      pipeline: mockPipeline,
      cos_sim: mockCosSim
    } as never);

    const ready = await service.ensureReady();
    expect(ready).toBe(true);
    expect(service.isAvailable).toBe(true);

    // A second call to ensureReady should return true without re-initializing
    const readyAgain = await service.ensureReady();
    expect(readyAgain).toBe(true);
    expect(mockPipeline).toHaveBeenCalledTimes(1);
  });

  it('should fallback to unavailable if transformers fail to load', async () => {
    // Mock failure
    mockLoadTransformersModule.mockResolvedValue(null as never);

    const ready = await service.ensureReady();
    expect(ready).toBe(false);
    expect(service.isAvailable).toBe(false);
  });

  it('should handle embed requests by caching successful queries', async () => {
    const mockExtractor = jest.fn().mockResolvedValue({ data: [0.5, 0.5, 0.5] } as never);
    const mockPipeline = jest.fn().mockResolvedValue(mockExtractor as never);

    mockLoadTransformersModule.mockResolvedValue({
      pipeline: mockPipeline,
      cos_sim: jest.fn()
    } as never);

    await service.ensureReady();

    const embedding = await service.embed("test text", 1);
    expect(embedding).toBeInstanceOf(Float32Array);
    expect(service.cacheSize).toBe(1);

    // Should return from cache on next call with same thoughtNumber
    const cachedEmbedding = await service.embed("test text", 1);
    expect(cachedEmbedding).toBe(embedding);
    expect(mockExtractor).toHaveBeenCalledTimes(1); // not called again

    service.clearCache();
    expect(service.cacheSize).toBe(0);
  });

  it('should return null for embed when unavailable', async () => {
    mockLoadTransformersModule.mockResolvedValue(null as never);
    await service.ensureReady();

    const embedding = await service.embed("test text", 1);
    expect(embedding).toBeNull();
  });

  it('should calculate relevance using sliding window for later thoughts', async () => {
    let callCount = 0;
    const mockExtractor = jest.fn().mockImplementation(() => {
      callCount++;
      return Promise.resolve({ data: [0.1 * callCount, 0.2 * callCount] });
    });

    const mockPipeline = jest.fn().mockResolvedValue(mockExtractor as never);
    // return a fixed cos_sim for test
    const mockCosSim = jest.fn().mockReturnValue(0.85);

    mockLoadTransformersModule.mockResolvedValue({
      pipeline: mockPipeline,
      cos_sim: mockCosSim
    } as never);

    await service.ensureReady();

    // Populate cache with dummy thoughts
    await service.embed("thought 1", 1);
    await service.embed("thought 2", 2);
    await service.embed("thought 3", 3);
    await service.embed("thought 4", 4);

    const relevance = service.relevance(4);

    expect(relevance).toBeCloseTo(0.85, 5);
    expect(mockCosSim).toHaveBeenCalledTimes(4);
  });
});
