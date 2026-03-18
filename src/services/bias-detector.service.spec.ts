import { BiasDetectorService } from './bias-detector.service';

describe('BiasDetectorService', () => {
  let service: BiasDetectorService;

  beforeEach(() => {
    service = new BiasDetectorService();
  });

  it('should detect confirmation bias if explicit bias is provided', () => {
    const result = service.detect('thought', 1, 10, 0.8, [], 'confirmation');
    expect(result).toEqual({
      type: 'confirmation_bias',
      suggestion: 'Consider alternative viewpoints or counterarguments. What evidence would disprove your conclusion?',
    });
  });

  it('should detect confirmation bias due to high similarity with recent history', () => {
    const history = [
      'The data indicates strongly that X is the absolute correct approach to take here',
      'This absolutely proves X is the right correct approach to take here',
      'Yes, X is the definitely correct right approach to take here'
    ];
    const result = service.detect('X is the absolutely correct right approach to take here', 4, 10, undefined, history);
    expect(result?.type).toBe('confirmation_bias');
  });

  it('should detect anchoring bias', () => {
    const history = ['My first thought is X is the only absolute way'];
    const result = service.detect('Initially I thought X is the only absolute way and I will stick to it', 6, 10, undefined, history);
    expect(result?.type).toBe('anchoring_bias');
  });

  it('should detect overconfidence bias', () => {
    const result = service.detect('I am absolutely sure', 2, 10, 0.95, []);
    expect(result?.type).toBe('overconfidence_bias');
  });

  it('should detect availability heuristic', () => {
    const result = service.detect('I just saw the latest news about this recent event', 11, 20, undefined, []);
    expect(result?.type).toBe('availability_heuristic');
  });

  it('should detect sunk cost fallacy', () => {
    const result = service.detect('We have already invested a lot, let us continue and proceed', 8, 10, undefined, []);
    expect(result?.type).toBe('sunk_cost_fallacy');
  });

  it('should return null when no bias is detected', () => {
    const result = service.detect('This is a balanced thought considering multiple options.', 5, 10, 0.5, ['previous thought']);
    expect(result).toBeNull();
  });

  it('should return null when explicit bias type is unrecognized', () => {
    const result = service.detect('thought', 1, 10, 0.5, [], 'unrecognized_bias_type_123');
    expect(result).toBeNull();
  });
});
