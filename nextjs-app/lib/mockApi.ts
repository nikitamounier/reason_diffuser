import { GenerationParameters, GenerationStatus, GenerationStep } from './api';

// Generate a unique ID for mock generations
const generateMockId = () => `mock-${Date.now()}`;

// Mock generation storage
const mockGenerations: Record<string, {
  prompt: string;
  parameters: GenerationParameters;
  status: GenerationStatus;
  startTime: number;
  mockInterval: NodeJS.Timeout | null;
}> = {};

// Sample texts to show progressive generation - we'll use shorter texts for better visualization
const sampleTexts = [
  "The artificial intelligence revolution has transformed how we approach problem solving.",
  "The artificial intelligence revolution has transformed how we approach problem solving. Machine learning algorithms can now analyze vast amounts of data.",
  "The artificial intelligence revolution has transformed how we approach problem solving. Machine learning algorithms can now analyze vast amounts of data and extract patterns that humans might miss.",
  "The artificial intelligence revolution has transformed how we approach problem solving. Machine learning algorithms can now analyze vast amounts of data and extract patterns that humans might miss. This capability has led to breakthroughs in fields ranging from medicine to finance.",
];

// Initial masked text (with a pattern to make it more visually interesting)
const createInitialMaskedText = (length: number): string => {
  // Create a mix of characters and masks to make the initial state more visually interesting
  let result = '';
  for (let i = 0; i < length; i++) {
    // Start with just a few real characters visible (e.g., only the first word)
    if (i < 3) {
      result += "The";
    } else {
      result += '█';
    }
  }
  return result;
};

// Create realistic blockmasked text by replacing characters with black squares
const createMaskedText = (text: string, maskPercentage: number): string => {
  const characters = text.split('');
  let result = '';

  for (let i = 0; i < characters.length; i++) {
    // Keep the beginning of the text more stable
    if (i < 10 && characters[i] !== ' ') {
      result += characters[i];
    } else {
      // Mask characters based on position and random chance
      result += Math.random() < maskPercentage ? '█' : characters[i];
    }
  }

  return result;
};

// Generate deterministic progressive text unmasking
const generateProgressiveText = (fullText: string, unmaskPercentage: number): string => {
  const characters = fullText.split('');
  let result = '';

  for (let i = 0; i < characters.length; i++) {
    // Deterministic unmasking based on position and progress
    // This creates a more predictable left-to-right unmasking effect
    const positionThreshold = characters.length * unmaskPercentage;

    // Characters before the threshold are more likely to be unmasked
    if (i < positionThreshold) {
      // First 80% of characters before threshold are fully unmasked
      if (i < positionThreshold * 0.8) {
        result += characters[i];
      }
      // Last 20% of characters before threshold are partially masked
      else {
        result += Math.random() < 0.7 ? characters[i] : '█';
      }
    }
    // Characters after the threshold are mostly masked
    else {
      // Add some randomness to make it look interesting
      result += Math.random() < 0.1 ? characters[i] : '█';
    }
  }

  return result;
};

// Simulate the text diffusion process
const simulateDiffusion = (
  id: string,
  prompt: string,
  parameters: GenerationParameters
) => {
  // Get parameters with defaults
  const steps = parameters.steps || 128;
  const blockLength = parameters.block_length || 32;

  // Use the first text initially, then gradually expand
  let currentTextIndex = 0;

  // Initial status
  mockGenerations[id] = {
    prompt,
    parameters,
    startTime: Date.now(),
    mockInterval: null,
    status: {
      status: 'in_progress',
      steps: []
    }
  };

  // Add initial block_score step showing the first few characters
  const initialText = sampleTexts[0];
  const initialStep: GenerationStep = {
    type: 'block_score',
    timestamp: Date.now(),
    block_text: "The " + "█".repeat(initialText.length - 4),
    score: 0.5
  };

  // Initialize steps array if needed
  if (!mockGenerations[id].status.steps) {
    mockGenerations[id].status.steps = [];
  }

  mockGenerations[id].status.steps.push(initialStep);

  let currentStep = 0;
  let currentBlock = 0;
  let textExpanded = false;

  // Create a mock interval to simulate generation steps
  const interval = setInterval(() => {
    const generation = mockGenerations[id];

    // If the generation was canceled or doesn't exist
    if (!generation || !generation.status || !generation.status.steps) {
      clearInterval(interval);
      return;
    }

    currentStep++;

    // Calculate overall progress
    const progress = Math.min(currentStep / steps, 0.95);

    // Expand to a longer text at certain intervals
    if (progress > 0.25 && currentTextIndex < 1 && !textExpanded) {
      currentTextIndex = 1;
      textExpanded = true;
    } else if (progress > 0.5 && currentTextIndex < 2 && !textExpanded) {
      currentTextIndex = 2;
      textExpanded = true;
    } else if (progress > 0.75 && currentTextIndex < 3 && !textExpanded) {
      currentTextIndex = 3;
      textExpanded = true;
    } else {
      textExpanded = false;
    }

    // Get current target text
    const currentTargetText = sampleTexts[currentTextIndex];

    // Generate text with progressive unmasking
    if (currentStep % 3 === 0) {
      // Generate the new text state with appropriate masking
      const progressiveText = generateProgressiveText(currentTargetText, progress);

      // Add a step to show the progress
      const progressStep: GenerationStep = {
        type: 'block_score',
        timestamp: Date.now(),
        block_text: progressiveText,
        score: 0.5 + progress * 0.5 // Score improves as we progress
      };

      generation.status.steps.push(progressStep);
    }

    // Simulate block boundaries (approximately)
    if (textExpanded || currentStep % Math.floor(steps / 10) === 0) {
      currentBlock++;

      // Occasionally apply backmasking (roughly 1 in 3 blocks)
      if ((textExpanded || Math.random() < 0.3) && currentBlock > 1) {
        // Collect current scores
        const blockScores = generation.status.steps
          .filter(step => step.type === 'block_score')
          .map(step => step.score);

        const backmaskedBlockScores = blockScores.map(score => score || 0.5);

        // Create a backmasking step
        const backmaskedStep: GenerationStep = {
          type: 'backmasking',
          timestamp: Date.now(),
          block_probs: backmaskedBlockScores,
          mask: [[true]], // Simplified mask representation
        };

        generation.status.steps.push(backmaskedStep);

        // After backmasking, create a new text with more masks to show the effect
        // Get the current text state
        const latestStep = generation.status.steps
          .filter(s => s.type === 'block_score')
          .pop();

        const currentText = latestStep?.block_text || '';

        // Apply more masking (focused on the end of the text)
        const maskedText = createMaskedText(currentText, 0.4);

        // Add a new block_score step with the backmasked text
        const postBackmaskStep: GenerationStep = {
          type: 'block_score',
          timestamp: Date.now() + 50,
          block_text: maskedText,
          score: 0.6 + Math.random() * 0.3 // Improved score after backmasking
        };

        generation.status.steps.push(postBackmaskStep);

        // Small delay before adding a follow-up step with improvements
        setTimeout(() => {
          if (mockGenerations[id] && mockGenerations[id].status && mockGenerations[id].status.steps) {
            // Generate an improved version with better scores
            const improvedText = generateProgressiveText(currentTargetText, progress + 0.1);

            const improvedStep: GenerationStep = {
              type: 'block_score',
              timestamp: Date.now() + 200,
              block_text: improvedText,
              score: 0.7 + Math.random() * 0.3 // Even better score
            };

            mockGenerations[id].status.steps.push(improvedStep);
          }
        }, 500);
      }
    }

    // Mark as completed when finished
    if (currentStep >= steps) {
      generation.status.status = 'completed';
      generation.status.text = sampleTexts[sampleTexts.length - 1];
      clearInterval(interval);
      generation.mockInterval = null;
    }

  }, 150); // Update every 150ms for a good balance of speed and visibility

  mockGenerations[id].mockInterval = interval;

  return id;
};

// Mock API implementations
export const mockStartGeneration = async (
  prompt: string,
  parameters: GenerationParameters = {}
) => {
  const id = generateMockId();
  simulateDiffusion(id, prompt, parameters);
  return { id, status: 'started' };
};

export const mockGetGenerationStatus = async (generationId: string): Promise<GenerationStatus> => {
  const generation = mockGenerations[generationId];

  if (!generation) {
    return { status: 'not_found' };
  }

  return { ...generation.status };
};

export const mockGetDefaultParameters = async (): Promise<GenerationParameters> => {
  // Return the default parameters similar to the backend
  return {
    steps: 128,
    gen_length: 512,
    block_length: 32,
    temperature: 0.3,
    cfg_scale: 0.0,
    remasking: "low_confidence",
    backmasking_alpha: 5.0,
    backmasking_intensity: 0.5,
    global_demasking: true,
    backmasking_frequency: 3,
    backmasking_threshold: 0.4
  };
}; 