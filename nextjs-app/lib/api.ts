import axios from 'axios';

const API_BASE_URL = 'https://p01--api--wnjb7knzxxx6.code.run/api';

export interface GenerationParameters {
  steps?: number;
  gen_length?: number;
  block_length?: number;
  temperature?: number;
  cfg_scale?: number;
  remasking?: string;
  backmasking_alpha?: number;
  backmasking_intensity?: number;
  global_demasking?: boolean;
  backmasking_frequency?: number;
  backmasking_threshold?: number;
}

export interface GenerationStep {
  type: string;
  timestamp: number;
  masked_indices?: number[];
  masked_text_info?: {
    token_idx: number;
    token_text: string;
    position_from_prompt: number;
  }[];
  block_text?: string;
  score?: number;
  [key: string]: any;
}

export interface GenerationStatus {
  status: 'started' | 'in_progress' | 'completed' | 'error' | 'not_found';
  text?: string;
  steps?: GenerationStep[];
  error?: string;
}

// Start a new generation with a prompt
export const startGeneration = async (
  prompt: string,
  parameters: GenerationParameters = {}
) => {
  try {
    const response = await axios.post(`${API_BASE_URL}/generate`, {
      prompt,
      parameters,
    });
    return response.data;
  } catch (error) {
    console.error('Error starting generation:', error);
    throw error;
  }
};

// Get the status and progress of a generation
export const getGenerationStatus = async (generationId: string): Promise<GenerationStatus> => {
  try {
    const response = await axios.get(`${API_BASE_URL}/status/${generationId}`);
    return response.data;
  } catch (error) {
    console.error('Error fetching generation status:', error);
    throw error;
  }
};

// Get the default parameters
export const getDefaultParameters = async (): Promise<GenerationParameters> => {
  try {
    const response = await axios.get(`${API_BASE_URL}/parameters`);
    return response.data;
  } catch (error) {
    console.error('Error fetching default parameters:', error);
    throw error;
  }
}; 