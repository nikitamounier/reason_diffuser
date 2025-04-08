"use client";

import { useEffect, useState } from "react";
import DiffusionForm from "@/components/diffuser/DiffusionForm";
import DiffusionVisualizer from "@/components/diffuser/DiffusionVisualizer";
import MockModeToggle from "@/components/diffuser/MockModeToggle";
import { startGeneration, getGenerationStatus, getDefaultParameters, GenerationParameters, GenerationStep, GenerationStatus } from "@/lib/api";
import { mockStartGeneration, mockGetGenerationStatus, mockGetDefaultParameters } from "@/lib/mockApi";

export default function Home() {
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [generationId, setGenerationId] = useState<string | null>(null);
  const [generationStatus, setGenerationStatus] = useState<GenerationStatus | null>(null);
  const [useMockApi, setUseMockApi] = useState<boolean>(false);
  const [defaultParameters, setDefaultParameters] = useState<GenerationParameters>({
    steps: 128,
    gen_length: 512,
    block_length: 32,
    temperature: 0.3,
    backmasking_alpha: 5.0,
    backmasking_intensity: 0.5,
    backmasking_frequency: 3,
    backmasking_threshold: 0.4,
  });
  const [pollingInterval, setPollingInterval] = useState<NodeJS.Timeout | null>(null);

  // Save mock mode preference in localStorage
  useEffect(() => {
    // Check if localStorage has a mock mode preference
    const savedMockMode = localStorage.getItem('useMockApi');
    if (savedMockMode) {
      setUseMockApi(savedMockMode === 'true');
    }
  }, []);

  // Update localStorage when mock mode changes
  useEffect(() => {
    localStorage.setItem('useMockApi', String(useMockApi));
  }, [useMockApi]);

  // Fetch default parameters on mount
  useEffect(() => {
    const fetchDefaultParameters = async () => {
      try {
        const apiFunc = useMockApi ? mockGetDefaultParameters : getDefaultParameters;
        const params = await apiFunc();
        setDefaultParameters(params);
      } catch (error) {
        console.error("Failed to fetch default parameters:", error);
      }
    };

    fetchDefaultParameters();
  }, [useMockApi]);

  // Set up polling for generation status
  useEffect(() => {
    if (generationId && isLoading) {
      // Clear any existing interval
      if (pollingInterval) {
        clearInterval(pollingInterval);
      }

      // Set polling interval (more frequent for mock mode)
      const pollingDelay = useMockApi ? 200 : 1000; // 200ms for mock, 1s for real API

      // Set up new polling interval
      const interval = setInterval(async () => {
        try {
          const apiFunc = useMockApi ? mockGetGenerationStatus : getGenerationStatus;
          const status = await apiFunc(generationId);
          setGenerationStatus(status);

          // Stop polling if generation is complete or failed
          if (status.status === 'completed' || status.status === 'error') {
            setIsLoading(false);
            clearInterval(interval);
            setPollingInterval(null);
          }
        } catch (error) {
          console.error("Error polling generation status:", error);
          clearInterval(interval);
          setPollingInterval(null);
          setIsLoading(false);
        }
      }, pollingDelay);

      setPollingInterval(interval);

      // Clean up on unmount
      return () => {
        clearInterval(interval);
      };
    }
  }, [generationId, isLoading, useMockApi]);

  const handleFormSubmit = async (prompt: string, parameters: GenerationParameters) => {
    try {
      setIsLoading(true);
      setGenerationStatus(null);

      const apiFunc = useMockApi ? mockStartGeneration : startGeneration;
      const response = await apiFunc(prompt, parameters);
      setGenerationId(response.id);
    } catch (error) {
      console.error("Error starting generation:", error);
      setIsLoading(false);
    }
  };

  const handleToggleMockMode = (value: boolean) => {
    // Reset any ongoing generation when switching modes
    if (isLoading) {
      setIsLoading(false);
      setGenerationStatus(null);
      if (pollingInterval) {
        clearInterval(pollingInterval);
        setPollingInterval(null);
      }
    }
    setUseMockApi(value);
  };

  return (
    <div className="flex flex-col min-h-screen">
      <header className="py-6 px-8 border-b bg-background">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold">Text Diffusion with Backmasking</h1>
            <p className="text-muted-foreground">
              Visualize how text diffusion models gradually transform noise into coherent text
            </p>
          </div>
          <MockModeToggle
            useMockApi={useMockApi}
            onToggle={handleToggleMockMode}
          />
        </div>
      </header>

      <main className="flex-1 container mx-auto py-8 px-4 flex flex-col gap-8 max-w-4xl">
        <DiffusionForm
          onSubmit={handleFormSubmit}
          isLoading={isLoading}
          defaultParameters={defaultParameters}
        />

        {(isLoading || generationStatus) && (
          <DiffusionVisualizer
            generationSteps={generationStatus?.steps || []}
            completed={generationStatus?.status === 'completed'}
            finalText={generationStatus?.text}
            error={generationStatus?.error}
            totalSteps={defaultParameters.steps || 100}
          />
        )}
      </main>

      <footer className="py-6 px-8 border-t">
        <div className="container mx-auto text-center text-sm text-muted-foreground">
          <p>Text Diffusion Visualizer - Demonstrating text diffusion and backmasking techniques</p>
          {useMockApi && (
            <p className="text-green-500 text-xs mt-1">
              Running in mock mode - no backend required
            </p>
          )}
        </div>
      </footer>
    </div>
  );
}
