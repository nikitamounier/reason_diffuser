"use client";

import React, { useState } from 'react';
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { GenerationParameters } from '@/lib/api';

interface DiffusionFormProps {
  onSubmit: (prompt: string, parameters: GenerationParameters) => void;
  isLoading: boolean;
  defaultParameters: GenerationParameters;
}

const DiffusionForm: React.FC<DiffusionFormProps> = ({
  onSubmit,
  isLoading,
  defaultParameters,
}) => {
  const [prompt, setPrompt] = useState<string>('');
  const [showAdvanced, setShowAdvanced] = useState<boolean>(false);
  const [parameters, setParameters] = useState<GenerationParameters>(defaultParameters);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (prompt.trim()) {
      onSubmit(prompt, parameters);
    }
  };

  const handleParameterChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value, type } = e.target;

    // Convert the value based on its type
    const parsedValue = type === 'number'
      ? parseFloat(value)
      : type === 'checkbox'
        ? (e.target as HTMLInputElement).checked
        : value;

    setParameters({
      ...parameters,
      [name]: parsedValue,
    });
  };

  return (
    <Card className="w-full">
      <form onSubmit={handleSubmit}>
        <CardHeader>
          <CardTitle>Text Diffusion</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <label htmlFor="prompt" className="text-sm font-medium">
              Enter your prompt
            </label>
            <Textarea
              id="prompt"
              placeholder="Ask a question or provide a prompt..."
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              className="min-h-32"
              required
            />
          </div>

          <Button
            type="button"
            variant="outline"
            className="w-full"
            onClick={() => setShowAdvanced(!showAdvanced)}
          >
            {showAdvanced ? "Hide" : "Show"} Advanced Parameters
          </Button>

          {showAdvanced && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 p-4 border rounded-md">
              <div className="space-y-2">
                <label htmlFor="steps" className="text-sm font-medium">
                  Steps
                </label>
                <Input
                  id="steps"
                  name="steps"
                  type="number"
                  value={parameters.steps}
                  onChange={handleParameterChange}
                  min={1}
                />
              </div>

              <div className="space-y-2">
                <label htmlFor="gen_length" className="text-sm font-medium">
                  Generation Length
                </label>
                <Input
                  id="gen_length"
                  name="gen_length"
                  type="number"
                  value={parameters.gen_length}
                  onChange={handleParameterChange}
                  min={1}
                />
              </div>

              <div className="space-y-2">
                <label htmlFor="block_length" className="text-sm font-medium">
                  Block Length
                </label>
                <Input
                  id="block_length"
                  name="block_length"
                  type="number"
                  value={parameters.block_length}
                  onChange={handleParameterChange}
                  min={1}
                />
              </div>

              <div className="space-y-2">
                <label htmlFor="temperature" className="text-sm font-medium">
                  Temperature
                </label>
                <Input
                  id="temperature"
                  name="temperature"
                  type="number"
                  value={parameters.temperature}
                  onChange={handleParameterChange}
                  step="0.1"
                  min="0"
                />
              </div>

              <div className="space-y-2">
                <label htmlFor="backmasking_alpha" className="text-sm font-medium">
                  Backmasking Alpha
                </label>
                <Input
                  id="backmasking_alpha"
                  name="backmasking_alpha"
                  type="number"
                  value={parameters.backmasking_alpha}
                  onChange={handleParameterChange}
                  step="0.1"
                  min="0"
                />
              </div>

              <div className="space-y-2">
                <label htmlFor="backmasking_intensity" className="text-sm font-medium">
                  Backmasking Intensity
                </label>
                <Input
                  id="backmasking_intensity"
                  name="backmasking_intensity"
                  type="number"
                  value={parameters.backmasking_intensity}
                  onChange={handleParameterChange}
                  step="0.1"
                  min="0"
                  max="1"
                />
              </div>

              <div className="space-y-2">
                <label htmlFor="backmasking_frequency" className="text-sm font-medium">
                  Backmasking Frequency
                </label>
                <Input
                  id="backmasking_frequency"
                  name="backmasking_frequency"
                  type="number"
                  value={parameters.backmasking_frequency}
                  onChange={handleParameterChange}
                  min="1"
                />
              </div>

              <div className="space-y-2">
                <label htmlFor="backmasking_threshold" className="text-sm font-medium">
                  Backmasking Threshold
                </label>
                <Input
                  id="backmasking_threshold"
                  name="backmasking_threshold"
                  type="number"
                  value={parameters.backmasking_threshold}
                  onChange={handleParameterChange}
                  step="0.01"
                  min="0"
                  max="1"
                />
              </div>
            </div>
          )}
        </CardContent>
        <CardFooter>
          <Button
            type="submit"
            className="w-full"
            disabled={isLoading || !prompt.trim()}
          >
            {isLoading ? "Generating..." : "Start Text Diffusion"}
          </Button>
        </CardFooter>
      </form>
    </Card>
  );
};

export default DiffusionForm; 