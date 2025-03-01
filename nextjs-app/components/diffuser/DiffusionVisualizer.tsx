"use client";

import React, { useEffect, useState, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import { GenerationStep } from '@/lib/api';

interface DiffusionVisualizerProps {
  generationSteps: GenerationStep[];
  completed: boolean;
  finalText?: string;
  error?: string;
  totalSteps: number;
}

interface TextVisualization {
  text: string;
  maskedRanges?: {
    start: number;
    end: number;
    tokenText: string;
    originalIndex: number;
  }[];
  timestamp: number;
  isBackmasking: boolean;
}

const DiffusionVisualizer: React.FC<DiffusionVisualizerProps> = ({
  generationSteps,
  completed,
  finalText,
  error,
  totalSteps,
}) => {
  const [progressValue, setProgressValue] = useState<number>(0);
  const [isBackmasking, setIsBackmasking] = useState<boolean>(false);
  const [diffusionVisualization, setDiffusionVisualization] = useState<TextVisualization | null>(null);
  const textRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom of text container when text updates
  useEffect(() => {
    if (textRef.current) {
      textRef.current.scrollTop = textRef.current.scrollHeight;
    }
  }, [diffusionVisualization]);

  // Calculate progress
  useEffect(() => {
    if (completed) {
      setProgressValue(100);
    } else {
      const progress = Math.min(
        (generationSteps.length / totalSteps) * 100,
        99
      );
      setProgressValue(progress);
    }
  }, [generationSteps, completed, totalSteps]);

  // Build the visualization from all blocks combined
  useEffect(() => {
    // Filter steps to only get block_score steps
    const blockScoreSteps = generationSteps.filter(step => step.type === 'block_score');

    if (blockScoreSteps.length === 0) {
      // Handle initial state
      if (generationSteps.length === 0) {
        setDiffusionVisualization({
          text: 'Waiting for diffusion to start...',
          timestamp: Date.now(),
          isBackmasking: false
        });
      } else {
        setDiffusionVisualization({
          text: 'Processing... Initializing diffusion process.',
          timestamp: Date.now(),
          isBackmasking: false
        });
      }
      return;
    }

    // Find all steps of type 'backmasking'
    const backmaskingSteps = generationSteps.filter(step => step.type === 'backmasking');

    // Get the latest backmasking step
    const latestBackmaskStep = backmaskingSteps.length > 0
      ? backmaskingSteps.sort((a, b) => b.timestamp - a.timestamp)[0]
      : null;

    // Combine all blocks into a single text
    let combinedText = '';
    const allBlocks: string[] = [];

    blockScoreSteps.forEach(step => {
      if (step.block_text) {
        allBlocks.push(step.block_text);
      }
    });

    combinedText = allBlocks.join('');

    // If completed with final text, use that instead
    if (completed && finalText) {
      combinedText = finalText;
    }

    // Extract masked token information from the latest backmasking step
    let maskedRanges: { start: number, end: number, tokenText: string, originalIndex: number }[] = [];

    if (latestBackmaskStep && latestBackmaskStep.masked_text_info) {
      // Use the new masked_text_info for precise token highlighting
      maskedRanges = latestBackmaskStep.masked_text_info.map(info => {
        // The position_from_prompt tells us the position of this token within the generated text
        const blockIndex = Math.floor(info.position_from_prompt / (blockScoreSteps.length > 0 ? (combinedText.length / blockScoreSteps.length) : 1));

        // Calculate approximate character position in the combined text
        // This is an approximation since tokens don't map 1:1 to characters
        const approximatePosition = blockIndex * (combinedText.length / blockScoreSteps.length);

        return {
          start: approximatePosition,
          end: approximatePosition + info.token_text.length,
          tokenText: info.token_text,
          originalIndex: info.position_from_prompt
        };
      });

      console.log('Latest backmasking step with text info:', latestBackmaskStep);
      console.log('Masked token count:', maskedRanges.length);
    }

    // Create the visualization
    setDiffusionVisualization({
      text: combinedText,
      maskedRanges: maskedRanges,
      timestamp: Date.now(),
      isBackmasking: !!latestBackmaskStep
    });

    // Check if most recent step is backmasking
    setIsBackmasking(
      generationSteps.length > 0 &&
      generationSteps[generationSteps.length - 1].type === 'backmasking'
    );

  }, [generationSteps, completed, finalText]);

  if (error) {
    return (
      <Card className="w-full">
        <CardHeader>
          <CardTitle className="text-red-500">Error</CardTitle>
        </CardHeader>
        <CardContent>
          <p>{error}</p>
        </CardContent>
      </Card>
    );
  }

  // Render the text with highlighted masked tokens
  const renderTextWithHighlights = (
    text: string,
    maskedRanges?: {
      start: number;
      end: number;
      tokenText: string;
      originalIndex: number;
    }[]
  ) => {
    if (!maskedRanges || maskedRanges.length === 0) {
      return text;
    }

    // Create a searchable index for the tokens in the text
    const textLower = text.toLowerCase();
    const highlightedText: React.ReactNode[] = [];
    let lastIndex = 0;

    // For each token, find and highlight all occurrences in the text
    maskedRanges.forEach((range, idx) => {
      const tokenText = range.tokenText.trim();
      if (!tokenText) return; // Skip empty tokens

      // Find all occurrences of this token in the text
      let searchIndex = 0;
      const tokenLower = tokenText.toLowerCase();

      while (searchIndex < textLower.length) {
        const foundIndex = textLower.indexOf(tokenLower, searchIndex);
        if (foundIndex === -1) break;

        // Add any non-highlighted text before this token
        if (foundIndex > lastIndex) {
          highlightedText.push(
            <span key={`regular-${lastIndex}-${foundIndex}`}>
              {text.substring(lastIndex, foundIndex)}
            </span>
          );
        }

        // Add the highlighted token
        highlightedText.push(
          <span
            key={`masked-${foundIndex}-${idx}`}
            className="bg-red-300 bg-opacity-60 text-black"
          >
            {text.substring(foundIndex, foundIndex + tokenText.length)}
          </span>
        );

        lastIndex = foundIndex + tokenText.length;
        searchIndex = lastIndex;
      }
    });

    // Add any remaining text
    if (lastIndex < text.length) {
      highlightedText.push(
        <span key={`regular-${lastIndex}-end`}>
          {text.substring(lastIndex)}
        </span>
      );
    }

    return highlightedText;
  };

  return (
    <Card className="w-full">
      <CardHeader className={isBackmasking ? "bg-amber-50" : ""}>
        <CardTitle>
          {isBackmasking ? "Backmasking in Progress..." : "Text Diffusion Process"}
        </CardTitle>
        <div className="flex flex-col space-y-1 w-full">
          <div className="flex items-center space-x-2">
            <Progress value={progressValue} className={`w-full ${isBackmasking ? "bg-amber-200" : ""}`} />
            <span className="text-sm text-muted-foreground whitespace-nowrap">
              {completed ? 'Complete' : `${Math.round(progressValue)}%`}
            </span>
          </div>
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>Total Steps: {totalSteps}</span>
            <span>Completed: {generationSteps.filter(s => s.type === 'block_score').length} blocks</span>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="flex justify-between items-center mb-2">
          <div className="flex items-center">
            <span className="bg-red-300 bg-opacity-60 px-2 py-0.5 rounded mr-2"></span>
            <span className="text-muted-foreground text-sm">Red highlights show masked tokens</span>
          </div>
          {diffusionVisualization?.maskedRanges && (
            <div className="text-sm text-amber-600 font-medium">
              {diffusionVisualization.maskedRanges.length > 0 ? (
                <>
                  <span className="mr-1">🔄</span>
                  {diffusionVisualization.maskedRanges.length} tokens being reconsidered
                </>
              ) : (
                <>
                  <span className="mr-1">✓</span>
                  No tokens masked
                </>
              )}
            </div>
          )}
        </div>

        {diffusionVisualization ? (
          <div
            ref={textRef}
            className={`whitespace-pre-wrap font-mono p-4 rounded-md max-h-96 overflow-y-auto ${isBackmasking ? 'border border-amber-300 bg-amber-50' : 'bg-muted'}`}
          >
            {diffusionVisualization.text === 'Waiting for diffusion to start...' ||
              diffusionVisualization.text === 'Processing... Initializing diffusion process.' ? (
              <p className="italic text-sm text-gray-500">{diffusionVisualization.text}</p>
            ) : (
              <div className="font-mono">
                {renderTextWithHighlights(diffusionVisualization.text, diffusionVisualization.maskedRanges)}
              </div>
            )}
          </div>
        ) : (
          <div className="space-y-2">
            <Skeleton className="h-4 w-full" />
            <Skeleton className="h-4 w-full" />
            <Skeleton className="h-4 w-3/4" />
          </div>
        )}

        {generationSteps.length > 0 && (
          <div className="mt-4">
            <h4 className="text-sm font-medium mb-2">Progress Details</h4>
            <div className="text-xs text-muted-foreground">
              <p>Block Scores: {
                blockScoreValues(generationSteps)
              }</p>
              <p>Backmasking Events: {
                generationSteps.filter(step => step.type === 'backmasking').length
              }</p>
              <p>Generation Steps: {generationSteps.length} / ~{totalSteps}</p>
              <p className="text-xs mt-1">
                {isBackmasking ?
                  "🔄 Backmasking: Reconsidering low-confidence parts of the text to improve quality" :
                  "🔄 Diffusing: Gradually unmasking text from noise"}
              </p>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

// Helper function to format block scores
function blockScoreValues(steps: GenerationStep[]): string {
  const blockScores = steps
    .filter(step => step.type === 'block_score' && step.score !== undefined)
    .map(step => step.score?.toFixed(2));

  if (blockScores.length === 0) {
    return 'None yet';
  }

  // Only show the last 5 scores to avoid cluttering
  const recentScores = blockScores.slice(-5);
  return recentScores.join(', ') + (blockScores.length > 5 ? ' ...' : '');
}

export default DiffusionVisualizer; 