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

const DiffusionVisualizer: React.FC<DiffusionVisualizerProps> = ({
  generationSteps,
  completed,
  finalText,
  error,
  totalSteps,
}) => {
  const [currentText, setCurrentText] = useState<string>('');
  const [previousText, setPreviousText] = useState<string>('');
  const [progressValue, setProgressValue] = useState<number>(0);
  const [isBackmasking, setIsBackmasking] = useState<boolean>(false);
  const textRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom of text container when text updates
  useEffect(() => {
    if (textRef.current) {
      textRef.current.scrollTop = textRef.current.scrollHeight;
    }
  }, [currentText]);

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

  // Update displayed text based on the latest steps
  useEffect(() => {
    if (completed && finalText) {
      setCurrentText(finalText);
      setIsBackmasking(false);
      return;
    }

    // Find all block_score steps, which contain the text
    const blockScoreSteps = generationSteps.filter(step => step.type === 'block_score');

    if (blockScoreSteps.length > 0) {
      // Extract the latest block text
      const latestBlockStep = blockScoreSteps[blockScoreSteps.length - 1];

      // Save the previous text for comparison
      setPreviousText(currentText);

      // Check if there was a backmasking step after the latest block_score
      const latestBlockTime = latestBlockStep.timestamp;
      const backmaskedSteps = generationSteps.filter(
        step => step.type === 'backmasking' && step.timestamp > latestBlockTime - 1000 // Give some leeway
      );

      let displayText = latestBlockStep.block_text || '';

      // If there are recent backmasking steps, apply the effect to visualize masked tokens
      if (backmaskedSteps.length > 0) {
        const latestBackmask = backmaskedSteps[backmaskedSteps.length - 1];
        setIsBackmasking(true);

        // For visualization, we'll highlight the backmasking process with colored text
        if (latestBackmask.block_probs && latestBackmask.block_probs.length > 0) {
          // Apply a visual effect to show backmasking in progress
          setCurrentText(
            `${displayText}\n\n[Backmasking in progress...]\nImproving generation quality by reconsidering parts of the text.`
          );
          return;
        }
      } else {
        setIsBackmasking(false);
      }

      // Display the latest text state
      setCurrentText(displayText);
    } else if (generationSteps.length === 0) {
      setCurrentText('Waiting for diffusion to start...');
      setIsBackmasking(false);
    } else {
      // If there are steps but no block_score steps, show a processing message
      setCurrentText('Processing... Initializing diffusion process.');
      setIsBackmasking(false);
    }

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

  // Highlight differences between previous and current text
  const highlightChanges = (prev: string, current: string): React.ReactElement => {
    if (!prev || prev === 'Waiting for diffusion to start...' ||
      prev === 'Processing... Initializing diffusion process.') {
      return <>{current}</>;
    }

    // For simplicity, just highlight characters that were previously masked
    const result: React.ReactElement[] = [];

    for (let i = 0; i < current.length; i++) {
      const prevChar = i < prev.length ? prev[i] : '█';
      const currChar = current[i];

      if (prevChar === '█' && currChar !== '█') {
        // New character revealed (was masked, now revealed)
        result.push(<span key={i} className="text-green-600 font-bold">{currChar}</span>);
      } else {
        result.push(<span key={i}>{currChar}</span>);
      }
    }

    return <>{result}</>;
  };

  return (
    <Card className="w-full">
      <CardHeader className={isBackmasking ? "bg-amber-50" : ""}>
        <CardTitle>
          {isBackmasking ? "Backmasking in Progress..." : "Text Diffusion Process"}
        </CardTitle>
        <div className="flex items-center space-x-2">
          <Progress value={progressValue} className={`w-full ${isBackmasking ? "bg-amber-200" : ""}`} />
          <span className="text-sm text-muted-foreground whitespace-nowrap">
            {completed ? 'Complete' : `${Math.round(progressValue)}%`}
          </span>
        </div>
      </CardHeader>
      <CardContent>
        {currentText ? (
          <div
            ref={textRef}
            className="whitespace-pre-wrap font-mono p-4 bg-muted rounded-md max-h-96 overflow-y-auto"
          >
            {previousText && !currentText.includes("[Backmasking in progress...]") ?
              highlightChanges(previousText, currentText) :
              currentText}
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