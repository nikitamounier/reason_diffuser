"use client";

import React from 'react';
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";

interface MockModeToggleProps {
  useMockApi: boolean;
  onToggle: (value: boolean) => void;
}

const MockModeToggle: React.FC<MockModeToggleProps> = ({
  useMockApi,
  onToggle,
}) => {
  return (
    <div className="flex items-center space-x-2 bg-muted/40 p-2 rounded-md">
      <Switch
        id="mock-mode"
        checked={useMockApi}
        onCheckedChange={onToggle}
      />
      <Label htmlFor="mock-mode" className="text-sm cursor-pointer">
        Mock Mode {useMockApi ? 'Enabled' : 'Disabled'}
      </Label>

      <div className="ml-1 text-xs text-muted-foreground">
        {useMockApi ? (
          <span className="text-green-500">Using simulated data (no backend required)</span>
        ) : (
          <span>Using real backend API</span>
        )}
      </div>
    </div>
  );
};

export default MockModeToggle; 