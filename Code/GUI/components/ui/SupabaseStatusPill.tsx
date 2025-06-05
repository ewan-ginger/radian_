"use client";

import { useSupabase } from '@/hooks/useSupabase';
import { Loader2 } from 'lucide-react';

export function SupabaseStatusPill() {
  const { isConfigured, isLoading, error } = useSupabase();

  return (
    <div className="flex items-center gap-2">
      {isLoading ? (
        <div className="flex items-center gap-1 text-xs text-muted-foreground">
          <Loader2 className="h-3 w-3 animate-spin" />
          <span>Connecting...</span>
        </div>
      ) : error ? (
        <div className="flex items-center gap-1">
          <div className="h-2 w-2 rounded-full bg-destructive"></div>
          <span className="text-xs text-destructive">Disconnected</span>
        </div>
      ) : isConfigured ? (
        <div className="flex items-center gap-1">
          <div className="h-2 w-2 rounded-full bg-green-500"></div>
          <span className="text-xs text-green-500 dark:text-green-400">Connected</span>
        </div>
      ) : (
        <div className="flex items-center gap-1">
          <div className="h-2 w-2 rounded-full bg-amber-500"></div>
          <span className="text-xs text-amber-500 dark:text-amber-400">Not Configured</span>
        </div>
      )}
    </div>
  );
} 