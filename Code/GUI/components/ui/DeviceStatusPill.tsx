"use client";

import { Loader2 } from 'lucide-react';

type DeviceStatus = 'disconnected' | 'connecting' | 'connected';

interface DeviceStatusPillProps {
  status: DeviceStatus;
}

export function DeviceStatusPill({ status }: DeviceStatusPillProps) {
  return (
    <div className="flex items-center gap-2">
      {status === 'connecting' ? (
        <div className="flex items-center gap-1 text-xs text-muted-foreground">
          <Loader2 className="h-3 w-3 animate-spin" />
          <span>Connecting...</span>
        </div>
      ) : status === 'disconnected' ? (
        <div className="flex items-center gap-1">
          <div className="h-2 w-2 rounded-full bg-destructive"></div>
          <span className="text-xs text-destructive">Disconnected</span>
        </div>
      ) : (
        <div className="flex items-center gap-1">
          <div className="h-2 w-2 rounded-full bg-green-500"></div>
          <span className="text-xs text-green-500 dark:text-green-400">Connected</span>
        </div>
      )}
    </div>
  );
} 