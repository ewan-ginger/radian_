'use client';

import React, { createContext, useContext, ReactNode } from 'react';
import { useDataRecording } from '@/hooks/useDataRecording';
import { PlayerEntity } from '@/types/database.types';

interface RecordingContextType {
  isRecording: boolean;
  sessionId: string | null;
  playerId: string | null;
  players: PlayerEntity[];
  recordingDuration: number;
  dataPoints: number;
  startRecording: (sessionName?: string, existingSessionId?: string) => Promise<boolean>;
  stopRecording: () => Promise<boolean>;
  setPlayer: (playerId: string) => void;
  addDataPoint: (data: number[]) => Promise<boolean>;
}

const RecordingContext = createContext<RecordingContextType | undefined>(undefined);

interface RecordingProviderProps {
  children: ReactNode;
  bufferSize?: number;
  flushInterval?: number;
  autoSelectPlayer?: boolean;
}

export function RecordingProvider({
  children,
  bufferSize = 50,
  flushInterval = 5000,
  autoSelectPlayer = true
}: RecordingProviderProps) {
  const recording = useDataRecording({
    bufferSize,
    flushInterval,
    autoSelectPlayer
  });
  
  return (
    <RecordingContext.Provider value={recording}>
      {children}
    </RecordingContext.Provider>
  );
}

export function useRecording() {
  const context = useContext(RecordingContext);
  if (context === undefined) {
    throw new Error('useRecording must be used within a RecordingProvider');
  }
  return context;
} 