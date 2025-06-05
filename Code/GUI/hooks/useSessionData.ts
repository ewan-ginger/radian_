'use client';

import { useState, useEffect, useCallback } from 'react';
import { SessionEntity, SessionSummary } from '@/types/database.types';
import * as sessionService from '@/lib/services/session-service';

export function useSessionData() {
  const [sessions, setSessions] = useState<SessionEntity[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const fetchSessions = useCallback(async () => {
    try {
      setIsLoading(true);
      setError(null);
      const data = await sessionService.getAllSessions();
      setSessions(data);
    } catch (err) {
      console.error('Error fetching sessions:', err);
      setError('Failed to fetch sessions. Please try again.');
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchSessions();
  }, [fetchSessions]);

  const createSession = useCallback(async (name?: string) => {
    try {
      setError(null);
      const newSession = await sessionService.createSession({
        name,
        start_time: new Date().toISOString(),
      });
      setSessions(prev => [newSession, ...prev]);
      return newSession;
    } catch (err) {
      console.error('Error creating session:', err);
      setError('Failed to create session. Please try again.');
      throw err;
    }
  }, []);

  const endSession = useCallback(async (id: string) => {
    try {
      setError(null);
      const updatedSession = await sessionService.endSession(id);
      setSessions(prev => prev.map(session => 
        session.id === id ? updatedSession : session
      ));
      return updatedSession;
    } catch (err) {
      console.error('Error ending session:', err);
      setError('Failed to end session. Please try again.');
      throw err;
    }
  }, []);

  const deleteSession = useCallback(async (id: string) => {
    try {
      setError(null);
      await sessionService.deleteSession(id);
      setSessions(prev => prev.filter(session => session.id !== id));
      return true;
    } catch (err) {
      console.error('Error deleting session:', err);
      setError('Failed to delete session. Please try again.');
      throw err;
    }
  }, []);

  const getSessionSummary = useCallback(async (id: string): Promise<SessionSummary | null> => {
    try {
      setError(null);
      return await sessionService.getSessionSummary(id);
    } catch (err) {
      console.error('Error fetching session summary:', err);
      setError('Failed to fetch session summary. Please try again.');
      return null;
    }
  }, []);

  return {
    sessions,
    isLoading,
    error,
    fetchSessions,
    createSession,
    endSession,
    deleteSession,
    getSessionSummary,
  };
} 