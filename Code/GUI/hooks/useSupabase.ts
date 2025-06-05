'use client';

import { useState, useEffect } from 'react';
import { supabaseClient, isSupabaseConfigured } from '@/lib/supabase/client';

export function useSupabase() {
  const [isConfigured, setIsConfigured] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const checkConfiguration = async () => {
      try {
        setIsLoading(true);
        const configured = isSupabaseConfigured();
        setIsConfigured(configured);
        
        if (configured) {
          // Test the connection by fetching a simple query
          const { error } = await supabaseClient.from('players').select('count', { count: 'exact', head: true });
          
          if (error) {
            console.error('Supabase connection error:', error);
            setError('Failed to connect to Supabase. Please check your configuration.');
          } else {
            setError(null);
          }
        } else {
          setError('Supabase is not configured. Please check your environment variables.');
        }
      } catch (err) {
        console.error('Supabase hook error:', err);
        setError('An unexpected error occurred while connecting to Supabase.');
      } finally {
        setIsLoading(false);
      }
    };

    checkConfiguration();
  }, []);

  return {
    supabase: supabaseClient,
    isConfigured,
    isLoading,
    error,
  };
} 