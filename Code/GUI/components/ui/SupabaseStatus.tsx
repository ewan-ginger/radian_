'use client';

import { useSupabase } from '@/hooks/useSupabase';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Loader2 } from 'lucide-react';

export function SupabaseStatus() {
  const { isConfigured, isLoading, error } = useSupabase();

  return (
    <Card>
      <CardHeader>
        <CardTitle>Supabase Connection</CardTitle>
        <CardDescription>Database connection status</CardDescription>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="flex items-center gap-2 text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" />
            <span>Checking connection...</span>
          </div>
        ) : error ? (
          <div className="text-destructive">
            <p>{error}</p>
            <p className="text-xs mt-2">
              Make sure you have created a Supabase project and updated the .env.local file with your project credentials.
            </p>
          </div>
        ) : isConfigured ? (
          <div className="text-green-500 dark:text-green-400">
            <p>Connected to Supabase</p>
            <p className="text-xs mt-2 text-muted-foreground">
              Your application is successfully connected to the Supabase backend.
            </p>
          </div>
        ) : (
          <div className="text-amber-500 dark:text-amber-400">
            <p>Supabase not configured</p>
            <p className="text-xs mt-2 text-muted-foreground">
              Please check your environment variables in .env.local file.
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );
} 