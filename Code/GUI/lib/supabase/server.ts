'use server';

import { createClient } from '@supabase/supabase-js';
import { cookies } from 'next/headers';
import { type Database } from '@/types/supabase';

// Create a Supabase client for use in server components and API routes
export async function createServerSupabaseClient() {
  const cookieStore = cookies();
  
  const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL as string;
  const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY as string;
  
  return createClient<Database>(supabaseUrl, supabaseAnonKey, {
    auth: {
      persistSession: false,
      autoRefreshToken: false,
    },
    cookies: {
      get(name: string) {
        return cookieStore.get(name)?.value;
      },
    },
  });
}

// Helper function to get data from Supabase in server components
export async function getServerData<T>(
  query: (client: ReturnType<typeof createServerSupabaseClient>) => Promise<{ data: T; error: any }>
): Promise<T | null> {
  const supabase = await createServerSupabaseClient();
  const { data, error } = await query(supabase);
  
  if (error) {
    console.error('Supabase query error:', error);
    return null;
  }
  
  return data;
} 