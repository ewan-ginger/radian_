/**
 * Database seeding script for Radian Sports Analytics Dashboard
 * 
 * This script initializes the Supabase database with the required tables
 * and inserts minimal seed data for development and testing.
 * 
 * Usage:
 * 1. Make sure your .env.local file contains valid Supabase credentials
 * 2. Run this script with: npx ts-node scripts/seed-database.ts
 */

import { createClient } from '@supabase/supabase-js';
import { completeSchemaSQL } from '../lib/supabase/schema';
import dotenv from 'dotenv';
import path from 'path';

// Load environment variables from .env.local
dotenv.config({ path: path.resolve(process.cwd(), '.env.local') });

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
const supabaseKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

if (!supabaseUrl || !supabaseKey) {
  console.error('Error: Supabase credentials not found in environment variables.');
  console.error('Make sure you have a .env.local file with NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_ANON_KEY.');
  process.exit(1);
}

const supabase = createClient(supabaseUrl, supabaseKey);

async function seedDatabase() {
  console.log('Starting database initialization...');
  
  try {
    // Execute the complete schema SQL
    const { error } = await supabase.rpc('exec_sql', { sql: completeSchemaSQL });
    
    if (error) {
      if (error.message.includes('relation "players" already exists')) {
        console.log('Tables already exist. Skipping schema creation.');
      } else {
        throw error;
      }
    } else {
      console.log('Database schema created successfully.');
    }
    
    // Verify tables were created by querying the players table
    const { data: players, error: playersError } = await supabase
      .from('players')
      .select('*');
    
    if (playersError) {
      throw playersError;
    }
    
    console.log(`Found ${players.length} players in the database.`);
    console.log('Database initialization completed successfully.');
    
  } catch (error) {
    console.error('Error initializing database:', error);
    process.exit(1);
  }
}

// Run the seeding function
seedDatabase(); 