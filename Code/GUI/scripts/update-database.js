// Update Database Schema Script
// Run this script to apply the migration SQL to your Supabase database

import { createClient } from '@supabase/supabase-js';
import fs from 'fs';
import path from 'path';
import dotenv from 'dotenv';

// Load environment variables
dotenv.config({ path: '.env.local' });

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

if (!supabaseUrl || !supabaseServiceKey) {
  console.error('Missing Supabase environment variables. Please check your .env.local file.');
  process.exit(1);
}

// Create Supabase client with service role key
const supabase = createClient(supabaseUrl, supabaseServiceKey);

async function updateSchema() {
  try {
    // Read and execute the migration SQL
    const sqlPath = path.join(process.cwd(), 'scripts', 'update-player-schema.sql');
    const sql = fs.readFileSync(sqlPath, 'utf8');
    
    console.log('Running database migration...');
    const { error } = await supabase.rpc('pgexec', { sql });
    
    if (error) {
      console.error('Error updating schema:', error);
      process.exit(1);
    }
    
    console.log('✅ Database schema updated successfully');
  } catch (error) {
    console.error('Unexpected error:', error);
    process.exit(1);
  }
}

updateSchema(); 