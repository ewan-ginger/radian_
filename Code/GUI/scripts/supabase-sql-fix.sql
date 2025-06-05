-- Add missing columns to player_profiles table
-- Execute this in the Supabase SQL Editor

-- Add stick_type column if it doesn't exist
ALTER TABLE player_profiles 
ADD COLUMN IF NOT EXISTS stick_type TEXT DEFAULT 'short-stick' NOT NULL;

-- Add position column if it doesn't exist
ALTER TABLE player_profiles 
ADD COLUMN IF NOT EXISTS position TEXT DEFAULT 'midfield' NOT NULL;

-- Add strong_hand column if it doesn't exist
ALTER TABLE player_profiles 
ADD COLUMN IF NOT EXISTS strong_hand TEXT DEFAULT 'right' NOT NULL;

-- Add updated_at column if it doesn't exist
ALTER TABLE player_profiles 
ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW();

-- Update any existing players to have valid values
UPDATE player_profiles 
SET stick_type = 'short-stick', position = 'midfield', strong_hand = 'right', updated_at = NOW()
WHERE stick_type IS NULL OR position IS NULL OR strong_hand IS NULL; 