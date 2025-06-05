-- Add missing columns to player_profiles table if they don't exist
DO $$ 
BEGIN
    -- Check if the columns exist before adding them
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'player_profiles' AND column_name = 'stick_type') THEN
        ALTER TABLE player_profiles ADD COLUMN stick_type TEXT DEFAULT 'short-stick' NOT NULL;
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'player_profiles' AND column_name = 'position') THEN
        ALTER TABLE player_profiles ADD COLUMN position TEXT DEFAULT 'midfield' NOT NULL;
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'player_profiles' AND column_name = 'strong_hand') THEN
        ALTER TABLE player_profiles ADD COLUMN strong_hand TEXT DEFAULT 'right' NOT NULL;
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'player_profiles' AND column_name = 'updated_at') THEN
        ALTER TABLE player_profiles ADD COLUMN updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW();
    END IF;
END $$; 