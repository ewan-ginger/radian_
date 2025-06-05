-- Complete Schema Update
-- This script implements all required changes to:
-- 1. Remove the unused players table
-- 2. Update the sensor_data table to use device_id instead of player_id

-- Step 1: Add device_id column to sensor_data if it doesn't exist
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'sensor_data' AND column_name = 'device_id'
    ) THEN
        ALTER TABLE sensor_data ADD COLUMN device_id TEXT;
    END IF;
END $$;

-- Step 2: Migrate player_id values to device_id
-- This converts all existing player_id values to device_id
UPDATE sensor_data
SET device_id = player_id::text
WHERE device_id IS NULL AND player_id IS NOT NULL;

-- Step 3: Make device_id NOT NULL after migration
ALTER TABLE sensor_data
ALTER COLUMN device_id SET NOT NULL;

-- Step 4: Remove the player_id column and its constraints
DO $$ 
BEGIN
    -- Check if the foreign key constraint exists
    IF EXISTS (
        SELECT 1 FROM information_schema.table_constraints 
        WHERE constraint_name = 'sensor_data_player_id_fkey'
        AND table_name = 'sensor_data'
    ) THEN
        -- Drop the foreign key constraint
        ALTER TABLE sensor_data DROP CONSTRAINT sensor_data_player_id_fkey;
    END IF;
    
    -- Check if the player_id column exists
    IF EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'sensor_data' AND column_name = 'player_id'
    ) THEN
        -- Drop the player_id column
        ALTER TABLE sensor_data DROP COLUMN player_id;
    END IF;
END $$;

-- Step 5: Check if players table exists and drop it if it does
DO $$ 
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.tables 
        WHERE table_schema = 'public' AND table_name = 'players'
    ) THEN
        -- Drop the players table
        DROP TABLE public.players;
        RAISE NOTICE 'The players table has been dropped.';
    ELSE
        RAISE NOTICE 'The players table does not exist. No action needed.';
    END IF;
END $$;

-- Step 6: Update any functions or triggers that reference player_id
-- Placeholder for custom function updates
-- Add specific function updates here if needed

-- Step 7: Add indices to improve device_id query performance
CREATE INDEX IF NOT EXISTS idx_sensor_data_device_id ON sensor_data(device_id);
CREATE INDEX IF NOT EXISTS idx_sensor_data_session_device ON sensor_data(session_id, device_id);

-- Step 8: Add a comment to the device_id column for documentation
COMMENT ON COLUMN sensor_data.device_id IS 'Raw device identifier from the sensor device, not a foreign key';

-- Confirmation that script completed successfully
DO $$ 
BEGIN
    RAISE NOTICE 'Database schema update completed successfully!';
END $$; 