-- Database Update Script 2024
-- This script implements the requested changes to database schema

-- PART 1: Modify the sensor_data table
-- 1. Add the new device_id column
ALTER TABLE sensor_data 
ADD COLUMN device_id TEXT;

-- 2. Set device_id = '1' for all existing rows
UPDATE sensor_data 
SET device_id = '1';

-- 3. Drop the foreign key constraint before dropping the column
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.table_constraints 
        WHERE constraint_name = 'sensor_data_player_id_fkey'
        AND table_name = 'sensor_data'
    ) THEN
        ALTER TABLE sensor_data DROP CONSTRAINT sensor_data_player_id_fkey;
    END IF;
END $$;

-- 4. Drop the old player_id column
ALTER TABLE sensor_data 
DROP COLUMN player_id;

-- PART 2: Modify the sessions table
-- 1. Add session_type column
ALTER TABLE sessions 
ADD COLUMN session_type TEXT;

-- 2. Set session_type = 'solo' for all existing sessions
UPDATE sessions 
SET session_type = 'solo';

-- PART 3: Create new session_players junction table
-- 1. Create the session_players table
CREATE TABLE session_players (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
  player_id UUID REFERENCES player_profiles(id) ON DELETE SET NULL,
  device_id TEXT,
  created_at TIMESTAMPTZ DEFAULT now()
);

-- 2. Add indexes for faster lookups
CREATE INDEX idx_session_players_session_id ON session_players(session_id);
CREATE INDEX idx_session_players_player_id ON session_players(player_id);
CREATE INDEX idx_session_players_device_id ON session_players(device_id);

-- PART 4: Insert default data for existing sessions
-- For each existing session, add a link to the default player with device_id = '1'
INSERT INTO session_players (session_id, player_id, device_id)
SELECT id, 'de6f3ecd-d726-40a2-bbfa-457b517f31f5', '1'
FROM sessions;

-- PART 5: Add validation for session_type
-- 1. Create a function to validate session_type
CREATE OR REPLACE FUNCTION validate_session_type()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.session_type IS NOT NULL AND 
       NEW.session_type NOT IN (
         'pass_catch_calibration', 
         'groundball_calibration', 
         'shot_calibration', 
         'faceoff_calibration', 
         'cradle_calibration', 
         '2v2', 
         'passing_partners', 
         'solo'
       ) THEN
        RAISE EXCEPTION 'Invalid session_type: %', NEW.session_type;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 2. Create a trigger to enforce valid session_type values
DROP TRIGGER IF EXISTS session_type_validation ON sessions;
CREATE TRIGGER session_type_validation
BEFORE INSERT OR UPDATE ON sessions
FOR EACH ROW
EXECUTE FUNCTION validate_session_type();

-- PART 6: Add comments to explain the new schema
COMMENT ON TABLE session_players IS 'Junction table linking players and devices to sessions';
COMMENT ON COLUMN session_players.session_id IS 'Reference to the session';
COMMENT ON COLUMN session_players.player_id IS 'Reference to the player profile';
COMMENT ON COLUMN session_players.device_id IS 'Device ID used in this session by this player';
COMMENT ON COLUMN sessions.session_type IS 'Type of session (solo, 2v2, etc)'; 