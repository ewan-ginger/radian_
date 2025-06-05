/**
 * Database schema definitions for the Radian Sports Analytics Dashboard
 * 
 * This file contains the SQL schema for the Supabase database tables:
 * - player_profiles: Stores information about players
 * - sessions: Stores information about recording sessions
 * - sensor_data: Stores sensor readings from ESP32 devices
 * - session_players: Links players and devices to sessions
 */

export const PLAYERS_TABLE = 'player_profiles';
export const SESSIONS_TABLE = 'sessions';
export const SENSOR_DATA_TABLE = 'sensor_data';
export const SESSION_PLAYERS_TABLE = 'session_players';

export const createPlayersTableSQL = `
CREATE TABLE ${PLAYERS_TABLE} (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  name TEXT NOT NULL,
  device_id TEXT,
  stick_type TEXT NOT NULL,
  position TEXT NOT NULL,
  strong_hand TEXT NOT NULL,
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
`;

export const createSessionsTableSQL = `
CREATE TABLE ${SESSIONS_TABLE} (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  name TEXT,
  start_time TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  end_time TIMESTAMP WITH TIME ZONE,
  duration INTERVAL DEFAULT NULL,
  session_type TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
`;

export const createSensorDataTableSQL = `
CREATE TABLE ${SENSOR_DATA_TABLE} (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  session_id UUID REFERENCES ${SESSIONS_TABLE}(id),
  device_id TEXT,
  timestamp float8 NOT NULL,
  accelerometer_x REAL,
  accelerometer_y REAL,
  accelerometer_z REAL,
  gyroscope_x REAL,
  gyroscope_y REAL,
  gyroscope_z REAL,
  magnetometer_x REAL,
  magnetometer_y REAL,
  magnetometer_z REAL,
  orientation_x REAL,
  orientation_y REAL,
  orientation_z REAL,
  battery_level REAL,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Add index for faster session lookups
CREATE INDEX idx_sensor_data_session_id ON ${SENSOR_DATA_TABLE}(session_id);
`;

export const createSessionPlayersTableSQL = `
CREATE TABLE ${SESSION_PLAYERS_TABLE} (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id UUID REFERENCES ${SESSIONS_TABLE}(id) ON DELETE CASCADE,
  player_id UUID REFERENCES ${PLAYERS_TABLE}(id) ON DELETE SET NULL,
  device_id TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

CREATE INDEX idx_session_players_session_id ON ${SESSION_PLAYERS_TABLE}(session_id);
CREATE INDEX idx_session_players_player_id ON ${SESSION_PLAYERS_TABLE}(player_id);
CREATE INDEX idx_session_players_device_id ON ${SESSION_PLAYERS_TABLE}(device_id);
`;

export const insertDefaultPlayerSQL = `
INSERT INTO ${PLAYERS_TABLE} (name, device_id, stick_type, position, strong_hand) VALUES ('Player 1', '1', 'short-stick', 'midfield', 'right');
`;

// Complete SQL script to create all tables
export const completeSchemaSQL = `
-- Create players table
${createPlayersTableSQL}

-- Create sessions table
${createSessionsTableSQL}

-- Create sensor_data table
${createSensorDataTableSQL}

-- Create session_players table
${createSessionPlayersTableSQL}

-- Insert default player
${insertDefaultPlayerSQL}
`; 