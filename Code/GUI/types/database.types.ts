/**
 * TypeScript types for database entities
 * 
 * These types represent the data structures used in the application
 * and correspond to the database schema defined in lib/supabase/schema.ts
 */

import { Player, Session, SessionPlayer, SensorData } from './supabase';

// Player entity with additional application-specific properties
export interface PlayerEntity extends Player {
  isActive?: boolean;
  lastSeen?: Date;
  stick_type: 'short-stick' | 'long-stick' | 'goalie-stick';
  position: 'attack' | 'midfield' | 'defense' | 'faceoff' | 'goalie';
  strong_hand: 'left' | 'right';
}

// Valid session types
export type SessionType = 
  | 'pass_calibration'
  | 'pass_catch_calibration'
  | 'groundball_calibration'
  | 'shot_calibration'
  | 'faceoff_calibration'
  | 'cradle_calibration'
  | '2v2'
  | 'passing_partners'
  | 'solo';

// Helper function to get required players for a session type
export const getRequiredPlayers = (sessionType?: SessionType, groundballCalibrationPlayers?: number): number => {
  if (!sessionType) return 1; // Default to 1 if type is unknown or not set
  switch (sessionType) {
    case 'pass_calibration':
    case 'cradle_calibration':
    case 'solo':
      return 1;
    case 'groundball_calibration':
      return groundballCalibrationPlayers || 1; // New logic: use provided count or default to 1
    case 'pass_catch_calibration':
    case 'shot_calibration':
    case 'faceoff_calibration':
    case 'passing_partners':
      return 2;
    case '2v2':
      return 5;
    default:
      // Ensure all session types are handled. If a new type is added,
      // this will help catch it during development.
      const _exhaustiveCheck: never = sessionType;
      return 1; // Default fallback
  }
};

// Session entity with additional application-specific properties
export interface SessionEntity {
  id: string;
  name: string | null;
  start_time: string;
  end_time: string | null;
  duration?: unknown; // Keep as unknown, handle in component
  session_type?: SessionType; // Optional to handle null values from database
  created_at: string;
  // Additional application properties
  playerName?: string;
  dataPoints?: number;
  // Linked players and devices
  players?: SessionPlayerEntity[];
}

// Session player entity representing a player-device pair in a session
// Includes optional nested player info as fetched by service, 
// but prefer using the top-level playerName if available.
export interface SessionPlayerEntity extends SessionPlayer {
  playerName?: string; // Convenience property with the player's name
  deviceName?: string; // Convenience property with a human-readable device name
  player?: { // Optional nested player info as returned by some queries
      id: string;
      name: string | null;
  } | null;
}

// Sensor data entity with additional application-specific properties
export interface SensorDataEntity {
  id: string;
  session_id: string;
  device_id: number | null;
  timestamp: number;
  accelerometer_x: number | null;
  accelerometer_y: number | null;
  accelerometer_z: number | null;
  gyroscope_x: number | null;
  gyroscope_y: number | null;
  gyroscope_z: number | null;
  magnetometer_x: number | null;
  magnetometer_y: number | null;
  magnetometer_z: number | null;
  orientation_x: number | null;
  orientation_y: number | null;
  orientation_z: number | null;
  battery_level: number | null;
  created_at: string;
  // Derived properties
  acceleration?: number; // Magnitude of acceleration vector
  rotationRate?: number; // Magnitude of rotation vector
  playerName?: string; // Resolved player name based on device_id
}

// Player statistics derived from sensor data
export interface PlayerStatistics {
  playerId: string;       // Reference to player_profiles.id
  playerName: string;     // Name from player_profiles
  totalSessions: number;
  totalDuration: number;  // in seconds
  averageAcceleration: number;
  maxAcceleration: number;
  averageRotationRate: number;
  maxRotationRate: number;
  lastSessionDate: Date;
}

// Session summary with aggregated data
export interface SessionSummary {
  sessionId: string;
  sessionName: string;
  startTime: Date;
  endTime?: Date;
  duration: number; // in seconds
  sessionType?: SessionType; // Type of session (solo, passing, etc.)
  devices: {
    deviceId: string;
    playerName: string;
    dataPoints: number;
  }[];
  players: string[]; // Array of player names involved in the session
  dataPointsCount: number;
}

// Time series data point for visualization
export interface TimeSeriesDataPoint {
  timestamp: number;
  value: number;
  label?: string;
}

// Orientation data for 3D visualization
export interface OrientationData {
  timestamp: number;
  x: number;
  y: number;
  z: number;
}

// Represents data stored in the training_sensor_data table
export interface TrainingSensorDataEntity {
  id: string; // uuid, primary key
  original_data_id: string | null; // uuid, foreign key to sensor_data
  session_id: string | null; // uuid, foreign key to sessions
  player_id: string | null; // uuid, foreign key to player_profiles
  timestamp: number; // float8 (double precision)
  accelerometer_x: number | null; // real
  accelerometer_y: number | null; // real
  accelerometer_z: number | null; // real
  gyroscope_x: number | null; // real
  gyroscope_y: number | null; // real
  gyroscope_z: number | null; // real
  magnetometer_x: number | null; // real
  magnetometer_y: number | null; // real
  magnetometer_z: number | null; // real
  orientation_x: number | null; // real
  orientation_y: number | null; // real
  orientation_z: number | null; // real
  label: string | null; // text
  metric: number | null; // real
} 