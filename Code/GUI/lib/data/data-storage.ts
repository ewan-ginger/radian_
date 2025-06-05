import { 
  insertSensorDataBatch, 
  getSensorDataBySession, 
  getSensorDataBySessionAndPlayer 
} from '@/lib/services/sensor-data-service';
import { 
  getAllSessions, 
  getSessionById, 
  getSessionsByPlayer 
} from '@/lib/services/session-service';
import { 
  getAllPlayers, 
  getPlayerById 
} from '@/lib/services/player-service';
import { 
  SensorDataInsert, 
  SessionInsert, 
  SessionUpdate 
} from '@/types/supabase';
import { 
  SensorDataEntity, 
  SessionEntity, 
  PlayerEntity, 
  TimeSeriesDataPoint 
} from '@/types/database.types';

/**
 * Save sensor data to Supabase
 * @param data Array of sensor data records
 * @returns True if data was saved successfully
 */
export async function saveSensorData(data: SensorDataInsert[]): Promise<boolean> {
  try {
    await insertSensorDataBatch(data);
    return true;
  } catch (error) {
    console.error('Error saving sensor data:', error);
    return false;
  }
}

/**
 * Get all sessions from Supabase
 * @returns Array of sessions
 */
export async function getSessions(): Promise<SessionEntity[]> {
  try {
    return await getAllSessions();
  } catch (error) {
    console.error('Error getting sessions:', error);
    return [];
  }
}

/**
 * Get a session by ID from Supabase
 * @param sessionId Session ID
 * @returns Session or null if not found
 */
export async function getSession(sessionId: string): Promise<SessionEntity | null> {
  try {
    return await getSessionById(sessionId);
  } catch (error) {
    console.error(`Error getting session ${sessionId}:`, error);
    return null;
  }
}

/**
 * Get all players from Supabase
 * @returns Array of players
 */
export async function getPlayers(): Promise<PlayerEntity[]> {
  try {
    return await getAllPlayers();
  } catch (error) {
    console.error('Error getting players:', error);
    return [];
  }
}

/**
 * Get a player by ID from Supabase
 * @param playerId Player ID
 * @returns Player or null if not found
 */
export async function getPlayer(playerId: string): Promise<PlayerEntity | null> {
  try {
    return await getPlayerById(playerId);
  } catch (error) {
    console.error(`Error getting player ${playerId}:`, error);
    return null;
  }
}

/**
 * Get sensor data for a session from Supabase
 * @param sessionId Session ID
 * @param limit Maximum number of records to return (default: 1000)
 * @param offset Offset for pagination (default: 0)
 * @returns Array of sensor data records
 */
export async function getSessionData(
  sessionId: string,
  limit: number = 1000,
  offset: number = 0
): Promise<SensorDataEntity[]> {
  try {
    return await getSensorDataBySession(sessionId, limit, offset);
  } catch (error) {
    console.error(`Error getting data for session ${sessionId}:`, error);
    return [];
  }
}

/**
 * Get sensor data for a player in a session from Supabase
 * @param sessionId Session ID
 * @param playerId Player ID
 * @param limit Maximum number of records to return (default: 1000)
 * @param offset Offset for pagination (default: 0)
 * @returns Array of sensor data records
 */
export async function getPlayerSessionData(
  sessionId: string,
  playerId: string,
  limit: number = 1000,
  offset: number = 0
): Promise<SensorDataEntity[]> {
  try {
    return await getSensorDataBySessionAndPlayer(sessionId, playerId, limit, offset);
  } catch (error) {
    console.error(`Error getting data for player ${playerId} in session ${sessionId}:`, error);
    return [];
  }
}

/**
 * Convert sensor data to time series format for visualization
 * @param data Array of sensor data records
 * @param dataType Type of data to extract (accelerometer, gyroscope, magnetometer, orientation)
 * @returns Array of time series data points
 */
export function convertToTimeSeriesData(
  data: SensorDataEntity[],
  dataType: 'accelerometer' | 'gyroscope' | 'magnetometer' | 'orientation'
): TimeSeriesDataPoint[] {
  return data.map(record => {
    const timestamp = record.timestamp;
    let x, y, z;
    
    switch (dataType) {
      case 'accelerometer':
        x = record.accelerometer_x;
        y = record.accelerometer_y;
        z = record.accelerometer_z;
        break;
      case 'gyroscope':
        x = record.gyroscope_x;
        y = record.gyroscope_y;
        z = record.gyroscope_z;
        break;
      case 'magnetometer':
        x = record.magnetometer_x;
        y = record.magnetometer_y;
        z = record.magnetometer_z;
        break;
      case 'orientation':
        x = record.orientation_x;
        y = record.orientation_y;
        z = record.orientation_z;
        break;
    }
    
    return {
      timestamp,
      x: x || 0,
      y: y || 0,
      z: z || 0
    };
  });
} 