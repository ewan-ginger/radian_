import { supabaseClient } from '@/lib/supabase/client';
import { createServerSupabaseClient } from '@/lib/supabase/server';
import { SENSOR_DATA_TABLE } from '@/lib/supabase/schema';
import { SensorData, SensorDataInsert } from '@/types/supabase';
import { SensorDataEntity, TimeSeriesDataPoint, OrientationData } from '@/types/database.types';

/**
 * Get sensor data for a session
 * @param sessionId Session ID
 * @returns Array of sensor data records
 */
export async function getSensorDataBySession(
  sessionId: string
): Promise<SensorDataEntity[]> {
  const PAGE_SIZE = 1000; // Default Supabase limit
  let allData: SensorDataEntity[] = [];
  let offset = 0;
  let keepFetching = true;

  console.log(`Fetching all sensor data for session ${sessionId} with pagination...`);

  while (keepFetching) {
    try {
      console.log(`Fetching page with offset: ${offset}, limit: ${PAGE_SIZE}`);
      const { data, error } = await supabaseClient
        .from(SENSOR_DATA_TABLE)
        .select('*')
        .eq('session_id', sessionId)
        .order('timestamp', { ascending: true })
        .range(offset, offset + PAGE_SIZE - 1);

      if (error) {
        console.error(`Error fetching page for session ${sessionId} (offset: ${offset}):`, error);
        throw new Error(`Failed to fetch sensor data page: ${error.message}`);
      }

      if (data && data.length > 0) {
        console.log(`Fetched ${data.length} records for this page.`);
        allData = allData.concat(data);
        offset += data.length;
        // If we received fewer than PAGE_SIZE records, this was the last page
        if (data.length < PAGE_SIZE) {
          keepFetching = false;
        }
      } else {
        // No more data found
        console.log('No more data found, stopping fetch loop.');
        keepFetching = false;
      }
    } catch (error) {
      // Stop fetching if an error occurs during pagination
      console.error('Stopping fetch loop due to error.');
      keepFetching = false;
      throw error; // Re-throw the error after logging
    }
  }

  console.log(`Finished fetching. Total records for session ${sessionId}: ${allData.length}`);
  return allData;
}

/**
 * Get sensor data for a specific device in a session
 * @param sessionId Session ID
 * @param deviceId Device ID (raw identifier from the sensor device, not a foreign key)
 * @param limit Maximum number of records to return (default: 1000)
 * @param offset Offset for pagination (default: 0)
 * @returns Array of sensor data records
 */
export async function getSensorDataBySessionAndDevice(
  sessionId: string,
  deviceId: string,
  limit: number = 1000,
  offset: number = 0
): Promise<SensorDataEntity[]> {
  const { data, error } = await supabaseClient
    .from(SENSOR_DATA_TABLE)
    .select('*')
    .eq('session_id', sessionId)
    .eq('device_id', deviceId)
    .order('timestamp', { ascending: true })
    .range(offset, offset + limit - 1);

  if (error) {
    console.error(`Error fetching sensor data for session ${sessionId} and device ${deviceId}:`, error);
    throw new Error(`Failed to fetch sensor data: ${error.message}`);
  }

  return data || [];
}

/**
 * Insert a single sensor data record
 * @param sensorData Sensor data to insert
 * @returns The inserted sensor data record
 */
export async function insertSensorData(sensorData: SensorDataInsert): Promise<SensorData> {
  // Add logging to help debug device ID issues
  console.log(`Inserting sensor data with device_id: ${sensorData.device_id}, timestamp: ${sensorData.timestamp}`);
  
  const { data, error } = await supabaseClient
    .from(SENSOR_DATA_TABLE)
    .insert(sensorData)
    .select()
    .single();

  if (error) {
    console.error('Error inserting sensor data:', error);
    console.error('Attempted to insert:', JSON.stringify(sensorData));
    throw new Error(`Failed to insert sensor data: ${error.message}`);
  }

  return data;
}

/**
 * Insert multiple sensor data records in a batch
 * @param sensorDataBatch Array of sensor data records to insert
 * @returns True if successful
 */
export async function insertSensorDataBatch(sensorDataBatch: SensorDataInsert[]): Promise<boolean> {
  if (sensorDataBatch.length === 0) {
    return true;
  }

  // Add logging to help debug device ID issues
  console.log(`Inserting batch of ${sensorDataBatch.length} sensor data records`);
  console.log(`First record device_id: ${sensorDataBatch[0].device_id}, timestamp: ${sensorDataBatch[0].timestamp}`);
  
  const { error } = await supabaseClient
    .from(SENSOR_DATA_TABLE)
    .insert(sensorDataBatch);

  if (error) {
    console.error('Error inserting sensor data batch:', error);
    console.error(`Attempted to insert ${sensorDataBatch.length} records`);
    // Log the first few records to help with debugging
    if (sensorDataBatch.length > 0) {
      console.error('First record:', JSON.stringify(sensorDataBatch[0]));
    }
    throw new Error(`Failed to insert sensor data batch: ${error.message}`);
  }

  return true;
}

/**
 * Get the latest sensor data for a device
 * @param deviceId Device ID (raw identifier from the sensor device, not a foreign key)
 * @returns The latest sensor data record or null if not found
 */
export async function getLatestSensorDataForDevice(deviceId: string): Promise<SensorDataEntity | null> {
  const { data, error } = await supabaseClient
    .from(SENSOR_DATA_TABLE)
    .select('*')
    .eq('device_id', deviceId)
    .order('timestamp', { ascending: false })
    .limit(1)
    .single();

  if (error) {
    if (error.code === 'PGRST116') {
      // No data found
      return null;
    }
    console.error(`Error fetching latest sensor data for device ${deviceId}:`, error);
    throw new Error(`Failed to fetch latest sensor data: ${error.message}`);
  }

  return data;
}

/**
 * Get time series data for a specific sensor value
 * @param sessionId Session ID
 * @param deviceId Device ID (raw identifier from the sensor device, not a foreign key)
 * @param sensorType Type of sensor data to extract (e.g., 'accelerometer_x')
 * @param limit Maximum number of points to return (default: 100)
 * @returns Array of time series data points
 */
export async function getTimeSeriesData(
  sessionId: string,
  deviceId: string,
  sensorType: keyof SensorData,
  limit: number = 100
): Promise<TimeSeriesDataPoint[]> {
  const { data, error } = await supabaseClient
    .from(SENSOR_DATA_TABLE)
    .select(`timestamp, ${sensorType}`)
    .eq('session_id', sessionId)
    .eq('device_id', deviceId)
    .order('timestamp', { ascending: true })
    .limit(limit);

  if (error) {
    console.error(`Error fetching time series data for session ${sessionId} and device ${deviceId}:`, error);
    throw new Error(`Failed to fetch time series data: ${error.message}`);
  }

  return data.map(item => {
    // Use type assertion to safely access the property
    const value = (item as any)[sensorType] || 0;
    return {
      timestamp: item.timestamp,
      value: value as number
    };
  });
}

/**
 * Get orientation data for 3D visualization
 * @param sessionId Session ID
 * @param deviceId Device ID (raw identifier from the sensor device, not a foreign key)
 * @param limit Maximum number of points to return (default: 100)
 * @returns Array of orientation data points
 */
export async function getOrientationData(
  sessionId: string,
  deviceId: string,
  limit: number = 100
): Promise<OrientationData[]> {
  const { data, error } = await supabaseClient
    .from(SENSOR_DATA_TABLE)
    .select('timestamp, orientation_x, orientation_y, orientation_z')
    .eq('session_id', sessionId)
    .eq('device_id', deviceId)
    .order('timestamp', { ascending: true })
    .limit(limit);

  if (error) {
    console.error(`Error fetching orientation data for session ${sessionId} and device ${deviceId}:`, error);
    throw new Error(`Failed to fetch orientation data: ${error.message}`);
  }

  return data.map(item => ({
    timestamp: item.timestamp,
    x: item.orientation_x || 0,
    y: item.orientation_y || 0,
    z: item.orientation_z || 0,
  }));
}

/**
 * Delete sensor data for a session
 * @param sessionId Session ID
 * @returns True if successful
 */
export async function deleteSensorDataBySession(sessionId: string): Promise<boolean> {
  const { error } = await supabaseClient
    .from(SENSOR_DATA_TABLE)
    .delete()
    .eq('session_id', sessionId);

  if (error) {
    console.error(`Error deleting sensor data for session ${sessionId}:`, error);
    throw new Error(`Failed to delete sensor data: ${error.message}`);
  }

  return true;
}

/**
 * Calculate the magnitude of a 3D vector
 * @param x X component
 * @param y Y component
 * @param z Z component
 * @returns Magnitude
 */
export function calculateMagnitude(x: number, y: number, z: number): number {
  return Math.sqrt(x * x + y * y + z * z);
}

/**
 * Process raw sensor data to calculate derived properties
 * Note: device_id in SensorData is a raw identifier from the sensor device itself,
 * not a foreign key to any other table.
 * @param data Raw sensor data
 * @returns Processed sensor data with derived properties
 */
export function processSensorData(data: SensorData): SensorDataEntity {
  const processed = { ...data } as SensorDataEntity;
  
  // Calculate acceleration magnitude if all components are present
  if (data.accelerometer_x !== null && data.accelerometer_y !== null && data.accelerometer_z !== null) {
    processed.acceleration = calculateMagnitude(
      data.accelerometer_x,
      data.accelerometer_y,
      data.accelerometer_z
    );
  }
  
  // Calculate rotation rate magnitude if all components are present
  if (data.gyroscope_x !== null && data.gyroscope_y !== null && data.gyroscope_z !== null) {
    processed.rotationRate = calculateMagnitude(
      data.gyroscope_x,
      data.gyroscope_y,
      data.gyroscope_z
    );
  }
  
  return processed;
} 