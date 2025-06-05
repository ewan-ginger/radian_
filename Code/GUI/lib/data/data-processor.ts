'use client';

import { SensorDataEntity } from '@/types/database.types';
import { TimeSeriesDataPoint, OrientationData } from '@/types/database.types';
import { calculateMagnitude } from '@/lib/services/sensor-data-service';

/**
 * Process raw sensor data to add derived properties
 * @param data Raw sensor data
 * @returns Processed sensor data with derived properties
 */
export function processSensorData(data: SensorDataEntity): SensorDataEntity {
  const processed: SensorDataEntity = { ...data };
  
  // Calculate acceleration magnitude if all components are present
  if (
    typeof data.accelerometer_x === 'number' &&
    typeof data.accelerometer_y === 'number' &&
    typeof data.accelerometer_z === 'number'
  ) {
    processed.acceleration = calculateMagnitude(
      data.accelerometer_x,
      data.accelerometer_y,
      data.accelerometer_z
    );
  }
  
  // Calculate rotation rate magnitude if all components are present
  if (
    typeof data.gyroscope_x === 'number' &&
    typeof data.gyroscope_y === 'number' &&
    typeof data.gyroscope_z === 'number'
  ) {
    processed.rotationRate = calculateMagnitude(
      data.gyroscope_x,
      data.gyroscope_y,
      data.gyroscope_z
    );
  }
  
  return processed;
}

/**
 * Calculate statistics for a set of sensor data
 * @param data Array of sensor data
 * @returns Statistics object with min, max, avg, and sum for each property
 */
export function calculateStatistics(data: SensorDataEntity[]): Record<string, { min: number; max: number; avg: number; sum: number }> {
  if (data.length === 0) {
    return {};
  }
  
  // Initialize statistics object
  const stats: Record<string, { min: number; max: number; avg: number; sum: number }> = {};
  
  // Process each data point
  data.forEach(point => {
    // Process each numeric property
    Object.entries(point).forEach(([key, value]) => {
      if (typeof value === 'number') {
        if (!stats[key]) {
          stats[key] = {
            min: value,
            max: value,
            avg: 0,
            sum: value,
          };
        } else {
          stats[key].min = Math.min(stats[key].min, value);
          stats[key].max = Math.max(stats[key].max, value);
          stats[key].sum += value;
        }
      }
    });
  });
  
  // Calculate averages
  Object.keys(stats).forEach(key => {
    stats[key].avg = stats[key].sum / data.length;
  });
  
  return stats;
}

/**
 * Extract time series data for a specific property
 * @param data Array of sensor data
 * @param property Property to extract
 * @param limit Maximum number of points to return (default: 100)
 * @returns Array of time series data points
 */
export function extractTimeSeriesData(
  data: SensorDataEntity[],
  property: keyof SensorDataEntity,
  limit: number = 100
): TimeSeriesDataPoint[] {
  // Sort data by timestamp
  const sortedData = [...data].sort((a, b) => a.timestamp - b.timestamp);
  
  // If there are more data points than the limit, downsample
  let processedData = sortedData;
  if (sortedData.length > limit) {
    processedData = downsampleData(sortedData, limit);
  }
  
  // Extract the property values
  return processedData.map(point => ({
    timestamp: point.timestamp,
    value: point[property] as number || 0,
  }));
}

/**
 * Extract orientation data for 3D visualization
 * @param data Array of sensor data
 * @param limit Maximum number of points to return (default: 100)
 * @returns Array of orientation data points
 */
export function extractOrientationData(
  data: SensorDataEntity[],
  limit: number = 100
): OrientationData[] {
  // Sort data by timestamp
  const sortedData = [...data].sort((a, b) => a.timestamp - b.timestamp);
  
  // If there are more data points than the limit, downsample
  let processedData = sortedData;
  if (sortedData.length > limit) {
    processedData = downsampleData(sortedData, limit);
  }
  
  // Extract orientation values
  return processedData.map(point => ({
    timestamp: point.timestamp,
    x: point.orientation_x || 0,
    y: point.orientation_y || 0,
    z: point.orientation_z || 0,
  }));
}

/**
 * Downsample data to a specified number of points
 * @param data Array of sensor data
 * @param targetCount Target number of points
 * @returns Downsampled data
 */
export function downsampleData<T>(data: T[], targetCount: number): T[] {
  if (data.length <= targetCount) {
    return data;
  }
  
  const result: T[] = [];
  const step = data.length / targetCount;
  
  for (let i = 0; i < targetCount; i++) {
    const index = Math.floor(i * step);
    result.push(data[index]);
  }
  
  return result;
}

/**
 * Smooth data using a simple moving average
 * @param data Array of time series data points
 * @param windowSize Window size for the moving average (default: 5)
 * @returns Smoothed data
 */
export function smoothData(data: TimeSeriesDataPoint[], windowSize: number = 5): TimeSeriesDataPoint[] {
  if (data.length <= windowSize) {
    return data;
  }
  
  const result: TimeSeriesDataPoint[] = [];
  
  for (let i = 0; i < data.length; i++) {
    let sum = 0;
    let count = 0;
    
    for (let j = Math.max(0, i - Math.floor(windowSize / 2)); j <= Math.min(data.length - 1, i + Math.floor(windowSize / 2)); j++) {
      sum += data[j].value;
      count++;
    }
    
    result.push({
      timestamp: data[i].timestamp,
      value: sum / count,
    });
  }
  
  return result;
}

/**
 * Calculate the derivative of a time series
 * @param data Array of time series data points
 * @returns Derivative of the time series
 */
export function calculateDerivative(data: TimeSeriesDataPoint[]): TimeSeriesDataPoint[] {
  if (data.length <= 1) {
    return [];
  }
  
  const result: TimeSeriesDataPoint[] = [];
  
  for (let i = 1; i < data.length; i++) {
    const dt = data[i].timestamp - data[i - 1].timestamp;
    const dy = data[i].value - data[i - 1].value;
    
    if (dt > 0) {
      result.push({
        timestamp: data[i].timestamp,
        value: dy / dt,
      });
    }
  }
  
  return result;
}

/**
 * Calculate the integral of a time series
 * @param data Array of time series data points
 * @returns Integral of the time series
 */
export function calculateIntegral(data: TimeSeriesDataPoint[]): TimeSeriesDataPoint[] {
  if (data.length <= 1) {
    return [];
  }
  
  const result: TimeSeriesDataPoint[] = [];
  let integral = 0;
  
  for (let i = 1; i < data.length; i++) {
    const dt = data[i].timestamp - data[i - 1].timestamp;
    const avgValue = (data[i].value + data[i - 1].value) / 2;
    
    integral += avgValue * dt;
    
    result.push({
      timestamp: data[i].timestamp,
      value: integral,
    });
  }
  
  return result;
} 