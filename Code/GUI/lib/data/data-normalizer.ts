'use client';

import { SensorDataEntity } from '@/types/database.types';
import { TimeSeriesDataPoint } from '@/types/database.types';

/**
 * Normalize timestamps in sensor data
 * 
 * ESP32 devices provide timestamps as milliseconds since boot,
 * which need to be converted to absolute timestamps for storage and visualization.
 * 
 * @param data Array of sensor data
 * @param referenceTime Reference time (current time if not provided)
 * @returns Normalized data with absolute timestamps
 */
export function normalizeTimestamps(
  data: SensorDataEntity[],
  referenceTime: number = Date.now()
): SensorDataEntity[] {
  if (data.length === 0) {
    return [];
  }
  
  // Find the minimum timestamp in the data
  const minTimestamp = Math.min(...data.map(point => point.timestamp));
  
  // Calculate the offset between the reference time and the minimum timestamp
  const offset = referenceTime - minTimestamp;
  
  // Apply the offset to all timestamps
  return data.map(point => ({
    ...point,
    timestamp: point.timestamp + offset,
  }));
}

/**
 * Normalize time series data to a specific time range
 * @param data Array of time series data points
 * @param startTime Start time of the range
 * @param endTime End time of the range
 * @returns Normalized data within the specified time range
 */
export function normalizeTimeRange(
  data: TimeSeriesDataPoint[],
  startTime: number,
  endTime: number
): TimeSeriesDataPoint[] {
  // Filter data to the specified time range
  return data.filter(point => point.timestamp >= startTime && point.timestamp <= endTime);
}

/**
 * Normalize values in time series data to a specific range
 * @param data Array of time series data points
 * @param minValue Minimum value of the range (default: 0)
 * @param maxValue Maximum value of the range (default: 1)
 * @returns Normalized data with values in the specified range
 */
export function normalizeValues(
  data: TimeSeriesDataPoint[],
  minValue: number = 0,
  maxValue: number = 1
): TimeSeriesDataPoint[] {
  if (data.length === 0) {
    return [];
  }
  
  // Find the minimum and maximum values in the data
  const values = data.map(point => point.value);
  const dataMin = Math.min(...values);
  const dataMax = Math.max(...values);
  
  // If all values are the same, return the original data
  if (dataMin === dataMax) {
    return data;
  }
  
  // Calculate the scale factor
  const scale = (maxValue - minValue) / (dataMax - dataMin);
  
  // Apply the normalization to all values
  return data.map(point => ({
    timestamp: point.timestamp,
    value: minValue + (point.value - dataMin) * scale,
  }));
}

/**
 * Resample time series data to a regular time interval
 * @param data Array of time series data points
 * @param interval Time interval in milliseconds
 * @param interpolate Whether to interpolate values (default: true)
 * @returns Resampled data with regular time intervals
 */
export function resampleData(
  data: TimeSeriesDataPoint[],
  interval: number,
  interpolate: boolean = true
): TimeSeriesDataPoint[] {
  if (data.length <= 1) {
    return data;
  }
  
  // Sort data by timestamp
  const sortedData = [...data].sort((a, b) => a.timestamp - b.timestamp);
  
  // Determine the start and end times
  const startTime = sortedData[0].timestamp;
  const endTime = sortedData[sortedData.length - 1].timestamp;
  
  // Create an array of regular timestamps
  const timestamps: number[] = [];
  for (let t = startTime; t <= endTime; t += interval) {
    timestamps.push(t);
  }
  
  // Resample the data
  const result: TimeSeriesDataPoint[] = [];
  
  for (const timestamp of timestamps) {
    // Find the closest data points
    let lowerIndex = -1;
    let upperIndex = -1;
    
    for (let i = 0; i < sortedData.length; i++) {
      if (sortedData[i].timestamp <= timestamp) {
        lowerIndex = i;
      } else {
        upperIndex = i;
        break;
      }
    }
    
    // Determine the value at this timestamp
    let value: number;
    
    if (lowerIndex === -1) {
      // Before the first data point
      value = sortedData[0].value;
    } else if (upperIndex === -1) {
      // After the last data point
      value = sortedData[sortedData.length - 1].value;
    } else if (interpolate) {
      // Interpolate between the two closest points
      const lowerPoint = sortedData[lowerIndex];
      const upperPoint = sortedData[upperIndex];
      
      const t = (timestamp - lowerPoint.timestamp) / (upperPoint.timestamp - lowerPoint.timestamp);
      value = lowerPoint.value + t * (upperPoint.value - lowerPoint.value);
    } else {
      // Use the closest point
      const lowerPoint = sortedData[lowerIndex];
      const upperPoint = sortedData[upperIndex];
      
      const lowerDist = timestamp - lowerPoint.timestamp;
      const upperDist = upperPoint.timestamp - timestamp;
      
      value = lowerDist <= upperDist ? lowerPoint.value : upperPoint.value;
    }
    
    result.push({
      timestamp,
      value,
    });
  }
  
  return result;
}

/**
 * Align multiple time series to a common time base
 * @param dataSeries Array of time series data arrays
 * @param interval Time interval for resampling
 * @returns Array of aligned time series
 */
export function alignTimeSeries(
  dataSeries: TimeSeriesDataPoint[][],
  interval: number
): TimeSeriesDataPoint[][] {
  if (dataSeries.length === 0) {
    return [];
  }
  
  // Find the global start and end times
  let globalStartTime = Infinity;
  let globalEndTime = -Infinity;
  
  for (const series of dataSeries) {
    if (series.length > 0) {
      const timestamps = series.map(point => point.timestamp);
      const seriesStartTime = Math.min(...timestamps);
      const seriesEndTime = Math.max(...timestamps);
      
      globalStartTime = Math.min(globalStartTime, seriesStartTime);
      globalEndTime = Math.max(globalEndTime, seriesEndTime);
    }
  }
  
  // If no valid data was found, return empty arrays
  if (globalStartTime === Infinity || globalEndTime === -Infinity) {
    return dataSeries;
  }
  
  // Resample each series to the common time base
  return dataSeries.map(series => resampleData(series, interval));
} 