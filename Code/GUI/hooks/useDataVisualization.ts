'use client';

import { useState, useEffect, useCallback } from 'react';

interface DataPoint {
  timestamp: number;
  x: number;
  y: number;
  z: number;
}

interface UseDataVisualizationOptions {
  maxDataPoints?: number;
  initialDataRate?: number;
  initialVisiblePoints?: number;
}

export function useDataVisualization({
  maxDataPoints = 1000,
  initialDataRate = 50,
  initialVisiblePoints = 100
}: UseDataVisualizationOptions = {}) {
  const [isRecording, setIsRecording] = useState(false);
  const [dataType, setDataType] = useState('orientation');
  const [dataRate, setDataRate] = useState(initialDataRate);
  const [visiblePoints, setVisiblePoints] = useState(initialVisiblePoints);
  const [orientationData, setOrientationData] = useState<DataPoint[]>([]);
  const [accelerometerData, setAccelerometerData] = useState<DataPoint[]>([]);
  const [gyroscopeData, setGyroscopeData] = useState<DataPoint[]>([]);
  const [magnetometerData, setMagnetometerData] = useState<DataPoint[]>([]);
  
  // Add a new data point
  const addDataPoint = useCallback((dataPoint: DataPoint) => {
    setOrientationData(prev => {
      const newData = [...prev, {
        timestamp: dataPoint.timestamp,
        x: dataPoint.x,
        y: dataPoint.y,
        z: dataPoint.z
      }];
      
      // Limit the number of data points to prevent memory issues
      if (newData.length > maxDataPoints) {
        return newData.slice(newData.length - maxDataPoints);
      }
      
      return newData;
    });
    
    setAccelerometerData(prev => {
      const newData = [...prev, {
        timestamp: dataPoint.timestamp,
        x: dataPoint.x * 0.2, // Scale for demonstration
        y: dataPoint.y * 0.2,
        z: dataPoint.z * 0.2
      }];
      
      if (newData.length > maxDataPoints) {
        return newData.slice(newData.length - maxDataPoints);
      }
      
      return newData;
    });
    
    setGyroscopeData(prev => {
      const newData = [...prev, {
        timestamp: dataPoint.timestamp,
        x: dataPoint.x * 0.1, // Scale for demonstration
        y: dataPoint.y * 0.1,
        z: dataPoint.z * 0.1
      }];
      
      if (newData.length > maxDataPoints) {
        return newData.slice(newData.length - maxDataPoints);
      }
      
      return newData;
    });
    
    setMagnetometerData(prev => {
      const newData = [...prev, {
        timestamp: dataPoint.timestamp,
        x: dataPoint.x * 0.05, // Scale for demonstration
        y: dataPoint.y * 0.05,
        z: dataPoint.z * 0.05
      }];
      
      if (newData.length > maxDataPoints) {
        return newData.slice(newData.length - maxDataPoints);
      }
      
      return newData;
    });
  }, [maxDataPoints]);
  
  // Start recording
  const startRecording = useCallback(() => {
    setIsRecording(true);
  }, []);
  
  // Stop recording
  const stopRecording = useCallback(() => {
    setIsRecording(false);
  }, []);
  
  // Reset all data
  const resetData = useCallback(() => {
    setOrientationData([]);
    setAccelerometerData([]);
    setGyroscopeData([]);
    setMagnetometerData([]);
  }, []);
  
  // Get the current data based on the selected data type
  const getCurrentData = useCallback(() => {
    switch (dataType) {
      case 'orientation':
        return orientationData;
      case 'accelerometer':
        return accelerometerData;
      case 'gyroscope':
        return gyroscopeData;
      case 'magnetometer':
        return magnetometerData;
      default:
        return orientationData;
    }
  }, [dataType, orientationData, accelerometerData, gyroscopeData, magnetometerData]);
  
  return {
    isRecording,
    startRecording,
    stopRecording,
    resetData,
    addDataPoint,
    dataType,
    setDataType,
    dataRate,
    setDataRate,
    visiblePoints,
    setVisiblePoints,
    orientationData,
    accelerometerData,
    gyroscopeData,
    magnetometerData,
    getCurrentData
  };
} 