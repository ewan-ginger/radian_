'use client';

import { useState, useEffect, useCallback } from 'react';
import { useSerialConnection } from './useSerialConnection';
import { ConnectionStatus, SerialEventType, SerialEvent, DeviceInfo } from '@/types/serial';
import { parseSensorData } from '@/lib/serial/serial-protocol';

interface SensorReading {
  timestamp: number;
  accelerometer: { x: number; y: number; z: number };
  gyroscope: { x: number; y: number; z: number };
  magnetometer: { x: number; y: number; z: number };
  orientation: { x: number; y: number; z: number };
  batteryLevel: number;
}

interface UseDeviceConnectionResult {
  isSupported: boolean;
  status: ConnectionStatus;
  error: Error | null;
  deviceInfo: DeviceInfo | null;
  isStreaming: boolean;
  sensorData: SensorReading[];
  connect: () => Promise<void>;
  disconnect: () => Promise<void>;
  startStreaming: () => Promise<void>;
  stopStreaming: () => Promise<void>;
  setSampleRate: (sampleRate: number) => Promise<void>;
  resetDevice: () => Promise<void>;
  clearData: () => void;
}

export function useDeviceConnection(): UseDeviceConnectionResult {
  const {
    isSupported,
    status,
    error,
    deviceInfo,
    isStreaming,
    connect: serialConnect,
    disconnect: serialDisconnect,
    startStreaming: serialStartStreaming,
    stopStreaming: serialStopStreaming,
    setSampleRate: serialSetSampleRate,
    resetDevice: serialResetDevice,
    addEventListener,
    removeEventListener,
  } = useSerialConnection();

  const [sensorData, setSensorData] = useState<SensorReading[]>([]);
  const [isConnecting, setIsConnecting] = useState(false);

  // Handle data received from the device
  useEffect(() => {
    const handleDataReceived = (event: SerialEvent) => {
      if (event.type === SerialEventType.DATA_RECEIVED && Array.isArray(event.data)) {
        try {
          // Parse the raw sensor data
          const parsedData = parseSensorData(event.data);
          
          if (parsedData) {
            // Create a sensor reading object
            const reading: SensorReading = {
              timestamp: event.timestamp,
              accelerometer: {
                x: parsedData.accelerometer_x,
                y: parsedData.accelerometer_y,
                z: parsedData.accelerometer_z,
              },
              gyroscope: {
                x: parsedData.gyroscope_x,
                y: parsedData.gyroscope_y,
                z: parsedData.gyroscope_z,
              },
              magnetometer: {
                x: parsedData.magnetometer_x,
                y: parsedData.magnetometer_y,
                z: parsedData.magnetometer_z,
              },
              orientation: {
                x: parsedData.orientation_x,
                y: parsedData.orientation_y,
                z: parsedData.orientation_z,
              },
              batteryLevel: parsedData.battery_level,
            };
            
            // Add the reading to the sensor data
            setSensorData(prevData => {
              const newData = [...prevData, reading];
              
              // Keep only the last 1000 readings to prevent memory issues
              if (newData.length > 1000) {
                return newData.slice(newData.length - 1000);
              }
              
              return newData;
            });
          }
        } catch (err) {
          console.error('Error parsing sensor data:', err);
        }
      }
    };

    addEventListener(SerialEventType.DATA_RECEIVED, handleDataReceived);

    return () => {
      removeEventListener(SerialEventType.DATA_RECEIVED, handleDataReceived);
    };
  }, [addEventListener, removeEventListener]);

  // Handle connection status changes
  useEffect(() => {
    if (status === ConnectionStatus.CONNECTED) {
      setIsConnecting(false);
    } else if (status === ConnectionStatus.DISCONNECTED && isConnecting) {
      setIsConnecting(false);
    }
  }, [status, isConnecting]);

  // Connect to the device
  const connect = useCallback(async () => {
    try {
      setIsConnecting(true);
      await serialConnect();
    } catch (err) {
      setIsConnecting(false);
      console.error('Failed to connect to device:', err);
    }
  }, [serialConnect]);

  // Disconnect from the device
  const disconnect = useCallback(async () => {
    try {
      await serialDisconnect();
      setSensorData([]);
    } catch (err) {
      console.error('Failed to disconnect from device:', err);
    }
  }, [serialDisconnect]);

  // Start streaming data from the device
  const startStreaming = useCallback(async () => {
    try {
      await serialStartStreaming();
    } catch (err) {
      console.error('Failed to start streaming:', err);
    }
  }, [serialStartStreaming]);

  // Stop streaming data from the device
  const stopStreaming = useCallback(async () => {
    try {
      await serialStopStreaming();
    } catch (err) {
      console.error('Failed to stop streaming:', err);
    }
  }, [serialStopStreaming]);

  // Set the sample rate for the device
  const setSampleRate = useCallback(async (sampleRate: number) => {
    try {
      await serialSetSampleRate(sampleRate);
    } catch (err) {
      console.error('Failed to set sample rate:', err);
    }
  }, [serialSetSampleRate]);

  // Reset the device
  const resetDevice = useCallback(async () => {
    try {
      await serialResetDevice();
      setSensorData([]);
    } catch (err) {
      console.error('Failed to reset device:', err);
    }
  }, [serialResetDevice]);

  // Clear the sensor data
  const clearData = useCallback(() => {
    setSensorData([]);
  }, []);

  return {
    isSupported,
    status,
    error,
    deviceInfo,
    isStreaming,
    sensorData,
    connect,
    disconnect,
    startStreaming,
    stopStreaming,
    setSampleRate,
    resetDevice,
    clearData,
  };
} 