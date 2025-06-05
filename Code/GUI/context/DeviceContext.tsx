'use client';

import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { useSerialConnection } from '@/hooks/useSerialConnection';
import { ConnectionStatus, SerialEventType, SerialEvent, DeviceInfo, ESP32_FILTERS } from '@/types/serial';

interface DeviceContextType {
  isSupported: boolean;
  status: ConnectionStatus;
  error: Error | null;
  deviceInfo: DeviceInfo | null;
  isStreaming: boolean;
  sensorData: number[][] | null;
  connect: () => Promise<void>;
  disconnect: () => Promise<void>;
  startStreaming: () => Promise<void>;
  stopStreaming: () => Promise<void>;
  setSampleRate: (sampleRate: number) => Promise<void>;
  resetDevice: () => Promise<void>;
}

const DeviceContext = createContext<DeviceContextType | undefined>(undefined);

interface DeviceProviderProps {
  children: ReactNode;
}

export function DeviceProvider({ children }: DeviceProviderProps) {
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

  const [sensorData, setSensorData] = useState<number[][] | null>(null);

  // Handle data received from the device
  useEffect(() => {
    const handleDataReceived = (event: SerialEvent) => {
      if (event.type === SerialEventType.DATA_RECEIVED && Array.isArray(event.data)) {
        setSensorData(prevData => {
          if (!prevData) return [event.data];
          
          // Keep only the last 1000 data points to prevent memory issues
          const newData = [...prevData, event.data];
          if (newData.length > 1000) {
            return newData.slice(newData.length - 1000);
          }
          return newData;
        });
      }
    };

    addEventListener(SerialEventType.DATA_RECEIVED, handleDataReceived);

    return () => {
      removeEventListener(SerialEventType.DATA_RECEIVED, handleDataReceived);
    };
  }, [addEventListener, removeEventListener]);

  // Connect to the device with ESP32 filters
  const connect = async () => {
    try {
      await serialConnect(ESP32_FILTERS);
    } catch (err) {
      console.error('Failed to connect to device:', err);
    }
  };

  // Disconnect from the device
  const disconnect = async () => {
    try {
      await serialDisconnect();
      setSensorData(null);
    } catch (err) {
      console.error('Failed to disconnect from device:', err);
    }
  };

  // Start streaming data from the device
  const startStreaming = async () => {
    try {
      await serialStartStreaming();
    } catch (err) {
      console.error('Failed to start streaming:', err);
    }
  };

  // Stop streaming data from the device
  const stopStreaming = async () => {
    try {
      await serialStopStreaming();
    } catch (err) {
      console.error('Failed to stop streaming:', err);
    }
  };

  // Set the sample rate for the device
  const setSampleRate = async (sampleRate: number) => {
    try {
      await serialSetSampleRate(sampleRate);
    } catch (err) {
      console.error('Failed to set sample rate:', err);
    }
  };

  // Reset the device
  const resetDevice = async () => {
    try {
      await serialResetDevice();
      setSensorData(null);
    } catch (err) {
      console.error('Failed to reset device:', err);
    }
  };

  const value = {
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
  };

  return (
    <DeviceContext.Provider value={value}>
      {children}
    </DeviceContext.Provider>
  );
}

export function useDevice() {
  const context = useContext(DeviceContext);
  if (context === undefined) {
    throw new Error('useDevice must be used within a DeviceProvider');
  }
  return context;
} 