'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import { 
  ConnectionStatus, 
  SerialPortFilter, 
  SerialPortConfig, 
  DEFAULT_ESP32_CONFIG,
  ESP32_FILTERS,
  SerialEventType,
  SerialEvent,
  DeviceInfo
} from '@/types/serial';
import { SerialConnection, isWebSerialSupported } from '@/lib/serial/serial-connection';

interface UseSerialConnectionOptions {
  autoConnect?: boolean;
  filters?: SerialPortFilter[];
  config?: SerialPortConfig;
}

interface UseSerialConnectionResult {
  isSupported: boolean;
  status: ConnectionStatus;
  error: Error | null;
  deviceInfo: DeviceInfo | null;
  isStreaming: boolean;
  connect: (filters?: SerialPortFilter[]) => Promise<void>;
  disconnect: () => Promise<void>;
  startStreaming: () => Promise<void>;
  stopStreaming: () => Promise<void>;
  setSampleRate: (sampleRate: number) => Promise<void>;
  resetDevice: () => Promise<void>;
  addEventListener: (eventType: SerialEventType, listener: (event: SerialEvent) => void) => void;
  removeEventListener: (eventType: SerialEventType, listener: (event: SerialEvent) => void) => void;
}

/**
 * React hook for managing serial connections to ESP32 devices
 * @param options Connection options
 * @returns Serial connection state and methods
 */
export function useSerialConnection(
  options: UseSerialConnectionOptions = {}
): UseSerialConnectionResult {
  const [status, setStatus] = useState<ConnectionStatus>(ConnectionStatus.DISCONNECTED);
  const [error, setError] = useState<Error | null>(null);
  const [deviceInfo, setDeviceInfo] = useState<DeviceInfo | null>(null);
  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const [isSupported, setIsSupported] = useState<boolean>(false);
  
  // Use a ref to store the SerialConnection instance to avoid recreating it on every render
  const connectionRef = useRef<SerialConnection | null>(null);
  
  // Initialize the serial connection
  useEffect(() => {
    // Check if Web Serial API is supported
    const supported = isWebSerialSupported();
    setIsSupported(supported);
    
    if (!supported) {
      return;
    }
    
    // Create a new SerialConnection instance
    connectionRef.current = new SerialConnection({
      config: options.config || DEFAULT_ESP32_CONFIG,
      filters: options.filters || ESP32_FILTERS,
      autoConnect: false, // We'll handle auto-connect in this hook
    });
    
    // Set up event listeners
    const handleConnect = (event: SerialEvent) => {
      setStatus(ConnectionStatus.CONNECTED);
      setError(null);
    };
    
    const handleDisconnect = (event: SerialEvent) => {
      setStatus(ConnectionStatus.DISCONNECTED);
      setDeviceInfo(null);
      setIsStreaming(false);
    };
    
    const handleError = (event: SerialEvent) => {
      setStatus(ConnectionStatus.ERROR);
      setError(event.data.error);
      setIsStreaming(false);
    };
    
    const connection = connectionRef.current;
    connection.addEventListener(SerialEventType.CONNECT, handleConnect);
    connection.addEventListener(SerialEventType.DISCONNECT, handleDisconnect);
    connection.addEventListener(SerialEventType.ERROR, handleError);
    
    // Auto-connect if specified
    if (options.autoConnect) {
      connect().catch(err => {
        console.error('Auto-connect failed:', err);
      });
    }
    
    // Clean up
    return () => {
      if (connection) {
        connection.removeEventListener(SerialEventType.CONNECT, handleConnect);
        connection.removeEventListener(SerialEventType.DISCONNECT, handleDisconnect);
        connection.removeEventListener(SerialEventType.ERROR, handleError);
        
        // Disconnect if connected
        if (status === ConnectionStatus.CONNECTED) {
          connection.disconnect().catch(err => {
            console.error('Disconnect on cleanup failed:', err);
          });
        }
      }
    };
  }, [options.autoConnect, options.config, options.filters]);
  
  // Update state from connection state
  useEffect(() => {
    const updateState = () => {
      if (!connectionRef.current) {
        return;
      }
      
      const state = connectionRef.current.getState();
      setStatus(state.status);
      setError(state.error);
      setDeviceInfo(state.deviceInfo);
      setIsStreaming(state.isStreaming);
    };
    
    // Update state immediately
    updateState();
    
    // Set up an interval to update state periodically
    const intervalId = setInterval(updateState, 1000);
    
    return () => {
      clearInterval(intervalId);
    };
  }, []);
  
  /**
   * Connect to an ESP32 device
   * @param filters Optional filters to specify which devices to show in the browser's device picker
   * @returns Promise that resolves when connected
   */
  const connect = useCallback(async (filters?: SerialPortFilter[]): Promise<void> => {
    if (!connectionRef.current) {
      throw new Error('Serial connection not initialized');
    }
    
    if (status === ConnectionStatus.CONNECTED) {
      return; // Already connected
    }
    
    try {
      setStatus(ConnectionStatus.CONNECTING);
      
      // Log the filters being used
      console.log('Connecting with filters:', filters || 'No filters');
      
      // Explicitly request port access before connecting
      if ('serial' in navigator && !filters) {
        console.log('No filters provided, using default ESP32 filters');
        filters = ESP32_FILTERS;
      }
      
      await connectionRef.current.connect(filters);
      
      // Get device info
      const info = await connectionRef.current.getDeviceInfo();
      setDeviceInfo(info);
    } catch (err) {
      console.error('Connection error:', err);
      setStatus(ConnectionStatus.ERROR);
      setError(err as Error);
      throw err;
    }
  }, [status]);
  
  /**
   * Disconnect from the ESP32 device
   * @returns Promise that resolves when disconnected
   */
  const disconnect = useCallback(async (): Promise<void> => {
    if (!connectionRef.current) {
      throw new Error('Serial connection not initialized');
    }
    
    if (status !== ConnectionStatus.CONNECTED) {
      return; // Not connected
    }
    
    try {
      await connectionRef.current.disconnect();
    } catch (err) {
      setError(err as Error);
      throw err;
    }
  }, [status]);
  
  /**
   * Start streaming sensor data from the ESP32 device
   * @returns Promise that resolves when streaming starts
   */
  const startStreaming = useCallback(async (): Promise<void> => {
    if (!connectionRef.current) {
      throw new Error('Serial connection not initialized');
    }
    
    if (status !== ConnectionStatus.CONNECTED) {
      throw new Error('Not connected to a device');
    }
    
    if (isStreaming) {
      return; // Already streaming
    }
    
    try {
      await connectionRef.current.startStreaming();
      setIsStreaming(true);
    } catch (err) {
      setError(err as Error);
      throw err;
    }
  }, [status, isStreaming]);
  
  /**
   * Stop streaming sensor data from the ESP32 device
   * @returns Promise that resolves when streaming stops
   */
  const stopStreaming = useCallback(async (): Promise<void> => {
    if (!connectionRef.current) {
      throw new Error('Serial connection not initialized');
    }
    
    if (status !== ConnectionStatus.CONNECTED) {
      throw new Error('Not connected to a device');
    }
    
    if (!isStreaming) {
      return; // Not streaming
    }
    
    try {
      await connectionRef.current.stopStreaming();
      setIsStreaming(false);
    } catch (err) {
      setError(err as Error);
      throw err;
    }
  }, [status, isStreaming]);
  
  /**
   * Set the sample rate for sensor data
   * @param sampleRate Sample rate in Hz
   * @returns Promise that resolves when sample rate is set
   */
  const setSampleRate = useCallback(async (sampleRate: number): Promise<void> => {
    if (!connectionRef.current) {
      throw new Error('Serial connection not initialized');
    }
    
    if (status !== ConnectionStatus.CONNECTED) {
      throw new Error('Not connected to a device');
    }
    
    try {
      await connectionRef.current.setSampleRate(sampleRate);
      
      // Update device info
      if (deviceInfo) {
        setDeviceInfo({
          ...deviceInfo,
          sampleRate,
        });
      }
    } catch (err) {
      setError(err as Error);
      throw err;
    }
  }, [status, deviceInfo]);
  
  /**
   * Reset the ESP32 device
   * @returns Promise that resolves when device is reset
   */
  const resetDevice = useCallback(async (): Promise<void> => {
    if (!connectionRef.current) {
      throw new Error('Serial connection not initialized');
    }
    
    if (status !== ConnectionStatus.CONNECTED) {
      throw new Error('Not connected to a device');
    }
    
    try {
      await connectionRef.current.resetDevice();
    } catch (err) {
      setError(err as Error);
      throw err;
    }
  }, [status]);
  
  /**
   * Add an event listener
   * @param eventType Event type to listen for
   * @param listener Function to call when event occurs
   */
  const addEventListener = useCallback((
    eventType: SerialEventType,
    listener: (event: SerialEvent) => void
  ): void => {
    if (!connectionRef.current) {
      throw new Error('Serial connection not initialized');
    }
    
    connectionRef.current.addEventListener(eventType, listener);
  }, []);
  
  /**
   * Remove an event listener
   * @param eventType Event type to stop listening for
   * @param listener Function to remove
   */
  const removeEventListener = useCallback((
    eventType: SerialEventType,
    listener: (event: SerialEvent) => void
  ): void => {
    if (!connectionRef.current) {
      throw new Error('Serial connection not initialized');
    }
    
    connectionRef.current.removeEventListener(eventType, listener);
  }, []);
  
  return {
    isSupported,
    status,
    error,
    deviceInfo,
    isStreaming,
    connect,
    disconnect,
    startStreaming,
    stopStreaming,
    setSampleRate,
    resetDevice,
    addEventListener,
    removeEventListener,
  };
} 