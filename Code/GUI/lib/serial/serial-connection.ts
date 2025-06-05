'use client';

import { 
  ConnectionStatus, 
  SerialPortConfig, 
  DEFAULT_ESP32_CONFIG,
  SerialConnectionOptions,
  SerialPortFilter,
  ESP32_FILTERS,
  CommandType,
  MessageType,
  SerialMessage,
  SerialEvent,
  SerialEventType,
  DeviceInfo,
  SerialConnectionState
} from '@/types/serial';

import {
  encodeMessage,
  decodeMessage,
  createCommandMessage,
  parseSensorData
} from './serial-protocol';

/**
 * Check if Web Serial API is supported in the current browser
 * @returns True if Web Serial API is supported
 */
export function isWebSerialSupported(): boolean {
  return 'serial' in navigator;
}

/**
 * Class for managing Web Serial API connections to ESP32 devices
 */
export class SerialConnection {
  private port: SerialPort | null = null;
  private reader: ReadableStreamDefaultReader<Uint8Array> | null = null;
  private writer: WritableStreamDefaultWriter<Uint8Array> | null = null;
  private readLoopPromise: Promise<void> | null = null;
  private abortController: AbortController | null = null;
  private eventListeners: Map<SerialEventType, ((event: SerialEvent) => void)[]> = new Map();
  private buffer: Uint8Array = new Uint8Array(0);
  private config: SerialPortConfig;
  private status: ConnectionStatus = ConnectionStatus.DISCONNECTED;
  private deviceInfo: DeviceInfo | null = null;
  private isStreaming: boolean = false;
  private error: Error | null = null;

  /**
   * Create a new SerialConnection instance
   * @param options Connection options
   */
  constructor(options: SerialConnectionOptions = {}) {
    this.config = options.config || DEFAULT_ESP32_CONFIG;
    
    // Initialize event listener maps
    Object.values(SerialEventType).forEach(eventType => {
      this.eventListeners.set(eventType, []);
    });
    
    // Auto-connect if specified
    if (options.autoConnect) {
      this.connect(options.filters);
    }
  }

  /**
   * Get the current connection state
   * @returns Current connection state
   */
  public getState(): SerialConnectionState {
    return {
      port: this.port,
      reader: this.reader,
      writer: this.writer,
      status: this.status,
      error: this.error,
      deviceInfo: this.deviceInfo,
      isStreaming: this.isStreaming
    };
  }

  /**
   * Connect to an ESP32 device
   * @param filters Optional filters to specify which devices to show in the browser's device picker
   * @returns Promise that resolves when connected
   */
  public async connect(filters?: SerialPortFilter[]): Promise<void> {
    if (!isWebSerialSupported()) {
      throw new Error('Web Serial API is not supported in this browser');
    }
    
    if (this.status === ConnectionStatus.CONNECTED) {
      return; // Already connected
    }
    
    try {
      this.setStatus(ConnectionStatus.CONNECTING);
      
      console.log('Requesting port with filters:', filters || ESP32_FILTERS);
      
      // Request port from user with more relaxed filters if none provided
      if (!filters || filters.length === 0) {
        console.log('No filters provided, using empty filters to show all devices');
        // Request port without filters to show all available devices
        this.port = await navigator.serial.requestPort({});
      } else {
        // Request port with the provided filters
        this.port = await navigator.serial.requestPort({
          filters: filters
        });
      }
      
      console.log('Port selected:', this.port);
      
      // Open the port
      await this.port.open(this.config);
      console.log('Port opened with config:', this.config);
      
      // Set up streams
      this.setupStreams();
      
      // Start the read loop
      this.startReadLoop();
      
      // Get device info
      await this.getDeviceInfo();
      
      this.setStatus(ConnectionStatus.CONNECTED);
      this.dispatchEvent(SerialEventType.CONNECT, { port: this.port });
    } catch (error) {
      console.error('Connection error:', error);
      this.handleError(error as Error);
      throw error;
    }
  }

  /**
   * Disconnect from the ESP32 device
   * @returns Promise that resolves when disconnected
   */
  public async disconnect(): Promise<void> {
    if (this.status !== ConnectionStatus.CONNECTED) {
      return; // Not connected
    }
    
    try {
      // Stop streaming if active
      if (this.isStreaming) {
        await this.stopStreaming();
      }
      
      // Stop the read loop
      this.stopReadLoop();
      
      // Close the port
      if (this.port) {
        await this.port.close();
      }
      
      // Reset state
      this.port = null;
      this.reader = null;
      this.writer = null;
      this.deviceInfo = null;
      
      this.setStatus(ConnectionStatus.DISCONNECTED);
      this.dispatchEvent(SerialEventType.DISCONNECT, {});
    } catch (error) {
      this.handleError(error as Error);
      throw error;
    }
  }

  /**
   * Start streaming sensor data from the ESP32 device
   * @returns Promise that resolves when streaming starts
   */
  public async startStreaming(): Promise<void> {
    if (this.status !== ConnectionStatus.CONNECTED) {
      throw new Error('Not connected to a device');
    }
    
    if (this.isStreaming) {
      return; // Already streaming
    }
    
    try {
      const message = createCommandMessage(CommandType.START_STREAMING);
      await this.sendMessage(message);
      this.isStreaming = true;
    } catch (error) {
      this.handleError(error as Error);
      throw error;
    }
  }

  /**
   * Stop streaming sensor data from the ESP32 device
   * @returns Promise that resolves when streaming stops
   */
  public async stopStreaming(): Promise<void> {
    if (this.status !== ConnectionStatus.CONNECTED) {
      throw new Error('Not connected to a device');
    }
    
    if (!this.isStreaming) {
      return; // Not streaming
    }
    
    try {
      const message = createCommandMessage(CommandType.STOP_STREAMING);
      await this.sendMessage(message);
      this.isStreaming = false;
    } catch (error) {
      this.handleError(error as Error);
      throw error;
    }
  }

  /**
   * Set the sample rate for sensor data
   * @param sampleRate Sample rate in Hz
   * @returns Promise that resolves when sample rate is set
   */
  public async setSampleRate(sampleRate: number): Promise<void> {
    if (this.status !== ConnectionStatus.CONNECTED) {
      throw new Error('Not connected to a device');
    }
    
    try {
      const message = createCommandMessage(CommandType.SET_SAMPLE_RATE, { sampleRate });
      await this.sendMessage(message);
      
      // Update device info
      if (this.deviceInfo) {
        this.deviceInfo.sampleRate = sampleRate;
      }
    } catch (error) {
      this.handleError(error as Error);
      throw error;
    }
  }

  /**
   * Reset the ESP32 device
   * @returns Promise that resolves when device is reset
   */
  public async resetDevice(): Promise<void> {
    if (this.status !== ConnectionStatus.CONNECTED) {
      throw new Error('Not connected to a device');
    }
    
    try {
      const message = createCommandMessage(CommandType.RESET);
      await this.sendMessage(message);
      
      // Disconnect and reconnect
      await this.disconnect();
      await this.connect();
    } catch (error) {
      this.handleError(error as Error);
      throw error;
    }
  }

  /**
   * Get information about the connected ESP32 device
   * @returns Promise that resolves with device information
   */
  public async getDeviceInfo(): Promise<DeviceInfo> {
    if (this.status !== ConnectionStatus.CONNECTED) {
      throw new Error('Not connected to a device');
    }
    
    try {
      const message = createCommandMessage(CommandType.GET_DEVICE_INFO);
      await this.sendMessage(message);
      
      // Wait for device info response (timeout after 3 seconds)
      const timeoutPromise = new Promise<DeviceInfo>((_, reject) => {
        setTimeout(() => reject(new Error('Timeout waiting for device info')), 3000);
      });
      
      const infoPromise = new Promise<DeviceInfo>((resolve) => {
        const listener = (event: SerialEvent) => {
          if (event.type === SerialEventType.DATA_RECEIVED && 
              event.data.type === MessageType.ACK && 
              event.data.payload.deviceInfo) {
            this.removeEventListener(SerialEventType.DATA_RECEIVED, listener);
            resolve(event.data.payload.deviceInfo);
          }
        };
        
        this.addEventListener(SerialEventType.DATA_RECEIVED, listener);
      });
      
      // Race the promises
      this.deviceInfo = await Promise.race([infoPromise, timeoutPromise]);
      return this.deviceInfo;
    } catch (error) {
      this.handleError(error as Error);
      throw error;
    }
  }

  /**
   * Add an event listener
   * @param eventType Event type to listen for
   * @param listener Function to call when event occurs
   */
  public addEventListener(eventType: SerialEventType, listener: (event: SerialEvent) => void): void {
    const listeners = this.eventListeners.get(eventType) || [];
    listeners.push(listener);
    this.eventListeners.set(eventType, listeners);
  }

  /**
   * Remove an event listener
   * @param eventType Event type to stop listening for
   * @param listener Function to remove
   */
  public removeEventListener(eventType: SerialEventType, listener: (event: SerialEvent) => void): void {
    const listeners = this.eventListeners.get(eventType) || [];
    const index = listeners.indexOf(listener);
    
    if (index !== -1) {
      listeners.splice(index, 1);
      this.eventListeners.set(eventType, listeners);
    }
  }

  /**
   * Send a message to the ESP32 device
   * @param message Message to send
   * @returns Promise that resolves when message is sent
   */
  private async sendMessage(message: SerialMessage): Promise<void> {
    if (!this.writer) {
      throw new Error('Serial port writer not available');
    }
    
    try {
      const encodedMessage = encodeMessage(message);
      await this.writer.write(encodedMessage);
    } catch (error) {
      this.handleError(error as Error);
      throw error;
    }
  }

  /**
   * Set up read and write streams for the serial port
   */
  private setupStreams(): void {
    if (!this.port) {
      throw new Error('Serial port not available');
    }
    
    // Set up abort controller for cancelling the read loop
    this.abortController = new AbortController();
    
    // Set up readable stream
    const readable = this.port.readable;
    if (!readable) {
      throw new Error('Serial port readable stream not available');
    }
    this.reader = readable.getReader();
    
    // Set up writable stream
    const writable = this.port.writable;
    if (!writable) {
      throw new Error('Serial port writable stream not available');
    }
    this.writer = writable.getWriter();
  }

  /**
   * Start the read loop to continuously read data from the serial port
   */
  private startReadLoop(): void {
    if (!this.reader) {
      throw new Error('Serial port reader not available');
    }
    
    this.readLoopPromise = this.readLoop();
  }

  /**
   * Stop the read loop
   */
  private stopReadLoop(): void {
    if (this.abortController) {
      this.abortController.abort();
      this.abortController = null;
    }
    
    if (this.reader) {
      this.reader.releaseLock();
      this.reader = null;
    }
    
    if (this.writer) {
      this.writer.releaseLock();
      this.writer = null;
    }
  }

  /**
   * Main read loop for processing incoming data
   */
  private async readLoop(): Promise<void> {
    if (!this.reader || !this.abortController) {
      return;
    }
    
    try {
      while (true) {
        // Check if we should abort
        if (this.abortController.signal.aborted) {
          break;
        }
        
        // Read data from the serial port
        const { value, done } = await this.reader.read();
        
        if (done) {
          break;
        }
        
        if (value) {
          // Append new data to the buffer
          const newBuffer = new Uint8Array(this.buffer.length + value.length);
          newBuffer.set(this.buffer);
          newBuffer.set(value, this.buffer.length);
          this.buffer = newBuffer;
          
          // Process the buffer
          this.processBuffer();
        }
      }
    } catch (error) {
      this.handleError(error as Error);
    } finally {
      // Clean up
      if (this.reader) {
        this.reader.releaseLock();
        this.reader = null;
      }
    }
  }

  /**
   * Process the buffer to extract complete messages
   */
  private processBuffer(): void {
    // Look for start and end markers
    let startIndex = -1;
    let endIndex = -1;
    
    for (let i = 0; i < this.buffer.length; i++) {
      if (this.buffer[i] === 0x24 && startIndex === -1) { // '$'
        startIndex = i;
      } else if (this.buffer[i] === 0x23 && startIndex !== -1) { // '#'
        endIndex = i;
        
        // Extract the message
        const messageBuffer = this.buffer.slice(startIndex, endIndex + 1);
        const message = decodeMessage(messageBuffer);
        
        if (message) {
          // Dispatch the message
          this.dispatchEvent(SerialEventType.DATA_RECEIVED, message);
          
          // Process data messages
          if (message.type === MessageType.DATA) {
            try {
              const sensorData = parseSensorData(message);
              // Convert to SensorDataInsert format for database
              const dataForDb = {
                session_id: '', // This would be set by the recording service
                device_id: sensorData.deviceID,
                timestamp: sensorData.timestamp,
                accelerometer_x: sensorData.accelerometer_x,
                accelerometer_y: sensorData.accelerometer_y,
                accelerometer_z: sensorData.accelerometer_z,
                gyroscope_x: sensorData.gyroscope_x,
                gyroscope_y: sensorData.gyroscope_y,
                gyroscope_z: sensorData.gyroscope_z,
                magnetometer_x: sensorData.magnetometer_x,
                magnetometer_y: sensorData.magnetometer_y,
                magnetometer_z: sensorData.magnetometer_z,
                orientation_x: sensorData.orientation_x,
                orientation_y: sensorData.orientation_y,
                orientation_z: sensorData.orientation_z,
                battery_level: sensorData.battery_level,
              };
              
              // This would be handled by the recording service
              // For now, just log it
              console.debug('Received sensor data:', dataForDb);
            } catch (error) {
              console.error('Error parsing sensor data:', error);
            }
          }
        }
        
        // Remove the processed message from the buffer
        this.buffer = this.buffer.slice(endIndex + 1);
        
        // Reset indices and start over
        startIndex = -1;
        endIndex = -1;
        i = -1; // Will be incremented to 0 in the next iteration
      }
    }
  }

  /**
   * Dispatch an event to all registered listeners
   * @param eventType Type of event to dispatch
   * @param data Data to include with the event
   */
  private dispatchEvent(eventType: SerialEventType, data: any): void {
    const event: SerialEvent = {
      type: eventType,
      timestamp: Date.now(),
      data,
    };
    
    const listeners = this.eventListeners.get(eventType) || [];
    listeners.forEach(listener => {
      try {
        listener(event);
      } catch (error) {
        console.error('Error in event listener:', error);
      }
    });
  }

  /**
   * Handle errors that occur during serial communication
   * @param error Error that occurred
   */
  private handleError(error: Error): void {
    this.error = error;
    this.setStatus(ConnectionStatus.ERROR);
    
    console.error('Serial connection error:', error);
    
    this.dispatchEvent(SerialEventType.ERROR, { error });
    
    // Clean up
    this.stopReadLoop();
    
    // Close the port if it's open
    if (this.port && this.port.readable) {
      this.port.close().catch(e => {
        console.error('Error closing port after error:', e);
      });
    }
    
    // Reset state
    this.port = null;
    this.reader = null;
    this.writer = null;
    this.deviceInfo = null;
    this.isStreaming = false;
  }

  /**
   * Set the connection status
   * @param status New connection status
   */
  private setStatus(status: ConnectionStatus): void {
    this.status = status;
  }
} 