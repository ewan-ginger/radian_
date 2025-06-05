/**
 * TypeScript types for data packets from ESP32 devices
 */

/**
 * Raw data packet from ESP32 device
 * 
 * Each packet contains 15 values:
 * 0: timestamp (ms since device boot)
 * 1-3: accelerometer (x, y, z) in m/s²
 * 4-6: gyroscope (x, y, z) in rad/s
 * 7-9: magnetometer (x, y, z) in μT
 * 10-12: orientation (x, y, z) in degrees
 * 13: battery level (percentage)
 * 14: reserved for future use
 */
export interface RawDataPacket {
  timestamp: number;
  accelerometer: [number, number, number]; // x, y, z
  gyroscope: [number, number, number]; // x, y, z
  magnetometer: [number, number, number]; // x, y, z
  orientation: [number, number, number]; // x, y, z
  batteryLevel: number;
  reserved: number;
}

/**
 * Processed data packet with derived values
 */
export interface ProcessedDataPacket extends RawDataPacket {
  // Derived values
  acceleration: number; // Magnitude of acceleration vector
  rotationRate: number; // Magnitude of rotation vector
  magneticField: number; // Magnitude of magnetic field vector
}

/**
 * Data packet batch for efficient storage
 */
export interface DataPacketBatch {
  sessionId: string;
  playerId: string;
  startTime: number;
  endTime: number;
  packets: RawDataPacket[];
}

/**
 * Data packet for visualization
 */
export interface VisualizationDataPacket {
  timestamp: number;
  values: Record<string, number>;
  orientation: {
    x: number;
    y: number;
    z: number;
  };
}

/**
 * Data packet statistics
 */
export interface DataPacketStatistics {
  count: number;
  duration: number; // in milliseconds
  sampleRate: number; // in Hz
  acceleration: {
    min: number;
    max: number;
    avg: number;
  };
  rotationRate: {
    min: number;
    max: number;
    avg: number;
  };
  batteryLevel: {
    min: number;
    max: number;
    avg: number;
  };
}

/**
 * Data packet filter options
 */
export interface DataPacketFilter {
  startTime?: number;
  endTime?: number;
  minAcceleration?: number;
  maxAcceleration?: number;
  minRotationRate?: number;
  maxRotationRate?: number;
  minBatteryLevel?: number;
  maxBatteryLevel?: number;
} 