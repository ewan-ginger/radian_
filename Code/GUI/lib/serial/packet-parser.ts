'use client';

import { DataMessage } from '@/types/serial';
import { SensorDataInsert } from '@/types/supabase';
import { createDataMessage, parseSensorData } from './serial-protocol';

/**
 * Packet format for ESP32 sensor data
 * 
 * Each packet contains 15 values:
 * 0: device ID (unique identifier for the sensor device)
 * 1: timestamp (ms since device boot)
 * 2-4: accelerometer (x, y, z) in m/s²
 * 5-7: gyroscope (x, y, z) in rad/s
 * 8-10: magnetometer (x, y, z) in μT
 * 11-13: orientation (x, y, z) in degrees
 * 14: battery level (percentage)
 */

// Constants for packet parsing
export const PACKET_SIZE = 15;
export const PACKET_HEADER = 0xAA;
export const PACKET_FOOTER = 0x55;

/**
 * Parse a binary data packet from an ESP32 device
 * @param buffer Binary data buffer
 * @returns Parsed data message or null if the packet is invalid
 */
export function parseDataPacket(buffer: Uint8Array): DataMessage | null {
  // Check if the buffer is long enough to contain a valid packet
  if (buffer.length < PACKET_SIZE * 4 + 2) {
    return null; // Not enough data
  }
  
  // Check header and footer
  if (buffer[0] !== PACKET_HEADER || buffer[buffer.length - 1] !== PACKET_FOOTER) {
    return null; // Invalid packet
  }
  
  // Extract the 15 float values (4 bytes each)
  const values: number[] = [];
  
  for (let i = 0; i < PACKET_SIZE; i++) {
    const offset = 1 + i * 4; // Skip header byte
    const view = new DataView(buffer.buffer, buffer.byteOffset + offset, 4);
    const value = view.getFloat32(0, true); // Little-endian
    values.push(value);
  }
  
  // Create a data message
  return createDataMessage(values);
}

/**
 * Convert a data message to a sensor data insert object
 * @param message Data message
 * @param sessionId Session ID
 * @returns Sensor data insert object
 */
export function dataMessageToSensorData(
  message: DataMessage,
  sessionId: string,
): SensorDataInsert {
  const sensorData = parseSensorData(message);
  
  return {
    session_id: sessionId,
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
}

/**
 * Check if a buffer contains a complete packet
 * @param buffer Binary data buffer
 * @returns True if the buffer contains a complete packet
 */
export function isCompletePacket(buffer: Uint8Array): boolean {
  // Check if the buffer is long enough to contain a valid packet
  if (buffer.length < PACKET_SIZE * 4 + 2) {
    return false; // Not enough data
  }
  
  // Check header and footer
  return buffer[0] === PACKET_HEADER && buffer[buffer.length - 1] === PACKET_FOOTER;
}

/**
 * Find the start of the next packet in a buffer
 * @param buffer Binary data buffer
 * @param startIndex Index to start searching from
 * @returns Index of the next packet header or -1 if not found
 */
export function findNextPacketStart(buffer: Uint8Array, startIndex: number = 0): number {
  for (let i = startIndex; i < buffer.length; i++) {
    if (buffer[i] === PACKET_HEADER) {
      return i;
    }
  }
  
  return -1; // Not found
}

/**
 * Extract all complete packets from a buffer
 * @param buffer Binary data buffer
 * @returns Array of data messages and the remaining buffer
 */
export function extractPackets(buffer: Uint8Array): {
  messages: DataMessage[];
  remainingBuffer: Uint8Array;
} {
  const messages: DataMessage[] = [];
  let currentIndex = 0;
  
  while (currentIndex < buffer.length) {
    // Find the start of the next packet
    const packetStart = findNextPacketStart(buffer, currentIndex);
    
    if (packetStart === -1) {
      // No more packets, return the remaining buffer
      return {
        messages,
        remainingBuffer: buffer.slice(currentIndex),
      };
    }
    
    // Check if there's enough data for a complete packet
    const packetEnd = packetStart + PACKET_SIZE * 4 + 1; // Header + 15 floats + footer
    
    if (packetEnd >= buffer.length) {
      // Not enough data, return the remaining buffer
      return {
        messages,
        remainingBuffer: buffer.slice(packetStart),
      };
    }
    
    // Extract the packet
    const packet = buffer.slice(packetStart, packetEnd + 1);
    const message = parseDataPacket(packet);
    
    if (message) {
      messages.push(message);
    }
    
    // Move to the next packet
    currentIndex = packetEnd + 1;
  }
  
  // No more data
  return {
    messages,
    remainingBuffer: new Uint8Array(0),
  };
} 