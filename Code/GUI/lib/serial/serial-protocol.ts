import { CommandType, MessageType, SerialMessage, CommandMessage, DataMessage } from '@/types/serial';

/**
 * Serial protocol for communicating with ESP32 devices
 * 
 * This protocol defines how messages are encoded and decoded when communicating
 * with ESP32 devices over the Web Serial API.
 * 
 * Message Format:
 * - Start marker: '$'
 * - Message type: 1 byte (C: Command, D: Data, A: Ack, E: Error)
 * - Payload length: 2 bytes (little-endian)
 * - Payload: Variable length
 * - Checksum: 1 byte (XOR of all payload bytes)
 * - End marker: '#'
 */

// Protocol constants
export const PROTOCOL = {
  START_MARKER: 0x24, // '$'
  END_MARKER: 0x23, // '#'
  TYPE_COMMAND: 0x43, // 'C'
  TYPE_DATA: 0x44, // 'D'
  TYPE_ACK: 0x41, // 'A'
  TYPE_ERROR: 0x45, // 'E'
  HEADER_SIZE: 4, // Start marker (1) + Type (1) + Length (2)
  FOOTER_SIZE: 2, // Checksum (1) + End marker (1)
  MAX_PAYLOAD_SIZE: 1024,
};

/**
 * Calculate checksum for a message payload
 * @param payload Uint8Array containing the message payload
 * @returns Checksum byte (XOR of all payload bytes)
 */
export function calculateChecksum(payload: Uint8Array): number {
  let checksum = 0;
  for (let i = 0; i < payload.length; i++) {
    checksum ^= payload[i];
  }
  return checksum;
}

/**
 * Encode a message to be sent to the ESP32 device
 * @param message Message to encode
 * @returns Uint8Array containing the encoded message
 */
export function encodeMessage(message: SerialMessage): Uint8Array {
  // Convert message to JSON string
  const jsonString = JSON.stringify(message);
  
  // Convert JSON string to Uint8Array
  const encoder = new TextEncoder();
  const payload = encoder.encode(jsonString);
  
  if (payload.length > PROTOCOL.MAX_PAYLOAD_SIZE) {
    throw new Error(`Message payload too large: ${payload.length} bytes (max ${PROTOCOL.MAX_PAYLOAD_SIZE})`);
  }
  
  // Calculate payload length (2 bytes, little-endian)
  const length = payload.length;
  const lengthBytes = new Uint8Array(2);
  lengthBytes[0] = length & 0xFF; // Low byte
  lengthBytes[1] = (length >> 8) & 0xFF; // High byte
  
  // Determine message type byte
  let typeByte: number;
  switch (message.type) {
    case MessageType.COMMAND:
      typeByte = PROTOCOL.TYPE_COMMAND;
      break;
    case MessageType.DATA:
      typeByte = PROTOCOL.TYPE_DATA;
      break;
    case MessageType.ACK:
      typeByte = PROTOCOL.TYPE_ACK;
      break;
    case MessageType.ERROR:
      typeByte = PROTOCOL.TYPE_ERROR;
      break;
    default:
      throw new Error(`Unknown message type: ${message.type}`);
  }
  
  // Calculate checksum
  const checksum = calculateChecksum(payload);
  
  // Create the complete message
  const messageSize = PROTOCOL.HEADER_SIZE + payload.length + PROTOCOL.FOOTER_SIZE;
  const buffer = new Uint8Array(messageSize);
  
  // Header
  buffer[0] = PROTOCOL.START_MARKER;
  buffer[1] = typeByte;
  buffer[2] = lengthBytes[0];
  buffer[3] = lengthBytes[1];
  
  // Payload
  buffer.set(payload, PROTOCOL.HEADER_SIZE);
  
  // Footer
  buffer[PROTOCOL.HEADER_SIZE + payload.length] = checksum;
  buffer[PROTOCOL.HEADER_SIZE + payload.length + 1] = PROTOCOL.END_MARKER;
  
  return buffer;
}

/**
 * Decode a message received from the ESP32 device
 * @param buffer Uint8Array containing the encoded message
 * @returns Decoded message or null if the message is invalid
 */
export function decodeMessage(buffer: Uint8Array): SerialMessage | null {
  // Check if the buffer is long enough to contain a valid message
  if (buffer.length < PROTOCOL.HEADER_SIZE + PROTOCOL.FOOTER_SIZE) {
    return null;
  }
  
  // Check start and end markers
  if (buffer[0] !== PROTOCOL.START_MARKER || buffer[buffer.length - 1] !== PROTOCOL.END_MARKER) {
    return null;
  }
  
  // Get message type
  const typeByte = buffer[1];
  let messageType: MessageType;
  
  switch (typeByte) {
    case PROTOCOL.TYPE_COMMAND:
      messageType = MessageType.COMMAND;
      break;
    case PROTOCOL.TYPE_DATA:
      messageType = MessageType.DATA;
      break;
    case PROTOCOL.TYPE_ACK:
      messageType = MessageType.ACK;
      break;
    case PROTOCOL.TYPE_ERROR:
      messageType = MessageType.ERROR;
      break;
    default:
      return null; // Unknown message type
  }
  
  // Get payload length
  const lengthLow = buffer[2];
  const lengthHigh = buffer[3];
  const payloadLength = lengthLow | (lengthHigh << 8);
  
  // Check if the buffer contains the complete message
  const expectedLength = PROTOCOL.HEADER_SIZE + payloadLength + PROTOCOL.FOOTER_SIZE;
  if (buffer.length !== expectedLength) {
    return null;
  }
  
  // Extract payload
  const payload = buffer.slice(PROTOCOL.HEADER_SIZE, PROTOCOL.HEADER_SIZE + payloadLength);
  
  // Verify checksum
  const expectedChecksum = calculateChecksum(payload);
  const actualChecksum = buffer[PROTOCOL.HEADER_SIZE + payloadLength];
  
  if (expectedChecksum !== actualChecksum) {
    return null; // Checksum mismatch
  }
  
  // Decode payload from JSON
  try {
    const decoder = new TextDecoder();
    const jsonString = decoder.decode(payload);
    const message = JSON.parse(jsonString) as SerialMessage;
    
    // Ensure the message has the correct type
    message.type = messageType;
    
    return message;
  } catch (error) {
    console.error('Error decoding message:', error);
    return null;
  }
}

/**
 * Create a command message to send to the ESP32 device
 * @param command Command type
 * @param payload Command payload
 * @returns Command message
 */
export function createCommandMessage(command: CommandType, payload: any = {}): CommandMessage {
  return {
    type: MessageType.COMMAND,
    timestamp: Date.now(),
    command,
    payload,
  };
}

/**
 * Create a data message from sensor values
 * @param values Array of sensor values
 * @returns Data message
 */
export function createDataMessage(values: number[]): DataMessage {
  return {
    type: MessageType.DATA,
    timestamp: Date.now(),
    payload: values,
  };
}

/**
 * Parse a data message into sensor values
 * @param dataMessage Data message
 * @returns Object containing sensor values
 */
export function parseSensorData(dataMessage: DataMessage): Record<string, any> {
  const values = dataMessage.payload;
  
  // Expected format: [device_id, timestamp, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, mag_x, mag_y, mag_z, orient_x, orient_y, orient_z, battery]
  if (values.length !== 15) {
    throw new Error(`Invalid sensor data: expected 15 values, got ${values.length}`);
  }
  
  // Convert the device ID to a string as required by the database
  const deviceId = values[0].toString();
  
  return {
    deviceID: deviceId,
    timestamp: values[1],
    accelerometer_x: values[2],
    accelerometer_y: values[3],
    accelerometer_z: values[4],
    gyroscope_x: values[5],
    gyroscope_y: values[6],
    gyroscope_z: values[7],
    magnetometer_x: values[8],
    magnetometer_y: values[9],
    magnetometer_z: values[10],
    orientation_x: values[11],
    orientation_y: values[12],
    orientation_z: values[13],
    battery_level: values[14]
  };
} 