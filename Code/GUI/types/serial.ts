/**
 * TypeScript types for Web Serial API communication
 */

// Connection status enum
export enum ConnectionStatus {
  DISCONNECTED = 'disconnected',
  CONNECTING = 'connecting',
  CONNECTED = 'connected',
  ERROR = 'error',
}

// Serial port configuration
export interface SerialPortConfig {
  baudRate: number;
  dataBits?: number;
  stopBits?: number;
  parity?: 'none' | 'even' | 'odd';
  bufferSize?: number;
  flowControl?: 'none' | 'hardware';
}

// Default serial port configuration for ESP32 devices
export const DEFAULT_ESP32_CONFIG: SerialPortConfig = {
  baudRate: 115200,
  dataBits: 8,
  stopBits: 1,
  parity: 'none',
  bufferSize: 4096,
  flowControl: 'none',
};

// Serial connection options
export interface SerialConnectionOptions {
  config?: SerialPortConfig;
  autoConnect?: boolean;
  filters?: SerialPortFilter[];
}

// Serial port filter for device selection
export interface SerialPortFilter {
  usbVendorId?: number;
  usbProductId?: number;
}

// Common ESP32 USB vendor and product IDs
export const ESP32_FILTERS: SerialPortFilter[] = [
  // Silicon Labs CP210x USB to UART Bridge (common in ESP32 dev boards)
  { usbVendorId: 0x10C4, usbProductId: 0xEA60 },
  // FTDI FT232 USB to UART Bridge
  { usbVendorId: 0x0403, usbProductId: 0x6001 },
  // CH340 USB to UART Bridge
  { usbVendorId: 0x1A86, usbProductId: 0x7523 },
];

// Serial message types
export enum MessageType {
  COMMAND = 'command',
  DATA = 'data',
  ACK = 'ack',
  ERROR = 'error',
}

// Serial command types
export enum CommandType {
  START_STREAMING = 'start_streaming',
  STOP_STREAMING = 'stop_streaming',
  SET_SAMPLE_RATE = 'set_sample_rate',
  GET_DEVICE_INFO = 'get_device_info',
  RESET = 'reset',
}

// Serial message interface
export interface SerialMessage {
  type: MessageType;
  timestamp: number;
  payload: any;
}

// Serial command message
export interface CommandMessage extends SerialMessage {
  type: MessageType.COMMAND;
  command: CommandType;
  payload: any;
}

// Serial data message
export interface DataMessage extends SerialMessage {
  type: MessageType.DATA;
  payload: number[];
}

// Device information
export interface DeviceInfo {
  deviceId: string;
  firmwareVersion: string;
  batteryLevel: number;
  sampleRate: number;
}

// Serial event types
export enum SerialEventType {
  CONNECT = 'connect',
  DISCONNECT = 'disconnect',
  DATA_RECEIVED = 'data_received',
  ERROR = 'error',
}

// Serial event interface
export interface SerialEvent {
  type: SerialEventType;
  timestamp: number;
  data?: any;
}

// Serial connection state
export interface SerialConnectionState {
  port: SerialPort | null;
  reader: ReadableStreamDefaultReader<Uint8Array> | null;
  writer: WritableStreamDefaultWriter<Uint8Array> | null;
  status: ConnectionStatus;
  error: Error | null;
  deviceInfo: DeviceInfo | null;
  isStreaming: boolean;
} 