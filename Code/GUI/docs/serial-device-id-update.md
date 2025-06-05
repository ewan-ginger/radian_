# Serial Data Processing Update

This document outlines the changes needed to update the serial data processing to use `deviceID` instead of `playerID`.

## Background

The application receives sensor data over serial connections. Previously, this data contained a field identified as `playerID`. As part of our database restructuring, we have clarified that this is actually a device identifier and not directly linked to a player profile.

## Changes Required

1. In the serial packet parser (`radian-app/lib/serial/packet-parser.ts`), update all references from `playerID` to `deviceID`.

2. When processing the data packets, treat the identifier as a raw string from the device and not as a foreign key.

3. Update any processing code that looks up player information directly from the ID to instead use the `session_players` junction table.

## Implementation Details

### Packet Structure

The serial data packet structure should be updated in documentation and comments:

```
Data packet format:
- deviceID: String identifier for the physical sensor device
- timestamp: Milliseconds since epoch
- sensorData: Array of values for different sensors
  * accelerometer (x, y, z) in m/s²
  * gyroscope (x, y, z) in rad/s
  * magnetometer (x, y, z) in μT
  * orientation (x, y, z) as quaternion
  * battery_level (percentage)
```

### Data Processing

When receiving data from the serial port:

1. Extract the `deviceID` from the incoming packet
2. Use the `deviceID` to associate the data with the current session
3. Store the data in the `sensor_data` table with the `device_id` field set to this value
4. If player information is needed for display purposes, look it up using the `session_players` junction table

## Testing

After making these changes, test the serial communication to ensure:

1. Data is correctly received with the `deviceID` field
2. Data is properly stored in the database with the `device_id` field
3. Player association works correctly through the `session_players` table
4. Live data visualization correctly identifies and separates data by device 