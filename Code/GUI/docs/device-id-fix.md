# Device ID Fix for Session Data Saving

## Issue Summary
The web app was unable to save session data after the database schema changes that replaced `player_id` with `device_id` in the `sensor_data` table.

## Root Cause
The issue was in the serial data processing pipeline. The packet parser wasn't correctly extracting the device ID from the data packet. The device ID is actually the first value in the packet (index 0).

## Changes Made

1. **Updated packet format documentation**:
   - Clarified that the first value (index 0) in the packet is the device ID
   - Updated the indices for all other values to reflect the correct positions

2. **Fixed the parseSensorData function**:
   - Modified to extract the device ID from the first position in the data array
   - Properly shifted the indices for all other values (timestamp, accelerometer, etc.)
   - Converted the device ID to a string as required by the database

3. **Added debugging logs**:
   - Enhanced `insertSensorData` and `insertSensorDataBatch` functions with additional logging
   - Improved error messages to include the data that failed to insert

## Packet Format
The correct packet format from the ESP32 device is:
```
[0]: device ID (unique identifier for the sensor device)
[1]: timestamp (ms since device boot)
[2-4]: accelerometer (x, y, z) in m/s²
[5-7]: gyroscope (x, y, z) in rad/s
[8-10]: magnetometer (x, y, z) in μT
[11-13]: orientation (x, y, z) in degrees
[14]: battery level (percentage)
```

## Testing
To verify this fix:

1. Connect a sensor device to the app
2. Start a new recording session
3. Ensure data is being saved to the database
4. Check the browser console for logs showing the device ID
5. Check the Supabase console to confirm data is being inserted with the correct device ID

## Future Improvements
In the future, we should implement a more robust way to handle device IDs:

1. Modify the firmware to include the device ID in the packet data
2. Update the packet parser to extract the device ID from the packet
3. Alternatively, assign device IDs dynamically based on the serial port or connection

## Related Files
- `radian-app/lib/serial/serial-protocol.ts`
- `radian-app/lib/serial/packet-parser.ts`
- `radian-app/lib/services/sensor-data-service.ts` 