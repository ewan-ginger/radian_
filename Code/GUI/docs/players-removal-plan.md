# Players Table Removal Plan

This document outlines the steps taken to remove the unused `players` table and clarify the usage of `device_id` in the application.

## Summary of Changes

1. Dropped the `players` table from the database
2. Updated sensor data services to use `device_id` instead of `player_id`
3. Updated session services to work with the new data model
4. Added documentation about `device_id` not being a foreign key

## Database Changes

- Dropped the `players` table
- Kept `player_profiles` table which is still in use
- Changed references in `sensor_data` from `player_id` to `device_id`
- The `device_id` in `sensor_data` is a raw string identifier from the sensor device, not a foreign key

## Code Changes

### Files Deleted
- `radian-app/lib/services/player-service.ts` - Removed service for unused table

### Files Updated
- `radian-app/lib/services/sensor-data-service.ts`
  - Changed all functions to use `device_id` instead of `player_id`
  - Added documentation clarifying `device_id` is not a foreign key
  - Renamed functions to reflect the use of devices instead of players
  - Updated type assertions for safer property access

- `radian-app/lib/services/session-service.ts`
  - Removed references to the `PLAYERS_TABLE`
  - Updated `getSessionSummary` to use devices instead of players
  - Now links devices to players through `session_players` table

- `radian-app/types/database.types.ts`
  - Updated `SessionSummary` interface to use `devices` instead of `players`
  - Added `playerName` property to `SensorDataEntity` for convenience

## Device ID Clarification

The `device_id` in the `sensor_data` table:
1. Is a raw string identifier coming directly from the sensor device
2. Does not reference any other table as a foreign key
3. Is used to group sensor data by the physical device that generated it
4. Can be linked to a player through the `session_players` junction table

This approach provides better separation of concerns:
- Device information is kept separate from player profile information
- One player can use different devices across different sessions
- One device can be used by different players across different sessions
- Session data can be analyzed by device ID regardless of player assignment 