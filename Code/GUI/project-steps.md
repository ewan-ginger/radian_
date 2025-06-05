# Implementation Plan
## Project Setup and Configuration
- [x] Step 1: Initialize Next.js project with TypeScript
 - **Task**: Create a new Next.js project with TypeScript support, set up basic folder structure, and configure essential dependencies
 - **Files**:
   - `package.json`: Set up dependencies including next, react, react-dom, typescript
   - `tsconfig.json`: Configure TypeScript for Next.js
   - `.gitignore`: Standard Next.js gitignore file
   - `next.config.js`: Basic Next.js configuration
   - `README.md`: Project description and setup instructions
 - **Step Dependencies**: None
 - **User Instructions**: Run `npx create-next-app@latest radian --typescript` and follow the prompts to set up the project

- [x] Step 2: Install and configure Shadcn UI with dark mode
 - **Task**: Set up Shadcn UI, configure dark mode theme, and establish the design system with specified accent colors
 - **Files**:
   - `package.json`: Add Shadcn UI dependencies
   - `components.json`: Shadcn UI configuration
   - `tailwind.config.js`: Configure color scheme with green, blue, and red accents
   - `app/globals.css`: Set up global CSS including dark mode
   - `providers/theme-provider.tsx`: Set up theme provider component
   - `lib/utils.ts`: Utility functions for styling and theme management
 - **Step Dependencies**: Step 1
 - **User Instructions**: Follow the Shadcn UI installation instructions with `npx shadcn-ui@latest init` and add base components with `npx shadcn-ui@latest add button card toast`

- [x] Step 3: Set up Supabase integration
 - **Task**: Configure Supabase client for Next.js, set up environment variables, and create utility functions for database operations
 - **Files**:
   - `package.json`: Add Supabase dependencies
   - `.env.local.example`: Template for environment variables
   - `lib/supabase/client.ts`: Client-side Supabase configuration
   - `lib/supabase/server.ts`: Server-side Supabase configuration
   - `types/supabase.ts`: TypeScript types for Supabase data
 - **Step Dependencies**: Step 1
 - **User Instructions**: 
   1. Create a Supabase project at supabase.com
   2. Rename `.env.example` to `.env.local` so it has the following variables:
      ```
      NEXT_PUBLIC_SUPABASE_URL=your_supabase_url
      NEXT_PUBLIC_SUPABASE_ANON_KEY=your_supabase_anon_key
      ```

## Database Schema and Data Management
- [x] Step 4: Create database schema for players and session data
 - **Task**: Design and implement database tables to store player information, session data, and sensor readings
 - **Files**:
   - `lib/supabase/schema.ts`: Database schema definitions
   - `types/database.types.ts`: TypeScript types for database entities
   - `scripts/seed-database.ts`: Script to initialize database with minimal seed data
 - **Step Dependencies**: Step 3
 - **User Instructions**: Run the following SQL in the Supabase SQL Editor:
   ```sql
   -- Create players table
   CREATE TABLE players (
     id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
     name TEXT NOT NULL,
     device_id TEXT,
     created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
   );

   -- Create sessions table
   CREATE TABLE sessions (
     id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
     name TEXT,
     start_time TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
     end_time TIMESTAMP WITH TIME ZONE,
     created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
   );

   -- Create sensor_data table
   CREATE TABLE sensor_data (
     id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
     session_id UUID REFERENCES sessions(id),
     player_id UUID REFERENCES players(id),
     timestamp BIGINT NOT NULL,
     accelerometer_x REAL,
     accelerometer_y REAL,
     accelerometer_z REAL,
     gyroscope_x REAL,
     gyroscope_y REAL,
     gyroscope_z REAL,
     magnetometer_x REAL,
     magnetometer_y REAL,
     magnetometer_z REAL,
     orientation_x REAL,
     orientation_y REAL,
     orientation_z REAL,
     battery_level REAL,
     created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
   );

   -- Insert default player
   INSERT INTO players (name, device_id) VALUES ('Player 1', '1');
   ```

- [x] Step 5: Implement data service functions
 - **Task**: Create service functions to handle data operations (create, read, update) for players, sessions, and sensor data
 - **Files**:
   - `lib/services/player-service.ts`: Functions for player data management
   - `lib/services/session-service.ts`: Functions for session data management
   - `lib/services/sensor-data-service.ts`: Functions for sensor data operations
 - **Step Dependencies**: Step 4
 - **User Instructions**: None

## Web Serial API Integration
- [x] Step 6: Implement Web Serial API utilities
 - **Task**: Create utility functions to connect to ESP32 devices using Web Serial API, handle connection status, and manage serial communication
 - **Files**:
   - `lib/serial/serial-connection.ts`: Core Web Serial API functionality
   - `lib/serial/serial-protocol.ts`: Protocol definitions for communicating with ESP32 devices
   - `hooks/useSerialConnection.ts`: React hook for managing serial connections
   - `types/serial.ts`: TypeScript types for serial communication
 - **Step Dependencies**: Step 1
 - **User Instructions**: None

- [x] Step 7: Implement data packet processing
 - **Task**: Create functions to parse and process 15-value data packets from ESP32 devices, normalize timestamps, and prepare data for visualization
 - **Files**:
   - `lib/serial/packet-parser.ts`: Functions to parse binary data packets
   - `lib/data/data-processor.ts`: Process and transform data for storage and visualization
   - `lib/data/data-normalizer.ts`: Normalize timestamp data
   - `types/data-packet.ts`: TypeScript types for data packets
 - **Step Dependencies**: Step 6
 - **User Instructions**: None

## UI Components
- [x] Step 8: Create core layout and navigation components
 - **Task**: Implement the main layout structure with dark mode theme integration and responsive design
 - **Files**:
   - `app/layout.tsx`: Root layout with theme provider
   - `app/page.tsx`: Main page container
   - `components/layout/Header.tsx`: Application header
   - `components/layout/Footer.tsx`: Application footer
   - `components/ui/ThemeToggle.tsx`: Toggle for dark/light mode
 - **Step Dependencies**: Step 2
 - **User Instructions**: None

- [x] Step 9: Implement device connection interface
 - **Task**: Create UI components for connecting to ESP32 devices, showing connection status, and controlling data recording
 - **Files**:
   - `components/devices/DeviceConnection.tsx`: Main device connection component
   - `components/devices/ConnectionStatus.tsx`: Status indicator component
   - `components/devices/DeviceIllustration.tsx`: Visual representation of the device
   - `components/devices/ControlButtons.tsx`: Start/stop recording buttons
   - `components/ui/StatusPill.tsx`: Reusable status indicator pill component
 - **Step Dependencies**: Steps 6, 8
 - **User Instructions**: None

- [x] Step 10: Implement data visualization components
 - **Task**: Create components for visualizing player orientation data with auto-scrolling graphs
 - **Files**:
   - `package.json`: Add data visualization dependencies (recharts)
   - `components/data/OrientationGraph.tsx`: Graph component for orientation data
   - `components/data/DataDisplay.tsx`: Container for data visualization
   - `components/data/DataControls.tsx`: Controls for data display
   - `hooks/useDataVisualization.ts`: Hook for managing visualization state
 - **Step Dependencies**: Step 8
 - **User Instructions**: Install recharts with `npm install recharts`

- [ ] Step 11: Create player information components
 - **Task**: Implement components for displaying and editing player information
 - **Files**:
   - `components/players/PlayerCard.tsx`: Card component for player data
   - `components/players/PlayerNameEditor.tsx`: Inline editor for player name
   - `components/players/PlayerStatistics.tsx`: Component for displaying player stats
   - `hooks/usePlayerData.ts`: Hook for managing player data
 - **Step Dependencies**: Steps 5, 8
 - **User Instructions**: None

## Main Application Features
- [x] Step 12: Implement hardware connection feature
 - **Task**: Integrate the Web Serial API utilities with the UI to create a complete hardware connection feature
 - **Files**:
   - `app/page.tsx`: Update main page to include device connection section
   - `components/devices/DeviceSection.tsx`: Container for all device-related components
   - `hooks/useDeviceConnection.ts`: Comprehensive hook for device connection management
   - `context/DeviceContext.tsx`: Context for sharing device connection state
 - **Step Dependencies**: Steps 6, 9
 - **User Instructions**: None

- [x] Step 13: Implement data recording and processing
 - **Task**: Create functionality to start/stop recording sessions, process incoming data, and store it in Supabase
 - **Files**:
   - `hooks/useDataRecording.ts`: Hook for managing data recording state
   - `lib/data/session-manager.ts`: Functions for creating and managing recording sessions
   - `lib/data/data-storage.ts`: Functions for storing processed data in Supabase
   - `context/RecordingContext.tsx`: Context for sharing recording state
 - **Step Dependencies**: Steps 5, 7, 12
 - **User Instructions**: None

- [ ] Step 14: Implement player data visualization feature
 - **Task**: Integrate data processing with visualization components to display live player data
 - **Files**:
   - `components/players/PlayerDataBox.tsx`: Main container for player data visualization
   - `hooks/usePlayerVisualization.ts`: Hook for managing player visualization state
   - `lib/data/visualization-data-transformer.ts`: Transform data for visualization
   - `context/VisualizationContext.tsx`: Context for sharing visualization state
 - **Step Dependencies**: Steps 10, 11, 13
 - **User Instructions**: None

- [ ] Step 15: Implement data history and playback
 - **Task**: Create functionality to retrieve historical data from Supabase and play it back in the visualization components
 - **Files**:
   - `components/history/SessionList.tsx`: Component to list recorded sessions
   - `components/history/SessionPlayer.tsx`: Component to play back recorded sessions
   - `lib/data/history-loader.ts`: Functions to load historical data
   - `hooks/useSessionPlayback.ts`: Hook for managing session playback
 - **Step Dependencies**: Steps 5, 14
 - **User Instructions**: None

## Finalization and Optimization
- [ ] Step 16: Implement error handling and notifications
 - **Task**: Add comprehensive error handling throughout the application and implement a notification system for user feedback
 - **Files**:
   - `components/ui/Toast.tsx`: Toast notification component
   - `components/ui/ErrorDisplay.tsx`: Error display component
   - `context/NotificationContext.tsx`: Context for managing notifications
   - `lib/error-handler.ts`: Centralized error handling utilities
   - `hooks/useErrorHandling.ts`: Hook for error handling in components
 - **Step Dependencies**: Step 8
 - **User Instructions**: None

- [ ] Step 17: Add responsive design and polish UI
 - **Task**: Ensure the application is fully responsive and add final UI polish for a professional look and feel
 - **Files**:
   - `app/globals.css`: Update global styles for responsiveness
   - `components/layout/ResponsiveContainer.tsx`: Container with responsive behavior
   - `components/ui/ResponsiveGrid.tsx`: Grid layout that adapts to screen size
   - Update various component files to enhance mobile responsiveness
 - **Step Dependencies**: Steps 9, 10, 11, 14
 - **User Instructions**: None 