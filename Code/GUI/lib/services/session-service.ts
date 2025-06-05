import { supabaseClient } from '@/lib/supabase/client';
import { createServerSupabaseClient } from '@/lib/supabase/server';
import { SESSIONS_TABLE, SENSOR_DATA_TABLE, SESSION_PLAYERS_TABLE } from '@/lib/supabase/schema';
import { Session, SessionInsert, SessionUpdate, SessionPlayerInsert } from '@/types/supabase';
import { SessionEntity, SessionSummary, SessionType } from '@/types/database.types';
import { addSessionPlayer } from './session-player-service';

/**
 * Get all sessions
 * @returns Array of sessions
 */
export async function getAllSessions(): Promise<SessionEntity[]> {
  try {
    // First get all sessions
    const { data: sessions, error } = await supabaseClient
      .from(SESSIONS_TABLE)
      .select('*')
      .order('start_time', { ascending: false });

    if (error) {
      console.error('Error fetching sessions:', error);
      throw new Error(`Failed to fetch sessions: ${error.message}`);
    }

    // Then get all session players with player names
    const { data: sessionPlayers, error: playersError } = await supabaseClient
      .from(SESSION_PLAYERS_TABLE)
      .select(`
        session_id,
        player_id,
        device_id,
        player_profiles ( name ) 
      `);

    if (playersError) {
      console.error('Error fetching session players:', playersError);
      // Continue with sessions but without player data
    }

    // Create a map of session_id to players
    const sessionPlayersMap = new Map<string, { player_id: string, name: string }[]>();
    
    if (sessionPlayers) {
      sessionPlayers.forEach(sp => {
        if (!sessionPlayersMap.has(sp.session_id)) {
          sessionPlayersMap.set(sp.session_id, []);
        }
        
        // Use type assertion to bypass potential inference issues
        const profile = sp.player_profiles as any; 
        if (profile?.name) {
          sessionPlayersMap.get(sp.session_id)?.push({
            player_id: sp.player_id,
            name: profile.name
          });
        }
      });
    }

    // Add player data to sessions
    const sessionsWithPlayers = sessions.map(session => {
      const players = sessionPlayersMap.get(session.id) || [];
      return {
        ...session,
        players: players.map(p => ({
          player_id: p.player_id,
          playerName: p.name
        }))
      };
    });

    return sessionsWithPlayers;
  } catch (error) {
    console.error('Error in getAllSessions:', error);
    throw error;
  }
}

/**
 * Get a session by ID
 * @param id Session ID
 * @returns Session or null if not found
 */
export async function getSessionById(id: string): Promise<SessionEntity | null> {
  const { data, error } = await supabaseClient
    .from(SESSIONS_TABLE)
    .select('*')
    .eq('id', id)
    .single();

  if (error) {
    if (error.code === 'PGRST116') {
      // PGRST116 is the error code for "no rows returned"
      return null;
    }
    console.error(`Error fetching session with ID ${id}:`, error);
    throw new Error(`Failed to fetch session: ${error.message}`);
  }

  return data;
}

/**
 * Check if a session name already exists
 * @param name Session name to check
 * @returns True if the name exists, false otherwise
 */
export async function checkSessionNameExists(name: string): Promise<boolean> {
  try {
    console.log(`Checking if session name exists: "${name}"`);
    
    // Get all sessions with this exact name
    const { data, error } = await supabaseClient
      .from(SESSIONS_TABLE)
      .select('id')
      .eq('name', name.trim());

    if (error) {
      console.error('Error checking session name:', error);
      throw new Error(`Failed to check session name: ${error.message}`);
    }

    console.log(`Found ${data?.length || 0} sessions with name "${name}"`);
    
    // Check if any sessions were found with this exact name
    return Array.isArray(data) && data.length > 0;
  } catch (error) {
    console.error('Error checking session name:', error);
    throw error;
  }
}

/**
 * Create a new session
 * @param session Session data to insert
 * @param skipNameCheck If true, skips checking if the session name already exists
 * @returns The created session
 */
export async function createSession(session: SessionInsert, skipNameCheck = false): Promise<Session> {
  // Check if name already exists
  if (session.name && !skipNameCheck) {
    const exists = await checkSessionNameExists(session.name);
    if (exists) {
      throw new Error(`Session name "${session.name}" already exists`);
    }
  }

  const { data, error } = await supabaseClient
    .from(SESSIONS_TABLE)
    .insert(session)
    .select()
    .single();

  if (error) {
    console.error('Error creating session:', error);
    throw new Error(`Failed to create session: ${error.message}`);
  }

  return data;
}

/**
 * Update a session
 * @param id Session ID
 * @param updates Session data to update
 * @returns The updated session
 */
export async function updateSession(id: string, updates: SessionUpdate): Promise<Session> {
  const { data, error } = await supabaseClient
    .from(SESSIONS_TABLE)
    .update(updates)
    .eq('id', id)
    .select()
    .single();

  if (error) {
    console.error(`Error updating session with ID ${id}:`, error);
    throw new Error(`Failed to update session: ${error.message}`);
  }

  return data;
}

/**
 * End a session by setting its end_time to the provided time and calculating duration
 * @param id Session ID
 * @param intendedEndTime The Date object representing when the session should end.
 * @returns The updated session
 */
export async function endSession(id: string, intendedEndTime: Date): Promise<Session> {
  // Get the session to calculate duration
  const session = await getSessionById(id);
  if (!session) {
    throw new Error('Session not found');
  }

  // Use the provided end time
  const endTime = intendedEndTime;
  const startTime = new Date(session.start_time);
  
  // Calculate duration in seconds
  const durationSeconds = Math.max(0, Math.floor((endTime.getTime() - startTime.getTime()) / 1000));
  
  console.log('Ending session (service):', {
    id,
    startTime: startTime.toISOString(),
    endTime: endTime.toISOString(),
    durationSeconds
  });
  
  // Use raw SQL to update both end_time and duration
  // Ensure the function `update_session_duration` uses the provided end time
  // If the RPC function doesn't accept end_time, we might need to modify it
  // or fall back to a simple update.
  const { data, error } = await supabaseClient.rpc('update_session_duration', {
    session_id: id,
    end_time_param: endTime.toISOString(), // Assuming RPC accepts end_time
    duration_seconds: durationSeconds
  });

  if (error) {
    console.error(`Error updating session via RPC for ID ${id}:`, { error, endTime: endTime.toISOString(), durationSeconds });
    
    // Fall back to regular update using the intendedEndTime
    console.log('Falling back to regular update with intended end time');
    return updateSession(id, { 
      end_time: endTime.toISOString()
      // Note: Duration won't be set in this fallback unless updateSession also handles it
    });
  }

  console.log('Session updated successfully via RPC with duration');
  
  // Fetch the updated session
  const updatedSession = await getSessionById(id);
  if (!updatedSession) {
    throw new Error('Failed to fetch updated session');
  }

  return updatedSession as Session;
}

async function deleteInBatches(tableName: string, sessionId: string, batchSize: number = 100): Promise<Error | null> {
  console.log(`Starting batch delete for ${tableName}, session ${sessionId}`);
  let deletedCount = 0;
  while (true) {
    try {
      // Find a batch of primary keys to delete
      const { data: idsToDelete, error: selectError } = await supabaseClient
        .from(tableName)
        .select('id') // Select only the primary key
        .eq('session_id', sessionId)
        .limit(batchSize);

      if (selectError) {
        console.error(`Error selecting batch from ${tableName} for session ${sessionId}:`, selectError);
        return new Error(`Failed to select batch from ${tableName}: ${selectError.message}`);
      }

      // If no more rows found for this session, we're done
      if (!idsToDelete || idsToDelete.length === 0) {
        console.log(`Finished batch delete for ${tableName}, session ${sessionId}. Total deleted: ${deletedCount}`);
        return null; // Success
      }

      const ids = idsToDelete.map(row => row.id);
      
      // Delete the batch by primary key
      const { error: deleteError } = await supabaseClient
        .from(tableName)
        .delete()
        .in('id', ids);
        
      if (deleteError) {
        console.error(`Error deleting batch from ${tableName} for session ${sessionId}:`, deleteError);
        return new Error(`Failed to delete batch from ${tableName}: ${deleteError.message}`);
      }

      deletedCount += ids.length;
      console.log(`Deleted batch of ${ids.length} from ${tableName}. Total deleted so far: ${deletedCount}`);

    } catch (batchError) {
      console.error(`Unexpected error during batch delete for ${tableName}, session ${sessionId}:`, batchError);
      return batchError instanceof Error ? batchError : new Error('Unknown error during batch delete');
    }
  }
}

/**
 * Delete a session
 * @param id Session ID
 * @returns True if successful
 */
export async function deleteSession(id: string): Promise<boolean> {
  try {
    // Batch delete training_sensor_data records (use batch size 100)
    const trainingDataError = await deleteInBatches('training_sensor_data', id, 100);
    if (trainingDataError) {
      // Decide if this error is critical. If so, throw.
      console.error(`Failed to fully delete training data for session ${id}:`, trainingDataError.message);
      throw trainingDataError; // Or handle differently, e.g., log and continue
    }

    // Batch delete sensor_data records (use batch size 100)
    const sensorDataError = await deleteInBatches('sensor_data', id, 100);
    if (sensorDataError) {
      // Sensor data deletion failure is critical as per original logic
      console.error(`Failed to fully delete sensor data for session ${id}:`, sensorDataError.message);
      throw sensorDataError;
    }

    // Delete session player mappings (usually not large, batching likely overkill)
    const { error: sessionPlayersError } = await supabaseClient
      .from(SESSION_PLAYERS_TABLE)
      .delete()
      .eq('session_id', id);

    if (sessionPlayersError) {
      console.error(`Error deleting session players for session ${id}:`, sessionPlayersError);
      // Decide if this error is critical. Let's throw for safety.
      throw new Error(`Failed to delete session players: ${sessionPlayersError.message}`);
    }

    // Delete the session itself
    const { error: sessionError } = await supabaseClient
      .from(SESSIONS_TABLE)
      .delete()
      .eq('id', id);

    if (sessionError) {
      console.error(`Error deleting session record ${id}:`, sessionError);
      throw new Error(`Failed to delete session record: ${sessionError.message}`);
    }

    console.log(`Successfully deleted session ${id} and all related data.`);
    return true;

  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    console.error(`Error in deleteSession process for ${id}: ${errorMessage}`, error);
    throw new Error(`Failed to delete session ${id}: ${errorMessage}`);
  }
}

/**
 * Get session summary with device information and data point counts
 * @param sessionId Session ID
 * @returns Session summary or null if not found
 */
export async function getSessionSummary(sessionId: string): Promise<SessionSummary | null> {
  // First, get the session
  const session = await getSessionById(sessionId);

  if (!session) {
    return null;
  }

  // Get data points count directly
  const { count, error: countError } = await supabaseClient
    .from(SENSOR_DATA_TABLE)
    .select('*' , { count: 'exact', head: true }) // Use count and head:true for efficiency
    .eq('session_id', sessionId);

  if (countError) {
    console.error(`Error counting sensor data for session ${sessionId}:`, countError);
    throw new Error(`Failed to count sensor data: ${countError.message}`);
  }

  const dataPointsCount = count ?? 0;

  // Get session players information to list players involved
  const { data: sessionPlayers, error: sessionPlayersError } = await supabaseClient
    .from(SESSION_PLAYERS_TABLE)
    .select(`
      player_id,
      device_id,
      player_profiles ( name )
    `)
    .eq('session_id', sessionId);

  if (sessionPlayersError) {
    console.error(`Error fetching session players for session ${sessionId}:`, sessionPlayersError);
    // Continue without player info if needed, or handle error as appropriate
  }

  // Extract unique player names
  const playerNames: string[] = [];
  const seenPlayerIds = new Set<string>();
  sessionPlayers?.forEach(sp => {
    // Ensure player_profiles and name exist, and player hasn't been added yet
    // Use type assertion to bypass potential inference issues
    const profile = sp.player_profiles as any;
    if (sp.player_id && profile?.name && !seenPlayerIds.has(sp.player_id)) {
        playerNames.push(profile.name);
        seenPlayerIds.add(sp.player_id); 
    }
  });

  // Calculate duration
  const startTime = new Date(session.start_time);
  const endTime = session.end_time ? new Date(session.end_time) : new Date();
  const duration = Math.round((endTime.getTime() - startTime.getTime()) / 1000); // in seconds

  return {
    sessionId: session.id,
    sessionName: session.name || `Session ${session.id.substring(0, 8)}`,
    startTime,
    endTime: session.end_time ? new Date(session.end_time) : undefined,
    duration,
    sessionType: session.session_type,
    devices: [], // Add empty array to satisfy the type for now
    players: playerNames,
    dataPointsCount: dataPointsCount, // Use the accurate count
  };
}

/**
 * Server-side function to get all sessions
 * @returns Array of sessions
 */
export async function getAllSessionsServer(): Promise<SessionEntity[]> {
  const supabase = await createServerSupabaseClient();
  
  const { data, error } = await supabase
    .from(SESSIONS_TABLE)
    .select('*')
    .order('start_time', { ascending: false });

  if (error) {
    console.error('Error fetching sessions on server:', error);
    throw new Error(`Failed to fetch sessions on server: ${error.message}`);
  }

  // Cast via unknown first as suggested by linter
  return (data || []) as unknown as SessionEntity[];
}

/**
 * Create a new session with player-device mapping
 * @param sessionData Session data to insert
 * @param playerId Player profile ID to link to this session
 * @param deviceId Device ID to use for this player
 * @param skipNameCheck If true, skips checking if the session name already exists
 * @returns The created session
 */
export async function createSessionWithPlayerDevice(
  sessionData: {
    name?: string | null;
    session_type: SessionType;
  },
  playerId: string,
  deviceId: string,
  skipNameCheck = false
): Promise<Session> {
  // Start a transaction to create both the session and player mapping
  try {
    // 1. Create the session
    const sessionInsert: SessionInsert = {
      name: sessionData.name,
      start_time: new Date().toISOString(),
      session_type: sessionData.session_type
    };
    
    console.log('Creating session with data:', sessionInsert);
    const session = await createSession(sessionInsert, skipNameCheck);
    
    // 2. Create the player-device mapping
    const sessionPlayerInsert: SessionPlayerInsert = {
      session_id: session.id,
      player_id: playerId,
      device_id: deviceId
    };
    
    console.log('Creating session-player mapping:', sessionPlayerInsert);
    await addSessionPlayer(sessionPlayerInsert);
    
    return session;
  } catch (error) {
    console.error('Error creating session with player-device mapping:', error);
    throw new Error(`Failed to create session with player-device mapping: ${error instanceof Error ? error.message : String(error)}`);
  }
} 