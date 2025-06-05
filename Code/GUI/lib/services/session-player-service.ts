import { supabaseClient } from '@/lib/supabase/client';
import { createServerSupabaseClient } from '@/lib/supabase/server';
import { SESSION_PLAYERS_TABLE, PLAYERS_TABLE } from '@/lib/supabase/schema';
import { SessionPlayerEntity } from '@/types/database.types';
import { SessionPlayer, SessionPlayerInsert, SessionPlayerUpdate } from '@/types/supabase';

/**
 * Get all session players for a session
 * @param sessionId Session ID
 * @returns Array of session players with device assignments
 */
export async function getSessionPlayersBySessionId(sessionId: string): Promise<SessionPlayerEntity[]> {
  const { data, error } = await supabaseClient
    .from(SESSION_PLAYERS_TABLE)
    .select(`
      *,
      player:${PLAYERS_TABLE}(id, name)
    `)
    .eq('session_id', sessionId);

  if (error) {
    console.error(`Error fetching session players for session ${sessionId}:`, error);
    throw new Error(`Failed to fetch session players: ${error.message}`);
  }

  // Transform the data to add convenience properties
  return data.map(item => ({
    ...item,
    playerName: item.player?.name || 'Unknown Player'
  }));
}

/**
 * Add a player-device assignment to a session
 * @param sessionPlayer Session player data to insert
 * @returns The created session player assignment
 */
export async function addSessionPlayer(sessionPlayer: SessionPlayerInsert): Promise<SessionPlayer> {
  const { data, error } = await supabaseClient
    .from(SESSION_PLAYERS_TABLE)
    .insert(sessionPlayer)
    .select()
    .single();

  if (error) {
    console.error('Error adding session player:', error);
    throw new Error(`Failed to add session player: ${error.message}`);
  }

  return data;
}

/**
 * Update a session player assignment
 * @param id Session player ID
 * @param updates Session player data to update
 * @returns The updated session player assignment
 */
export async function updateSessionPlayer(id: string, updates: SessionPlayerUpdate): Promise<SessionPlayer> {
  const { data, error } = await supabaseClient
    .from(SESSION_PLAYERS_TABLE)
    .update(updates)
    .eq('id', id)
    .select()
    .single();

  if (error) {
    console.error(`Error updating session player with ID ${id}:`, error);
    throw new Error(`Failed to update session player: ${error.message}`);
  }

  return data;
}

/**
 * Remove a player-device assignment from a session
 * @param id Session player ID
 * @returns True if successful
 */
export async function removeSessionPlayer(id: string): Promise<boolean> {
  const { error } = await supabaseClient
    .from(SESSION_PLAYERS_TABLE)
    .delete()
    .eq('id', id);

  if (error) {
    console.error(`Error removing session player with ID ${id}:`, error);
    throw new Error(`Failed to remove session player: ${error.message}`);
  }

  return true;
}

/**
 * Get a session player by ID
 * @param id Session player ID
 * @returns Session player or null if not found
 */
export async function getSessionPlayerById(id: string): Promise<SessionPlayerEntity | null> {
  const { data, error } = await supabaseClient
    .from(SESSION_PLAYERS_TABLE)
    .select(`
      *,
      player:${PLAYERS_TABLE}(id, name)
    `)
    .eq('id', id)
    .single();

  if (error) {
    if (error.code === 'PGRST116') {
      // PGRST116 is the error code for "no rows returned"
      return null;
    }
    console.error(`Error fetching session player with ID ${id}:`, error);
    throw new Error(`Failed to fetch session player: ${error.message}`);
  }

  return {
    ...data,
    playerName: data.player?.name || 'Unknown Player'
  };
}

/**
 * Add multiple player-device assignments to a session in a single batch
 * @param sessionId The ID of the session to add players to
 * @param mappings Array of objects containing playerId and deviceId
 * @returns The created session player assignments
 */
export async function addSessionPlayersBatch(sessionId: string, mappings: { playerId: string, deviceId: string }[]): Promise<SessionPlayer[]> {
  if (mappings.length === 0) {
    console.log('No player mappings provided for batch add.');
    return [];
  }

  const recordsToInsert: SessionPlayerInsert[] = mappings.map(mapping => ({
    session_id: sessionId,
    player_id: mapping.playerId,
    device_id: mapping.deviceId
  }));

  console.log(`Attempting to insert ${recordsToInsert.length} session player records for session ${sessionId}`);

  const { data, error } = await supabaseClient
    .from(SESSION_PLAYERS_TABLE)
    .insert(recordsToInsert)
    .select();

  if (error) {
    console.error('Error adding session players batch:', error);
    throw new Error(`Failed to add session players batch: ${error.message}`);
  }

  console.log(`Successfully inserted ${data?.length || 0} session player records`);
  return data || [];
} 