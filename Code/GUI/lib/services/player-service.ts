import { supabaseClient } from '@/lib/supabase/client';
import { createServerSupabaseClient } from '@/lib/supabase/server';
import { PLAYERS_TABLE } from '@/lib/supabase/schema';
import { Player, PlayerInsert, PlayerUpdate } from '@/types/supabase';
import { PlayerEntity, PlayerStatistics } from '@/types/database.types';

/**
 * Get all players
 * @returns Array of players
 */
export async function getAllPlayers(): Promise<PlayerEntity[]> {
  const { data, error } = await supabaseClient
    .from(PLAYERS_TABLE)
    .select('*')
    .order('name');

  if (error) {
    console.error('Error fetching players:', error);
    throw new Error(`Failed to fetch players: ${error.message}`);
  }

  return data || [];
}

/**
 * Get a player by ID
 * @param id Player ID
 * @returns Player or null if not found
 */
export async function getPlayerById(id: string): Promise<PlayerEntity | null> {
  const { data, error } = await supabaseClient
    .from(PLAYERS_TABLE)
    .select('*')
    .eq('id', id)
    .single();

  if (error) {
    if (error.code === 'PGRST116') {
      // PGRST116 is the error code for "no rows returned"
      return null;
    }
    console.error(`Error fetching player with ID ${id}:`, error);
    throw new Error(`Failed to fetch player: ${error.message}`);
  }

  return data;
}

/**
 * Get a player by device ID
 * @param deviceId Device ID
 * @returns Player or null if not found
 */
export async function getPlayerByDeviceId(deviceId: string): Promise<PlayerEntity | null> {
  const { data, error } = await supabaseClient
    .from(PLAYERS_TABLE)
    .select('*')
    .eq('device_id', deviceId)
    .single();

  if (error) {
    if (error.code === 'PGRST116') {
      // No player found with this device ID
      return null;
    }
    console.error(`Error fetching player with device ID ${deviceId}:`, error);
    throw new Error(`Failed to fetch player by device ID: ${error.message}`);
  }

  return data;
}

/**
 * Create a new player
 * @param player Player data to insert
 * @returns The created player
 */
export async function createPlayer(player: PlayerInsert): Promise<Player> {
  const { data, error } = await supabaseClient
    .from(PLAYERS_TABLE)
    .insert({
      ...player,
      updated_at: new Date().toISOString()
    })
    .select()
    .single();

  if (error) {
    console.error('Error creating player:', error);
    throw new Error(`Failed to create player: ${error.message}`);
  }

  return data;
}

/**
 * Update a player
 * @param id Player ID
 * @param updates Player data to update
 * @returns The updated player
 */
export async function updatePlayer(id: string, updates: PlayerUpdate): Promise<Player> {
  const { data, error } = await supabaseClient
    .from(PLAYERS_TABLE)
    .update({
      ...updates,
      updated_at: new Date().toISOString()
    })
    .eq('id', id)
    .select()
    .single();

  if (error) {
    console.error(`Error updating player with ID ${id}:`, error);
    throw new Error(`Failed to update player: ${error.message}`);
  }

  return data;
}

/**
 * Delete a player
 * @param id Player ID
 * @returns True if successful
 */
export async function deletePlayer(id: string): Promise<boolean> {
  const { error } = await supabaseClient
    .from(PLAYERS_TABLE)
    .delete()
    .eq('id', id);

  if (error) {
    console.error(`Error deleting player with ID ${id}:`, error);
    throw new Error(`Failed to delete player: ${error.message}`);
  }

  return true;
}

/**
 * Server-side function to get all players
 * @returns Array of players
 */
export async function getAllPlayersServer(): Promise<PlayerEntity[]> {
  const supabase = await createServerSupabaseClient();
  
  const { data, error } = await supabase
    .from(PLAYERS_TABLE)
    .select('*')
    .order('name');

  if (error) {
    console.error('Error fetching players on server:', error);
    throw new Error(`Failed to fetch players on server: ${error.message}`);
  }

  return data || [];
}

/**
 * Get player statistics
 * @param playerId Player ID
 * @returns Player statistics
 */
export async function getPlayerStatistics(playerId: string): Promise<PlayerStatistics | null> {
  // This is a placeholder implementation
  // In a real application, you would calculate statistics from sensor data
  const player = await getPlayerById(playerId);
  
  if (!player) {
    return null;
  }
  
  return {
    playerId: player.id,
    playerName: player.name,
    totalSessions: 0,
    totalDuration: 0,
    averageAcceleration: 0,
    maxAcceleration: 0,
    averageRotationRate: 0,
    maxRotationRate: 0,
    lastSessionDate: new Date(),
  };
} 