'use client';

import { useState, useEffect, useCallback } from 'react';
import { PlayerEntity } from '@/types/database.types';
import * as playerService from '@/lib/services/player-service';
import { PlayerInsert, PlayerUpdate } from '@/types/supabase';

export function usePlayerData() {
  const [players, setPlayers] = useState<PlayerEntity[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const fetchPlayers = useCallback(async () => {
    try {
      setIsLoading(true);
      setError(null);
      const data = await playerService.getAllPlayers();
      setPlayers(data);
    } catch (err) {
      console.error('Error fetching players:', err);
      setError('Failed to fetch players. Please try again.');
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchPlayers();
  }, [fetchPlayers]);

  const createPlayer = useCallback(async (playerData: PlayerInsert) => {
    try {
      setError(null);
      const newPlayer = await playerService.createPlayer(playerData);
      setPlayers(prev => [...prev, newPlayer]);
      return newPlayer;
    } catch (err) {
      console.error('Error creating player:', err);
      setError('Failed to create player. Please try again.');
      throw err;
    }
  }, []);

  const updatePlayer = useCallback(async (id: string, playerData: PlayerUpdate) => {
    try {
      setError(null);
      const updatedPlayer = await playerService.updatePlayer(id, playerData);
      setPlayers(prev => prev.map(player => 
        player.id === id ? updatedPlayer : player
      ));
      return updatedPlayer;
    } catch (err) {
      console.error('Error updating player:', err);
      setError('Failed to update player. Please try again.');
      throw err;
    }
  }, []);

  const deletePlayer = useCallback(async (id: string) => {
    try {
      setError(null);
      await playerService.deletePlayer(id);
      setPlayers(prev => prev.filter(player => player.id !== id));
      return true;
    } catch (err) {
      console.error('Error deleting player:', err);
      setError('Failed to delete player. Please try again.');
      throw err;
    }
  }, []);

  return {
    players,
    isLoading,
    error,
    fetchPlayers,
    createPlayer,
    updatePlayer,
    deletePlayer,
  };
} 