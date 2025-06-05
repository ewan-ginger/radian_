'use client';

import { useState, useCallback } from 'react';
import { SensorDataEntity, TimeSeriesDataPoint, OrientationData } from '@/types/database.types';
import { SensorDataInsert } from '@/types/supabase';
import * as sensorDataService from '@/lib/services/sensor-data-service';

export function useSensorData() {
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const getSensorDataBySession = useCallback(async (
    sessionId: string,
    limit: number = 1000,
    offset: number = 0
  ): Promise<SensorDataEntity[]> => {
    try {
      setIsLoading(true);
      setError(null);
      const data = await sensorDataService.getSensorDataBySession(sessionId, limit, offset);
      return data.map(sensorDataService.processSensorData);
    } catch (err) {
      console.error('Error fetching sensor data:', err);
      setError('Failed to fetch sensor data. Please try again.');
      return [];
    } finally {
      setIsLoading(false);
    }
  }, []);

  const getSensorDataBySessionAndPlayer = useCallback(async (
    sessionId: string,
    playerId: string,
    limit: number = 1000,
    offset: number = 0
  ): Promise<SensorDataEntity[]> => {
    try {
      setIsLoading(true);
      setError(null);
      const data = await sensorDataService.getSensorDataBySessionAndPlayer(
        sessionId,
        playerId,
        limit,
        offset
      );
      return data.map(sensorDataService.processSensorData);
    } catch (err) {
      console.error('Error fetching sensor data:', err);
      setError('Failed to fetch sensor data. Please try again.');
      return [];
    } finally {
      setIsLoading(false);
    }
  }, []);

  const insertSensorData = useCallback(async (sensorData: SensorDataInsert): Promise<SensorDataEntity> => {
    try {
      setError(null);
      const data = await sensorDataService.insertSensorData(sensorData);
      return sensorDataService.processSensorData(data);
    } catch (err) {
      console.error('Error inserting sensor data:', err);
      setError('Failed to insert sensor data. Please try again.');
      throw err;
    }
  }, []);

  const insertSensorDataBatch = useCallback(async (sensorDataBatch: SensorDataInsert[]): Promise<boolean> => {
    try {
      setError(null);
      return await sensorDataService.insertSensorDataBatch(sensorDataBatch);
    } catch (err) {
      console.error('Error inserting sensor data batch:', err);
      setError('Failed to insert sensor data batch. Please try again.');
      throw err;
    }
  }, []);

  const getTimeSeriesData = useCallback(async (
    sessionId: string,
    playerId: string,
    sensorType: keyof SensorDataEntity,
    limit: number = 100
  ): Promise<TimeSeriesDataPoint[]> => {
    try {
      setIsLoading(true);
      setError(null);
      return await sensorDataService.getTimeSeriesData(sessionId, playerId, sensorType, limit);
    } catch (err) {
      console.error('Error fetching time series data:', err);
      setError('Failed to fetch time series data. Please try again.');
      return [];
    } finally {
      setIsLoading(false);
    }
  }, []);

  const getOrientationData = useCallback(async (
    sessionId: string,
    playerId: string,
    limit: number = 100
  ): Promise<OrientationData[]> => {
    try {
      setIsLoading(true);
      setError(null);
      return await sensorDataService.getOrientationData(sessionId, playerId, limit);
    } catch (err) {
      console.error('Error fetching orientation data:', err);
      setError('Failed to fetch orientation data. Please try again.');
      return [];
    } finally {
      setIsLoading(false);
    }
  }, []);

  return {
    isLoading,
    error,
    getSensorDataBySession,
    getSensorDataBySessionAndPlayer,
    insertSensorData,
    insertSensorDataBatch,
    getTimeSeriesData,
    getOrientationData,
  };
} 