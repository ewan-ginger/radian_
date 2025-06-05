"use client";

import React, { useState, useEffect, useMemo } from 'react';
import { LacrosseStickAnimation } from './LacrosseStickAnimation';
import { Button } from '@/components/ui/button';
import { Play, Pause } from 'lucide-react';
import { SensorDataEntity, SessionPlayerEntity } from '@/types/database.types'; // Assuming these types are correct
import * as THREE from 'three'; // Import THREE for Quaternion and Euler

interface ActionGroup {
    label: string;
    startTime: number; // Relative seconds from session start
    endTime: number;   // Relative seconds from session start
    metric: number | null;
    count: number;
    playerName: string | null;
    playerId: string | null;
}

interface OrientationDataPoint {
    timestamp: number; // Milliseconds relative to action start
    x: number; // Roll
    y: number; // Pitch
    z: number; // Yaw
}

interface LacrosseStickAnimationLoaderProps {
    action: ActionGroup;
    sessionId: string;
    allSensorData: SensorDataEntity[];
    sessionPlayers: SessionPlayerEntity[];
    modelPath: string; // Path to the .glb model
}

const LacrosseStickAnimationLoader: React.FC<LacrosseStickAnimationLoaderProps> = ({ 
    action, 
    sessionId, 
    allSensorData,
    sessionPlayers,
    modelPath 
}) => {
    const [isPlaying, setIsPlaying] = useState(false);
    const [orientationData, setOrientationData] = useState<OrientationDataPoint[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    const actionDurationSeconds = useMemo(() => action.endTime - action.startTime, [action]);

    // Helper function for pre-interpolation
    const interpolateData = ( 
        inputData: OrientationDataPoint[], 
        targetIntervalMs: number = 2 // For 500Hz
    ): OrientationDataPoint[] => {
        if (inputData.length < 2) {
            return inputData; // Not enough data to interpolate
        }

        const interpolated: OrientationDataPoint[] = [];
        // Add the first point as is
        interpolated.push(inputData[0]);

        for (let i = 0; i < inputData.length - 1; i++) {
            const p0 = inputData[i];
            const p1 = inputData[i+1];

            const t0 = p0.timestamp;
            const t1 = p1.timestamp;

            if (t1 <= t0) { // Should not happen with sorted data, but guard it
                if (i + 1 < inputData.length -1) interpolated.push(inputData[i+1]); // push next original if this segment is invalid
                continue;
            }

            const eulerOrder: THREE.EulerOrder = 'ZYX'; // Match the order used in animation component

            const q0 = new THREE.Quaternion().setFromEuler(
                new THREE.Euler(
                    p0.x * (Math.PI / 180),
                    p0.y * (Math.PI / 180),
                    p0.z * (Math.PI / 180),
                    eulerOrder
                )
            );
            const q1 = new THREE.Quaternion().setFromEuler(
                new THREE.Euler(
                    p1.x * (Math.PI / 180),
                    p1.y * (Math.PI / 180),
                    p1.z * (Math.PI / 180),
                    eulerOrder
                )
            );

            // Ensure SLERP takes the shortest path
            if (q0.dot(q1) < 0) {
                q1.conjugate(); // Revert to conjugate, as multiplyScalar was incorrect for Quaternion here
            }

            let currentTime = t0 + targetIntervalMs;
            while (currentTime < t1) {
                const alpha = (currentTime - t0) / (t1 - t0);
                const qInterpolated = new THREE.Quaternion().copy(q0).slerp(q1, alpha);
                
                const eulerInterpolated = new THREE.Euler().setFromQuaternion(qInterpolated, eulerOrder);
                
                interpolated.push({
                    timestamp: Math.round(currentTime),
                    x: eulerInterpolated.x * (180 / Math.PI),
                    y: eulerInterpolated.y * (180 / Math.PI),
                    z: eulerInterpolated.z * (180 / Math.PI),
                });
                currentTime += targetIntervalMs;
            }
            // Add the next original point
            interpolated.push(p1);
        }
        
        // Deduplicate points that might have the same timestamp after interpolation/rounding
        // and ensure sorted order (though it should be mostly sorted)
        const uniqueSortedInterpolated = Array.from(new Map(interpolated.map(item => [item.timestamp, item])).values())
                                           .sort((a, b) => a.timestamp - b.timestamp);

        console.log(`Original points: ${inputData.length}, Interpolated to: ${uniqueSortedInterpolated.length} points at ${targetIntervalMs}ms interval.`);
        return uniqueSortedInterpolated;
    };

    useEffect(() => {
        setIsLoading(true);
        setError(null);
        setOrientationData([]);
        setIsPlaying(false); // Reset play state when action changes

        if (!action || !allSensorData || allSensorData.length === 0) {
            setError("No sensor data available for this session or action not provided.");
            setIsLoading(false);
            return;
        }

        // Find the device_id for the player associated with this action
        let targetDeviceId: string | null = null;
        if (action.playerId) {
            const player = sessionPlayers.find(p => p.player_id === action.playerId);
            if (player && player.device_id) {
                targetDeviceId = player.device_id;
            } else {
                console.warn(`No device_id found for player ${action.playerId} in action ${action.label}`);
                // Fallback: Try to find a device for any player if specific player not found or has no device
                // This might not be ideal, but better than no animation if data exists for *a* device
            }
        }

        // If no targetDeviceId yet (e.g. action.playerId was null, or player had no device_id)
        // and there's only one player/device in the session, assume that one.
        if (!targetDeviceId && sessionPlayers.length === 1 && sessionPlayers[0].device_id) {
            targetDeviceId = sessionPlayers[0].device_id;
            console.log(`Action player unknown or no device, defaulting to single device in session: ${targetDeviceId}`);
        }

        const actionStartTimeSeconds = action.startTime;
        const actionEndTimeSeconds = action.endTime;

        // Filter sensor data for the relevant time window and device (if known)
        const relevantSensorData = allSensorData.filter(data => {
            const dataTimestampSeconds = data.timestamp; // Assuming timestamp is in seconds from session start
            if (dataTimestampSeconds === null || dataTimestampSeconds === undefined) return false;

            const isCorrectDevice = targetDeviceId ? data.device_id === targetDeviceId : true; // If no device, take data from any
            const isInTimeWindow = dataTimestampSeconds >= actionStartTimeSeconds && dataTimestampSeconds <= actionEndTimeSeconds;
            
            return isCorrectDevice && isInTimeWindow &&
                   data.orientation_x !== null && data.orientation_y !== null && data.orientation_z !== null;
        });

        if (relevantSensorData.length === 0) {
            const errMessage = targetDeviceId 
                ? `No orientation data found for device ${targetDeviceId} during action: ${action.label} (${actionStartTimeSeconds.toFixed(2)}s - ${actionEndTimeSeconds.toFixed(2)}s).`
                : `No orientation data found for any device during action: ${action.label} (${actionStartTimeSeconds.toFixed(2)}s - ${actionEndTimeSeconds.toFixed(2)}s). Ensure player is assigned to a device.`;
            setError(errMessage);
            console.warn(errMessage, { action, targetDeviceId, allSensorDataCount: allSensorData.length });
            setIsLoading(false);
            return;
        }

        // Sort by timestamp and transform to OrientationDataPoint format (degrees, ms relative to action start)
        const initialProcessedData: OrientationDataPoint[] = relevantSensorData
            .sort((a, b) => (a.timestamp ?? 0) - (b.timestamp ?? 0))
            .map(data => ({
                timestamp: Math.round(((data.timestamp ?? 0) - actionStartTimeSeconds) * 1000), 
                x: data.orientation_x || 0, 
                y: data.orientation_y || 0, 
                z: data.orientation_z || 0, 
            }));

        if (initialProcessedData.length === 0) {
             setError(`No orientation data points after initial processing for action: ${action.label}`);
             setIsLoading(false);
             return;
        }

        console.log(`Initially processed ${initialProcessedData.length} orientation points for action '${action.label}', device ${targetDeviceId || 'any'}`);
        
        // Pre-interpolate the data
        const finalInterpolatedData = interpolateData(initialProcessedData, 2); // 2ms for 500Hz

        setOrientationData(finalInterpolatedData);
        setIsLoading(false);

    }, [action, sessionId, allSensorData, sessionPlayers]);

    const handlePlayPause = () => {
        if (orientationData.length > 0) {
            setIsPlaying(!isPlaying);
        }
    };

    if (isLoading) {
        return (
            <div className="p-4 text-center text-sm text-muted-foreground">
                <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-primary mx-auto mb-2"></div>
                Loading animation data...
            </div>
        );
    }

    if (error) {
        return <div className="p-4 text-sm text-red-600 bg-red-50 rounded-md">Error: {error}</div>;
    }

    if (orientationData.length === 0) {
        return <div className="p-4 text-center text-sm text-muted-foreground">No orientation data to display for this action.</div>;
    }

    return (
        <div className="my-4 p-1 border rounded-lg bg-background">
            <LacrosseStickAnimation
                orientationData={orientationData}
                modelPath={modelPath} // Make sure this path is correct
                isPlaying={isPlaying}
                actionDurationSeconds={actionDurationSeconds}
            />
            <div className="mt-2 flex justify-center items-center p-2 border-t">
                <Button onClick={handlePlayPause} variant="outline" size="sm" disabled={orientationData.length === 0}>
                    {isPlaying ? <Pause className="h-4 w-4 mr-1.5" /> : <Play className="h-4 w-4 mr-1.5" />}
                    {isPlaying ? 'Pause' : 'Play'} Animation
                </Button>
                 <p className="ml-3 text-xs text-muted-foreground">
                    Duration: {actionDurationSeconds.toFixed(2)}s | Points: {orientationData.length}
                 </p>
            </div>
        </div>
    );
};

export { LacrosseStickAnimationLoader }; 