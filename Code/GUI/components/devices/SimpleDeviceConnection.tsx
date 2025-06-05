'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Usb, Play, Pause, RefreshCw, Bluetooth } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { useRecording } from '@/context/RecordingContext';
import { formatDuration, formatSessionType } from '@/lib/utils';
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { checkSessionNameExists, createSession } from '@/lib/services/session-service';
import { addSessionPlayersBatch } from '@/lib/services/session-player-service';
import { LiveDataGraph } from './LiveDataGraph';
import { useRouter } from 'next/navigation';
import { 
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue
} from '@/components/ui/select';
import { usePlayerData } from '@/hooks/usePlayerData';
import { SessionType, getRequiredPlayers } from '@/types/database.types';
import { toast } from "sonner";
import { SessionManager } from '@/lib/data/session-manager';

// Simple Alert component since we don't have a dedicated alert component
const Alert = ({ className, children }: { className?: string, children: React.ReactNode }) => {
  return (
    <div className={`p-3 border rounded bg-yellow-50 text-yellow-800 ${className || ''}`}>
      {children}
    </div>
  );
};

// AlertDescription component
const AlertDescription = ({ children }: { children: React.ReactNode }) => {
  return <div className="text-sm">{children}</div>;
};

// Define calibration configurations including duration and beep intervals
// Moved here to be accessible for display logic
const CALIBRATION_CONFIG: Partial<Record<SessionType, { durationMinutes: number; beepIntervalSeconds: number | null }>> = {
  'pass_calibration':         { durationMinutes: 5, beepIntervalSeconds: 5 },
  'groundball_calibration':   { durationMinutes: 1, beepIntervalSeconds: 5 },
  'pass_catch_calibration':   { durationMinutes: 1, beepIntervalSeconds: 5 }, 
  'shot_calibration':         { durationMinutes: 1, beepIntervalSeconds: 10 },
  'faceoff_calibration':      { durationMinutes: 1, beepIntervalSeconds: 15 },
  'cradle_calibration':       { durationMinutes: 1, beepIntervalSeconds: null }, // No beeps for cradle
};

// Simple spinner component since we don't have a dedicated spinner component
const Spinner = ({ size = "md" }: { size?: "sm" | "md" | "lg" }) => {
  const sizeClasses = {
    sm: "h-4 w-4",
    md: "h-8 w-8",
    lg: "h-12 w-12"
  };
  
  return (
    <div className={`animate-spin rounded-full border-b-2 border-primary ${sizeClasses[size]}`}></div>
  );
};

// Need to declare the window.navigator interface for TypeScript
declare global {
  interface Navigator {
    serial?: {
      requestPort: (options?: any) => Promise<any>;
      getPorts: () => Promise<any[]>;
    };
  }
}

// Define all session types for the dropdown
const ALL_SESSION_TYPES: SessionType[] = [
  'solo',
  'pass_calibration',
  'pass_catch_calibration',
  'groundball_calibration',
  'shot_calibration',
  'faceoff_calibration',
  'cradle_calibration',
  'passing_partners',
  '2v2'
];

export function SimpleDeviceConnection() {
  const router = useRouter();
  const [isConnected, setIsConnected] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [status, setStatus] = useState('Disconnected');
  const [sensorData, setSensorData] = useState<string[]>([]);
  const [parsedSensorData, setParsedSensorData] = useState<Record<string, any[]>>({});
  const [sessionName, setSessionName] = useState('');
  const [sessionNameError, setSessionNameError] = useState('');
  const [isStopping, setIsStopping] = useState(false);
  const [isRedirecting, setIsRedirecting] = useState(false);
  const [sessionType, setSessionType] = useState<SessionType>('solo');
  const [deviceId, setDeviceId] = useState<string>('1');
  const [selectedPlayerId, setSelectedPlayerId] = useState<string>('');
  const [playerSelectionError, setPlayerSelectionError] = useState<string>('');
  const [calibrationTimeRemaining, setCalibrationTimeRemaining] = useState<number | null>(null);
  const [numberOfGroundballPlayers, setNumberOfGroundballPlayers] = useState<number>(1);
  const calibrationTimerRef = useRef<NodeJS.Timeout | null>(null);
  const beepTimerRef = useRef<NodeJS.Timeout | null>(null);
  const firstBeepTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const mainBeepAudioRef = useRef<HTMLAudioElement | null>(null);
  const readyBeepAudioRef = useRef<HTMLAudioElement | null>(null);
  const setBeepAudioRef = useRef<HTMLAudioElement | null>(null);
  const timeoutIdsRef = useRef<number[]>([]);
  const timeoutCleanupRef = useRef<(() => void) | null>(null);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [liveDuration, setLiveDuration] = useState<number>(0);
  const [liveDataPoints, setLiveDataPoints] = useState<number>(0);
  const [sessionStartTime, setSessionStartTime] = useState<number | null>(null);
  const sessionStartTimeRef = useRef<number | null>(null);
  const liveStatsIntervalRef = useRef<NodeJS.Timeout | null>(null);
  
  const { 
    isRecording, 
    recordingDuration, 
    dataPoints, 
    stopRecording,
    sessionId: contextSessionId
  } = useRecording();
  
  const portRef = useRef<any | null>(null);
  const readerRef = useRef<ReadableStreamDefaultReader<Uint8Array> | null>(null);
  const writerRef = useRef<WritableStreamDefaultWriter<Uint8Array> | null>(null);
  const readLoopRef = useRef<boolean>(false);
  
  // Get player data
  const { players, isLoading: playersLoading } = usePlayerData();
  
  // State for multiple player/device mappings
  const [playerDeviceMappings, setPlayerDeviceMappings] = useState<{ playerId: string, deviceId: string }[]>([{ playerId: '', deviceId: '' }]);
  const [playerDeviceErrors, setPlayerDeviceErrors] = useState<string[]>([]); // Array of errors, one per mapping
  
  // Local SessionManager instance
  const sessionManagerRef = useRef<SessionManager | null>(null);

  // Initialize SessionManager on mount
  useEffect(() => {
      sessionManagerRef.current = new SessionManager();
      console.log("Local SessionManager initialized.");
  }, []);
  
  // Create audio elements for beep sounds
  useEffect(() => {
    // Create audio elements
    mainBeepAudioRef.current = new Audio('/sounds/beep.mp3');
    readyBeepAudioRef.current = new Audio('/sounds/ready-beep.mp3');
    setBeepAudioRef.current = new Audio('/sounds/set-beep.mp3');
    
    // Add error handler for the ready beep
    if (readyBeepAudioRef.current) {
      readyBeepAudioRef.current.onerror = () => {
        console.warn('Ready beep sound file not found, using fallback');
        // Use main beep as fallback - try reassigning src and reloading
        // Check ref is still valid inside the handler
        if (readyBeepAudioRef.current) { 
          readyBeepAudioRef.current.src = '/sounds/beep.mp3'; // Reassign src
          readyBeepAudioRef.current.volume = 0.7;
          readyBeepAudioRef.current.playbackRate = 0.8; // Slower for "Ready"
          readyBeepAudioRef.current.load(); // Reload
        }
      };
    }
    
    // Add error handler for the set beep
    if (setBeepAudioRef.current) {
      setBeepAudioRef.current.onerror = () => {
        console.warn('Set beep sound file not found, using fallback');
        // Use main beep as fallback - try reassigning src and reloading
        // Check ref is still valid inside the handler
        if (setBeepAudioRef.current) { 
          setBeepAudioRef.current.src = '/sounds/beep.mp3'; // Reassign src
          setBeepAudioRef.current.volume = 0.8;
          setBeepAudioRef.current.playbackRate = 1.0; // Normal rate for "Set"
          setBeepAudioRef.current.load(); // Reload
        }
      };
    }
    
    // Set properties
    if (mainBeepAudioRef.current) {
      mainBeepAudioRef.current.volume = 1.0;
      mainBeepAudioRef.current.load();
    }
    
    if (readyBeepAudioRef.current) {
      readyBeepAudioRef.current.volume = 0.8;
      readyBeepAudioRef.current.load();
    }
    
    if (setBeepAudioRef.current) {
      setBeepAudioRef.current.volume = 0.9;
      setBeepAudioRef.current.load();
    }
    
    return () => {
      mainBeepAudioRef.current = null;
      readyBeepAudioRef.current = null;
      setBeepAudioRef.current = null;
    };
  }, []);
  
  // Function to play main beep sound with retry
  const playMainBeep = (volume = 1.0) => {
    if (!mainBeepAudioRef.current) return;
    
    // Reset audio to start
    mainBeepAudioRef.current.currentTime = 0;
    
    // Set the volume for this beep
    mainBeepAudioRef.current.volume = volume;
    
    // Play with retry logic if it fails
    const playPromise = mainBeepAudioRef.current.play();
    
    if (playPromise !== undefined) {
      playPromise.catch(err => {
        console.error('Error playing main beep, retrying:', err);
        // Retry after a short delay
        setTimeout(() => {
          if (mainBeepAudioRef.current) {
            mainBeepAudioRef.current.currentTime = 0;
            mainBeepAudioRef.current.play().catch(err => console.error('Retry failed:', err));
          }
        }, 100);
      });
    }
  };
  
  // Function to play ready beep sound with retry
  const playReadyBeep = (volume = 0.8) => {
    if (!readyBeepAudioRef.current) return;
    
    // Reset audio to start
    readyBeepAudioRef.current.currentTime = 0;
    
    // Set the volume for this beep
    readyBeepAudioRef.current.volume = volume;
    
    // Play with retry logic if it fails
    const playPromise = readyBeepAudioRef.current.play();
    
    if (playPromise !== undefined) {
      playPromise.catch(err => {
        console.error('Error playing ready beep, retrying:', err);
        // Retry after a short delay
        setTimeout(() => {
          if (readyBeepAudioRef.current) {
            readyBeepAudioRef.current.currentTime = 0;
            readyBeepAudioRef.current.play().catch(err => console.error('Retry failed:', err));
          }
        }, 100);
      });
    }
  };
  
  // Function to play set beep sound with retry
  const playSetBeep = (volume = 0.9) => {
    if (!setBeepAudioRef.current) return;
    
    // Reset audio to start
    setBeepAudioRef.current.currentTime = 0;
    
    // Set the volume for this beep
    setBeepAudioRef.current.volume = volume;
    
    // Play with retry logic if it fails
    const playPromise = setBeepAudioRef.current.play();
    
    if (playPromise !== undefined) {
      playPromise.catch(err => {
        console.error('Error playing set beep, retrying:', err);
        // Retry after a short delay
        setTimeout(() => {
          if (setBeepAudioRef.current) {
            setBeepAudioRef.current.currentTime = 0;
            setBeepAudioRef.current.play().catch(err => console.error('Retry failed:', err));
          }
        }, 100);
      });
    }
  };
  
  // Function to play the Ready-Set warning sequence
  const playReadySetSequence = (timeToMainBeep = 1000) => {
    // Calculate when to play each sound based on the time to main beep
    const readyTime = 0; // Play "Ready" immediately
    const setTime = timeToMainBeep - 500; // Play "Set" 0.5 seconds before main beep
    
    // Play "Ready" beep
    playReadyBeep();
    
    // Play "Set" beep at the calculated time
    setTimeout(() => {
      playSetBeep();
    }, setTime);
  };
  
  // Validate session name
  const validateSessionName = async () => {
    const trimmedName = sessionName.trim();
    if (!trimmedName) {
      setSessionNameError('Please enter a session name');
      return false;
    }

    try {
      console.log('Checking if session name exists:', trimmedName);
      const exists = await checkSessionNameExists(trimmedName);
      console.log('Session name exists?', exists);
      
      if (exists) {
        console.log('Session name already exists, showing error');
        setSessionNameError('A session with this name already exists');
        return false;
      }
      
      console.log('Session name is unique, proceeding');
      setSessionNameError('');
      return true;
    } catch (error) {
      console.error('Error checking session name:', error);
      // Show a more user-friendly error message
      setSessionNameError('Unable to validate session name. Please try a different name.');
      return false;
    }
  };
  
  // Function to send pairing command to selected devices
  async function handlePairDevices() {
    if (!isConnected) {
      console.error('Cannot pair: not connected');
      toast.error('Device not connected');
      return;
    }

    // Ensure all device IDs are selected
    const allDevicesSelected = playerDeviceMappings.every(m => m.deviceId);
    if (!allDevicesSelected) {
      toast.error('Please select a device ID for all players before pairing.');
      return;
    }

    console.log('Starting continuous pairing process for 10 seconds for devices:', playerDeviceMappings);
    toast.info('Sending pairing commands for 10 seconds...');

    const pairingPromises = playerDeviceMappings.map(mapping => {
      return new Promise<void>(async (resolve, reject) => {
        if (!mapping.deviceId) {
          // Skip if no device ID, but resolve immediately
          resolve();
          return;
        }

        const deviceId = mapping.deviceId;
        const command = `${deviceId}:connect`;
        const pairingDuration = 10000; // 10 seconds
        const sendInterval = 500; // Send every 100ms

        let intervalId: NodeJS.Timeout | null = null;
        let timeoutId: NodeJS.Timeout | null = null;

        try {
          console.log(`[Device ${deviceId}] Starting pairing commands.`);
          
          // Function to send the command
          const sendPairCommand = async () => {
            try {
              await sendCommand(command);
              // console.log(`[Device ${deviceId}] Sent: ${command}`); // Optional: Log each send
            } catch (sendError) {
              console.error(`[Device ${deviceId}] Error sending periodic pair command:`, sendError);
              // Stop sending for this device on error
              if (intervalId) clearInterval(intervalId);
              if (timeoutId) clearTimeout(timeoutId);
              reject(sendError); 
            }
          };

          // Send immediately once
          await sendPairCommand();

          // Then send repeatedly
          intervalId = setInterval(sendPairCommand, sendInterval);

          // Stop sending after 10 seconds
          timeoutId = setTimeout(() => {
            if (intervalId) {
              clearInterval(intervalId);
              console.log(`[Device ${deviceId}] Stopped sending pairing commands after 10 seconds.`);
              resolve(); // Pairing sequence for this device completed successfully
            }
          }, pairingDuration);

        } catch (initialSendError) {
          console.error(`[Device ${deviceId}] Error sending initial pair command:`, initialSendError);
          if (intervalId) clearInterval(intervalId); // Clean up interval if initial send failed
          if (timeoutId) clearTimeout(timeoutId);
          reject(initialSendError); // Reject the promise for this device
        }
      });
    });

    try {
      await Promise.all(pairingPromises);
      toast.success('Pairing commands sent successfully for 10 seconds.');
      console.log('All pairing sequences completed.');
    } catch (error) {
      console.error('Error during pairing sequence for one or more devices:', error);
      toast.error(`Pairing failed for one or more devices: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }
  
  // Effect to update the number of player/device inputs based on session type
  useEffect(() => {
    const requiredPlayers = getRequiredPlayers(sessionType, sessionType === 'groundball_calibration' ? numberOfGroundballPlayers : undefined);
    setPlayerDeviceMappings(currentMappings => {
      const newMappings = Array(requiredPlayers).fill(null).map((_, index) => {
        // Preserve existing mapping if possible, otherwise create new empty one
        // Auto-assign device IDs 1, 2, 3... for simplicity?
        const existingMapping = currentMappings[index];
        return {
           playerId: existingMapping?.playerId || '',
           deviceId: existingMapping?.deviceId || String(index + 1) // Default to Device 1, 2, 3...
        };
      });
      return newMappings;
    });
    // Reset errors when type changes
    setPlayerDeviceErrors(Array(requiredPlayers).fill('')); 
  }, [sessionType, numberOfGroundballPlayers]);
  
  // Function to update a specific player/device mapping
  const updatePlayerDeviceMapping = useCallback((index: number, field: 'playerId' | 'deviceId', value: string) => {
    setPlayerDeviceMappings(prev => {
      const newMappings = [...prev];
      newMappings[index] = { ...newMappings[index], [field]: value };
      return newMappings;
    });
    // Clear error for this specific input when changed
    setPlayerDeviceErrors(prev => {
        const newErrors = [...prev];
        if (newErrors[index]) { // Only clear if there was an error
            // More robust error clearing: check which field was updated
            if (field === 'playerId' && newErrors[index].includes('Player')) {
                newErrors[index] = newErrors[index].replace('Player required', '').replace(', ', '').trim();
            }
            if (field === 'deviceId' && newErrors[index].includes('Device')) {
                newErrors[index] = newErrors[index].replace('Device required', '').replace(', ', '').trim();
            }
        }
        return newErrors;
    });
  }, []);
  
  async function connectSerial() {
    if (isConnected) {
      console.log('Already connected');
      return;
    }
    
    try {
      if (!navigator.serial) {
        setStatus('Web Serial API not supported');
        console.error('Web Serial API not supported in this browser');
        return;
      }
      
      // Request a port from the user
      console.log('Requesting serial port...');
      portRef.current = await navigator.serial.requestPort({
        // Add filters if you need specific devices
      });
      
      console.log('Port selected:', portRef.current);
      console.log('Opening port with baudRate: 115200');
      
      await portRef.current.open({ baudRate: 115200 });
      
      console.log('Port opened successfully');
      
      // Get writer
      const writer = portRef.current.writable.getWriter();
      writerRef.current = writer;
      
      // Get reader
      const reader = portRef.current.readable.getReader();
      readerRef.current = reader;
      
      setIsConnected(true);
      setStatus('Connected');
      
      // Start reading from the serial port
      readLoopRef.current = true;
      readSerial();
      
      console.log('Connection established successfully');
    } catch (error) {
      console.error('Connection failed:', error);
      setStatus(`Error: ${error instanceof Error ? error.message : String(error)}`);
    }
  }
  
  async function disconnectSerial() {
    try {
      console.log('Disconnecting...');
      
      // Stop streaming if active
      if (isStreaming) {
        await handleStopStreaming();
      }
      
      // Stop the read loop
      readLoopRef.current = false;
      
      // Release reader
      if (readerRef.current) {
        await readerRef.current.cancel();
        readerRef.current.releaseLock();
        readerRef.current = null;
      }
      
      // Release writer
      if (writerRef.current) {
        writerRef.current.releaseLock();
        writerRef.current = null;
      }
      
      // Close port
      if (portRef.current) {
        await portRef.current.close();
        portRef.current = null;
      }
      
      setIsConnected(false);
      setStatus('Disconnected');
      
      console.log('Disconnected successfully');
    } catch (error) {
      console.error('Disconnection error:', error);
      setStatus(`Error: ${error instanceof Error ? error.message : String(error)}`);
    }
  }
  
  async function readSerial() {
    try {
      const decoder = new TextDecoder();
      let buffer = '';
      
      while (readLoopRef.current && readerRef.current) {
        const { value, done } = await readerRef.current.read();
        
        if (done) {
          console.log('Reader done');
          break;
        }
        
        buffer += decoder.decode(value, { stream: true });
        let lines = buffer.split('\n');
        buffer = lines.pop() || '';
        
        lines.forEach(line => {
          line = line.trim();
          // console.log('Received:', line); // Reduce console noise

          if (line.startsWith('DATA:')) {
            setSensorData(prev => [...prev.slice(-99), line]); // Keep raw data view minimal

            if (!isStopping) {
              try {
                const dataStr = line.substring(5).trim();
                const values = dataStr.split(',').map(val => parseFloat(val.trim()));

                if (values.length >= 15) {
                  const deviceId = String(values[0]);

                  if (sessionManagerRef.current) {
                    sessionManagerRef.current.addSensorData(values);
                  } else {
                    console.warn('Session Manager not available to add data point.');
                  }

                  const parsedDataPoint = {
                    // No need for deviceId inside the point itself anymore
                    timestamp: values[1] || 0,
                    orientation_x: values[3] || 0,
                    orientation_y: values[4] || 0,
                    orientation_z: values[5] || 0,
                    accelerometer_x: values[6] || 0,
                    accelerometer_y: values[7] || 0,
                    accelerometer_z: values[8] || 0,
                    gyroscope_x: values[9] || 0,
                    gyroscope_y: values[10] || 0,
                    gyroscope_z: values[11] || 0,
                    magnetometer_x: values[12] || 0,
                    magnetometer_y: values[13] || 0,
                    magnetometer_z: values[14] || 0,
                  };

                  // Update the state for the specific device ID
                  setParsedSensorData(prev => {
                    const currentDeviceData = prev[deviceId] || [];
                    const newData = [...currentDeviceData, parsedDataPoint];
                    // Keep only the last 100 points *per device*
                    const trimmedData = newData.length > 100 ? newData.slice(-100) : newData;
                    return {
                      ...prev,
                      [deviceId]: trimmedData,
                    };
                  });

                } else {
                  console.warn('Invalid data format, expected 15 values but got:', values.length, 'Line:', line);
                }
              } catch (err) {
                console.error('Error parsing data:', err, 'Line:', line);
              }
            } else {
              // console.log('Ignoring data point while stopping session'); // Reduce noise
            }
          }
        });
      }
      
      console.log('Read loop ended');
    } catch (error) {
      console.error('Error reading serial:', error);
      setStatus(`Error: ${error instanceof Error ? error.message : String(error)}`);
    }
  }
  
  async function sendCommand(command: string) {
    if (writerRef.current && isConnected) {
      try {
        console.log(`Sending command: ${command}`);
        await writerRef.current.write(new TextEncoder().encode(command + '\n'));
        console.log(`Sent command: ${command}`);
      } catch (error) {
        console.error('Error sending command:', error);
        setStatus(`Error: ${error instanceof Error ? error.message : String(error)}`);
      }
    }
  }
  
  // Function to start calibration timer if applicable
  const startCalibrationTimerIfNeeded = () => {
    // Get config for the current session type
    const config = CALIBRATION_CONFIG[sessionType];

    // Only proceed if it's a known calibration type with config
    if (!config) {
        console.log('Not a configured calibration session type, no timer/beeps needed.');
        return () => {}; // Return empty cleanup
    }

    const CALIBRATION_DURATION_MS = config.durationMinutes * 60 * 1000;
    const BEEP_INTERVAL_MS = config.beepIntervalSeconds ? config.beepIntervalSeconds * 1000 : null;
      
    console.log(`Starting calibration timer for ${sessionType}. Duration: ${config.durationMinutes} min. Beep Interval: ${config.beepIntervalSeconds}s`);

    // Calculate end time from now
    const startTime = Date.now();
    const endTime = startTime + CALIBRATION_DURATION_MS;
    
    // Set initial time remaining
    setCalibrationTimeRemaining(CALIBRATION_DURATION_MS / 1000);
    
    // Clear any existing timers
    if (calibrationTimerRef.current) {
      clearInterval(calibrationTimerRef.current);
    }
    if (beepTimerRef.current) {
      clearInterval(beepTimerRef.current);
    }
    if (firstBeepTimeoutRef.current) {
      clearTimeout(firstBeepTimeoutRef.current);
    }
    
    // Clean up any existing timeouts
    timeoutIdsRef.current.forEach(id => window.clearTimeout(id));
    timeoutIdsRef.current = [];
    
    // Start the countdown timer
    calibrationTimerRef.current = setInterval(() => {
      const now = Date.now();
      const remaining = Math.max(0, Math.floor((endTime - now) / 1000));
      setCalibrationTimeRemaining(remaining);
      
      // Auto-stop when timer reaches 0
      if (remaining === 0) {
        clearInterval(calibrationTimerRef.current!);
        calibrationTimerRef.current = null;
        
        // Also clear beep timer
        if (beepTimerRef.current) {
          clearInterval(beepTimerRef.current);
          beepTimerRef.current = null;
        }
        
        // Clear first beep timeout if it exists
        if (firstBeepTimeoutRef.current) {
          clearTimeout(firstBeepTimeoutRef.current);
          firstBeepTimeoutRef.current = null;
        }
        
        // Only stop if still streaming
        if (isStreaming && !isStopping) {
          console.log('Calibration timer complete. Auto-stopping session.');
          // Use setTimeout to ensure this runs after the current execution context
          setTimeout(() => {
            handleStopStreaming();
          }, 0);
        }
      }
    }, 1000);
    
    console.log(`Started calibration timer for ${CALIBRATION_DURATION_MS / 1000} seconds`);
    
    // Track all timeout IDs for potential cleanup
    const timeoutIds: number[] = [];
    
    // Schedule beeps only if an interval is defined
    if (BEEP_INTERVAL_MS) {
        console.log(`Scheduling beeps every ${BEEP_INTERVAL_MS}ms`);
        const scheduleAllBeeps = () => {
          // Add a little buffer time to ensure the first beep sequence timing is accurate
          const scheduleStartTime = Date.now() + 50; // Add 50ms buffer to account for scheduling delays
          
          // Calculate the total number of beeps needed
          const totalBeepsNeeded = Math.floor(CALIBRATION_DURATION_MS / BEEP_INTERVAL_MS);
          console.log(`Scheduling ${totalBeepsNeeded} beep sequences`);
          
          // Schedule all the beeps at once with precise timing
          for (let i = 0; i < totalBeepsNeeded; i++) {
            // Calculate the exact time for this beep from the start
            const beepTime = scheduleStartTime + (i + 1) * BEEP_INTERVAL_MS;
            const readyTime = beepTime - 1000; // 1 second before the beep
            const setTime = beepTime - 500; // 0.5 seconds before the beep
            
            // Log the first few beeps for debugging
            if (i < 3) {
              const nowMs = Date.now();
              console.log(`Scheduling beep ${i+1}: Ready at +${readyTime - nowMs}ms, Set at +${setTime - nowMs}ms, Go at +${beepTime - nowMs}ms`);
            }
            
            // Schedule the Ready beep
            const readyTimeoutId = window.setTimeout(() => {
               if (!isStopping && mainBeepAudioRef.current) { playReadyBeep(); }
            }, readyTime - Date.now());
            timeoutIds.push(readyTimeoutId);
            
            // Schedule the Set beep
            const setTimeoutId = window.setTimeout(() => {
               if (!isStopping && mainBeepAudioRef.current) { playSetBeep(); }
            }, setTime - Date.now());
            timeoutIds.push(setTimeoutId);
            
            // Schedule the Go beep
            const goTimeoutId = window.setTimeout(() => {
              if (!isStopping && mainBeepAudioRef.current) { playMainBeep(); }
            }, beepTime - Date.now());
            timeoutIds.push(goTimeoutId);
          }
        };
        
        // Store the timeouts in our component-level ref
        timeoutIdsRef.current = timeoutIds;
        
        // Start scheduling all the beeps
        scheduleAllBeeps();
    } else {
        console.log('No beep interval defined for this session type.');
    }
      
    // Return a cleanup function that will be called when the component unmounts
    // or when the session is stopped
    return () => {
      // Clear all timeout IDs
      timeoutIdsRef.current.forEach(id => window.clearTimeout(id));
      timeoutIdsRef.current = [];
    };
  };
  
  // Monitor calibration timer
  useEffect(() => {
    // If timer reaches zero and we're still streaming, stop the session
    if (calibrationTimeRemaining === 0 && isStreaming && !isStopping) {
      console.log('Calibration timer reached zero. Auto-stopping session.');
      handleStopStreaming();
    }
  }, [calibrationTimeRemaining, isStreaming, isStopping]);
  
  // Format the remaining time as MM:SS
  const formatRemainingTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };
  
  // Updated handleStartStreaming function
  async function handleStartStreaming() {
    if (!isConnected) {
      console.error('Cannot start streaming: not connected to device');
      return;
    }

    // Reset session ID state
    setCurrentSessionId(null); 

    // 1. Validate session name
    const isNameValid = await validateSessionName();
    if (!isNameValid) {
      return;
    }
    setSessionNameError('');

    // 2. Validate all player/device mappings
    let allMappingsValid = true;
    const currentErrors = Array(playerDeviceMappings.length).fill('');
    playerDeviceMappings.forEach((mapping, index) => {
      if (!mapping.playerId) {
        currentErrors[index] = 'Player required';
        allMappingsValid = false;
      }
      if (!mapping.deviceId) {
        currentErrors[index] = currentErrors[index] ? currentErrors[index] + ', Device required' : 'Device required';
        allMappingsValid = false;
      }
    });
    setPlayerDeviceErrors(currentErrors);

    if (!allMappingsValid) {
      console.error('Player/Device mappings are incomplete.');
      toast.error('Please select a player and device for each required input.'); // User feedback
      return;
    }

    // 3. Create Session via service
    try {
      console.log('Creating session with name:', sessionName, 'Type:', sessionType);
      const session = await createSession({
        name: sessionName.trim(), // Trim name
        session_type: sessionType,
        start_time: new Date().toISOString(),
      }, true); // Skip name check as we did it already

      if (!session.id) {
        throw new Error("Session creation failed to return an ID.");
      }
      const newSessionId = session.id;
      setCurrentSessionId(newSessionId); 
      console.log('Session created:', session);

      // 4. Add Player Mappings via service
      console.log('Adding player mappings:', playerDeviceMappings);
      await addSessionPlayersBatch(newSessionId, playerDeviceMappings);
      console.log('Player mappings added successfully.');

      // 4.5 Initialize the local SessionManager for this session
      if (sessionManagerRef.current) {
         sessionManagerRef.current.initializeSession(newSessionId, playerDeviceMappings);
         const startTime = sessionManagerRef.current.sessionStartTime;
         setSessionStartTime(startTime); // Set state
         sessionStartTimeRef.current = startTime; // Set ref
         console.log('[handleStartStreaming] Session Manager Start Time:', startTime);
         // Use a timeout to check state after potential batching
         setTimeout(() => {
           console.log('[handleStartStreaming] Local sessionStartTime State:', sessionStartTime);
         }, 0);
      } else {
         console.error("Session Manager not initialized!");
         throw new Error("Session Manager failed to initialize.");
      }
      
      // 5. Send start command to device(s)
      console.log('Sending start commands to devices:', playerDeviceMappings);
      for (const mapping of playerDeviceMappings) {
        if (mapping.deviceId) {
           await sendCommand(`${mapping.deviceId}:start`); 
        } else {
           console.warn('Skipping start command for mapping without device ID:', mapping);
        }
      }
      console.log('Sent start commands.')

      // 6. Update UI State
      console.log('Setting isStreaming to true in handleStartStreaming...');
      setIsStreaming(true);
      console.log('Streaming started (UI state).');
      setIsStopping(false); // Ensure beeps can play
      setParsedSensorData({}); // Clear old graph data (now an object)
      setLiveDuration(0); // Reset live stats
      setLiveDataPoints(0); // Reset live stats

      // TODO: Start updated calibration timer logic
      timeoutCleanupRef.current = startCalibrationTimerIfNeeded();
      console.log("Started calibration timer/beep logic if applicable.");

      // Start interval timer to update live stats
      if (liveStatsIntervalRef.current) clearInterval(liveStatsIntervalRef.current);
      liveStatsIntervalRef.current = setInterval(() => {
        const startTimeFromRef = sessionStartTimeRef.current;
        // console.log('[Interval] Timer fired. StartTime from Ref:', startTimeFromRef);
        if (sessionManagerRef.current && startTimeFromRef) { // Check ref
          // Calculate duration based on local start time
          const durationSeconds = (Date.now() - startTimeFromRef) / 1000;
          // Get cumulative points received from manager
          const points = sessionManagerRef.current.getTotalPointsReceived();
          // console.log('[Interval] Calculated Duration:', durationSeconds, 'Calculated Points:', points);
          setLiveDuration(durationSeconds);
          setLiveDataPoints(points);
        } else {
          // console.log('[Interval] Condition failed (sessionManagerRef or startTimeFromRef missing).');
        }
      }, 100); // Update every 100ms for smoother duration

    } catch (error) {
      console.error('Error starting session or mappings:', error);
      toast.error(`Failed to start session: ${error instanceof Error ? error.message : 'Unknown error'}`);
      setCurrentSessionId(null); // Clear session ID on error
      setIsStreaming(false); // Ensure UI reflects failure
      // Clear stats interval on failure
      if (liveStatsIntervalRef.current) {
        clearInterval(liveStatsIntervalRef.current);
        liveStatsIntervalRef.current = null;
      }
      setLiveDuration(0);
      setLiveDataPoints(0);
      setSessionStartTime(null); // Reset local start time
      sessionStartTimeRef.current = null; // Reset ref
    }
  }
  
  const handleStopStreaming = async () => {
    if (!isConnected || isStopping) {
      // Check isStopping here too
      console.log(`Stop requested but already stopping (${isStopping}) or not connected (${!isConnected})`);
      return;
    }

    let stopError: Error | null = null; 
    const intendedEndTime = new Date(); // Record end time *before* async operations
    
    try {
      setIsStopping(true);
      setIsRedirecting(true); // Show saving screen immediately
      console.log('HANDLE STOP: Stopping UI & Clearing Timers...');
      // Immediately stop all audio playback
      if (mainBeepAudioRef.current) {
        mainBeepAudioRef.current.pause();
        mainBeepAudioRef.current.currentTime = 0;
      }
      
      if (readyBeepAudioRef.current) {
        readyBeepAudioRef.current.pause();
        readyBeepAudioRef.current.currentTime = 0;
      }
      
      if (setBeepAudioRef.current) {
        setBeepAudioRef.current.pause();
        setBeepAudioRef.current.currentTime = 0;
      }
      
      // Clear calibration timer if it exists
      if (calibrationTimerRef.current) {
        console.log('Clearing calibration timer');
        clearInterval(calibrationTimerRef.current);
        calibrationTimerRef.current = null;
        setCalibrationTimeRemaining(null);
      }
      
      // Clear beep timer if it exists
      if (beepTimerRef.current) {
        console.log('Clearing beep timer');
        clearInterval(beepTimerRef.current);
        beepTimerRef.current = null;
      }
      
      // Clear first beep timeout if it exists
      if (firstBeepTimeoutRef.current) {
        console.log('Clearing first beep timeout');
        clearTimeout(firstBeepTimeoutRef.current);
        firstBeepTimeoutRef.current = null;
      }
      
      // Clear all scheduled setTimeout beeps
      // This will call the cleanup function returned by startCalibrationTimerIfNeeded
      if (timeoutCleanupRef.current) {
        console.log('Clearing all scheduled beep timeouts');
        timeoutCleanupRef.current();
        timeoutCleanupRef.current = null;
      }
      
      console.log('HANDLE STOP: Sending stop command to device(s)...');
      // Send stop command to each device
      for (const mapping of playerDeviceMappings) {
        if (mapping.deviceId) {
           await sendCommand(`${mapping.deviceId}:stop`);
        } else {
           console.warn('Skipping stop command for mapping without device ID:', mapping);
        }
      }
      console.log('Sent stop commands.');
      
      // End the session in SessionManager (flushes final data)
      console.log('HANDLE STOP: Ending session manager, passing intended end time:', intendedEndTime.toISOString());
      if (sessionManagerRef.current) {
        try {
            // Pass intendedEndTime to endSession
            await sessionManagerRef.current.endSession(intendedEndTime);
            console.log("HANDLE STOP: SessionManager ended session successfully.");
        } catch (smError) {
            console.error("HANDLE STOP: Error ending session in SessionManager:", smError);
            stopError = smError instanceof Error ? smError : new Error(String(smError));
            toast.error(`Error saving session data: ${stopError.message}`);
        }
      } else {
          console.warn("HANDLE STOP: Session Manager ref was null.");
      }

      // Stop recording context (if needed - currently seems unused here)
      // await stopRecording();
      
      // Stop streaming UI state and clear local UI data
      console.log('HANDLE STOP: Resetting UI state...');
      setIsStreaming(false);
      setSensorData([]);
      setParsedSensorData({}); // Reset graph data object
      setSessionName('');
      setLiveDuration(0); // Also reset live stats display
      setLiveDataPoints(0);
      setSessionStartTime(null); // Reset local start time
      sessionStartTimeRef.current = null; // Reset ref
      
      console.log('HANDLE STOP: Session processing complete.');
      
      // Add a delay before redirecting ONLY IF no error occurred
      if (!stopError) {
          console.log('HANDLE STOP: Waiting briefly before redirect...');
          await new Promise(resolve => setTimeout(resolve, 5000)); // Reduced delay to 5 seconds
          
          // Redirect ONLY if no error occurred
          console.log('HANDLE STOP: Redirecting...', currentSessionId); 
          if (currentSessionId) {
              window.location.href = `/sessions/${currentSessionId}`;
          } else {
              console.warn('No current session ID available to redirect to.');
              window.location.href = '/devices'; // Fallback
          }
      } else {
          console.log('HANDLE STOP: Error occurred, not redirecting automatically.');
          // Keep the user on the page to see the error state/logs
          setIsRedirecting(false); // Ensure loading spinner stops
      }
      
    } catch (error) {
       console.error('HANDLE STOP: Unexpected error in stop handler:', error);
       stopError = error instanceof Error ? error : new Error(String(error));
       toast.error(`Error stopping session: ${stopError.message}`);
       setIsRedirecting(false); // Ensure loading spinner stops
    } finally {
       console.log(`HANDLE STOP: Final cleanup. Error occurred: ${!!stopError}`);
       setIsStopping(false); // Allow trying again if needed
       // Don't clear currentSessionId if there was an error, maybe needed for retry?
       if (!stopError) { 
          setCurrentSessionId(null); 
       }
       setSessionStartTime(null); // Reset local start time
       sessionStartTimeRef.current = null; // Reset ref
       // Do NOT clear currentSessionId here if an error happened
       // so the user can potentially retry or see which session failed
    }
  };
  
  async function handleReset() {
    if (!isConnected) {
      return;
    }
    
    try {
      await sendCommand('reset');
      setSensorData([]);
      setParsedSensorData({});
    } catch (error) {
      console.error('Error resetting device:', error);
    }
  }
  
  // Reset all state when component mounts or when navigating to the page
  useEffect(() => {
    setIsStreaming(false);
    setSensorData([]);
    setParsedSensorData({}); // Reset graph data object
    setSessionName('');
    setSessionNameError('');
    setIsStopping(false);
    setIsRedirecting(false);
    setLiveDuration(0);
    setLiveDataPoints(0);
    setSessionStartTime(null); // Reset local start time
    sessionStartTimeRef.current = null; // Reset ref
    // Ensure interval is cleared on initial load/navigation
    if (liveStatsIntervalRef.current) {
        clearInterval(liveStatsIntervalRef.current);
        liveStatsIntervalRef.current = null;
    }
  }, []);
  
  // Cleanup interval on unmount
  useEffect(() => {
      return () => {
          if (liveStatsIntervalRef.current) {
              clearInterval(liveStatsIntervalRef.current);
              liveStatsIntervalRef.current = null;
              console.log('Cleaned up live stats interval on unmount.');
          }
          // Reset start time on unmount as well
          setSessionStartTime(null);
          sessionStartTimeRef.current = null; // Reset ref
      };
  }, []); // Empty dependency array ensures this runs only on mount and unmount

  // Available device IDs
  const deviceIds = ['1', '2', '3', '4', '5'];
  
  return (
    <div className="h-[calc(100vh-65px)] flex items-center justify-center p-4">
      <Card className={`mx-auto transition-all duration-300 ${isStreaming ? 'w-full' : 'max-w-md w-full'}`}>
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle>Device Connection</CardTitle>
          <Badge className={isConnected ? "bg-green-500" : "bg-red-500"}>
            {status}
          </Badge>
        </CardHeader>
        <CardContent 
          className={`space-y-6 ${isStreaming ? 'overflow-y-auto max-h-[calc(100vh-250px)]' : ''}`}
        >
          {isRedirecting ? (
            <div className="flex flex-col items-center justify-center py-8">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mb-4"></div>
              <p className="text-sm text-muted-foreground">Saving session data...</p>
            </div>
          ) : !isConnected ? (
            <>
              <div className="flex items-center justify-center p-6">
                <Usb className="h-16 w-16 text-muted-foreground" />
              </div>
              <p className="text-center text-sm text-muted-foreground">
                Connect your ESP32 device to start collecting data for the current session.
              </p>
              <div className="text-xs text-muted-foreground mb-4">
                <p>Make sure your device is:</p>
                <ul className="list-disc pl-5 mt-2">
                  <li>Plugged into your computer</li>
                  <li>Has the correct firmware installed</li>
                  <li>Not being used by another application</li>
                </ul>
              </div>
              <Button 
                className="w-full" 
                onClick={connectSerial}
              >
                Connect Device
              </Button>
            </>
          ) : (
            <div className="space-y-6">
              {!isStreaming && (
                <div className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="sessionName">Session Name</Label>
                    <Input
                      id="sessionName"
                      value={sessionName}
                      onChange={(e) => {
                        setSessionName(e.target.value);
                        setSessionNameError('');
                      }}
                      placeholder="Enter a name for this session"
                      className={sessionNameError ? 'border-red-500' : ''}
                    />
                    {sessionNameError && (
                      <p className="text-sm text-red-500">{sessionNameError}</p>
                    )}
                  </div>
                  
                  <div className="space-y-2">
                    <Label htmlFor="sessionType">Session Type</Label>
                    <Select
                      value={sessionType}
                      onValueChange={(value) => setSessionType(value as SessionType)}
                    >
                      <SelectTrigger id="sessionType">
                        <SelectValue placeholder="Select session type" />
                      </SelectTrigger>
                      <SelectContent>
                        {ALL_SESSION_TYPES.map(type => (
                           <SelectItem key={type} value={type}>
                             {formatSessionType(type)}
                           </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                  
                  {/* Groundball Player Number Selector */}
                  {sessionType === 'groundball_calibration' && (
                    <div className="space-y-2">
                      <Label htmlFor="groundballPlayers">Number of Players (Groundball Calibration)</Label>
                      <Select
                        value={String(numberOfGroundballPlayers)}
                        onValueChange={(value) => setNumberOfGroundballPlayers(parseInt(value, 10))}
                      >
                        <SelectTrigger id="groundballPlayers">
                          <SelectValue placeholder="Select number of players" />
                        </SelectTrigger>
                        <SelectContent>
                          {[1, 2, 3, 4, 5].map(num => (
                            <SelectItem key={num} value={String(num)}>
                              {num} Player{num > 1 ? 's' : ''}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                  )}
                  
                  {/* Session Type Details Display */}
                  {(sessionType.includes('calibration') || sessionType === 'pass_catch_calibration' || sessionType === 'shot_calibration') && (
                     <div className="p-3 border rounded bg-muted/20 text-sm text-muted-foreground space-y-1">
                        <p className="font-medium text-foreground">Session Details:</p>
                        {/* Display Duration/Beeps for Calibration Types */}
                        {CALIBRATION_CONFIG[sessionType] && (
                            <> 
                                <p>Duration: {CALIBRATION_CONFIG[sessionType]?.durationMinutes} minutes</p>
                                {CALIBRATION_CONFIG[sessionType]?.beepIntervalSeconds !== null ? (
                                    <p>Beep Interval: Every {CALIBRATION_CONFIG[sessionType]?.beepIntervalSeconds} seconds</p>
                                ) : (
                                    <p>Beep Interval: None</p>
                                )}
                            </>
                        )}
                        
                        {/* Role Assignment Logic */}
                        {(sessionType === 'pass_catch_calibration' || sessionType === 'shot_calibration') && playerDeviceMappings.length === 2 && (
                            () => {
                                // Ensure device IDs are valid numbers for comparison
                                const id1 = parseInt(playerDeviceMappings[0].deviceId || '-1', 10);
                                const id2 = parseInt(playerDeviceMappings[1].deviceId || '-1', 10);
                                const player1Name = players.find(p => p.id === playerDeviceMappings[0].playerId)?.name || 'Player 1';
                                const player2Name = players.find(p => p.id === playerDeviceMappings[1].playerId)?.name || 'Player 2';

                                if (id1 === -1 || id2 === -1) {
                                    return <p>Assign Device IDs to determine roles.</p>;
                                }

                                const playerWithLowerId = id1 < id2 ? player1Name : player2Name;
                                const playerWithHigherId = id1 < id2 ? player2Name : player1Name;
                                const lowerDeviceId = id1 < id2 ? id1 : id2;
                                const higherDeviceId = id1 < id2 ? id2 : id1;

                                if (sessionType === 'pass_catch_calibration') {
                                    return (
                                        <>
                                            <p>{playerWithLowerId} (Device {lowerDeviceId}) is First Thrower.</p>
                                            <p>{playerWithHigherId} (Device {higherDeviceId}) is First Catcher.</p>
                                        </>
                                    );
                                }
                                if (sessionType === 'shot_calibration') {
                                    return (
                                        <>
                                            <p>{playerWithLowerId} (Device {lowerDeviceId}) is the Shooter.</p>
                                            <p>{playerWithHigherId} (Device {higherDeviceId}) is the Goalie.</p>
                                        </>
                                    );
                                }
                                return null; // Should not happen
                            }
                        )()}{/* End Role Assignment Logic */}
                     </div>
                  )}
                  
                  {/* Dynamic Player/Device Inputs */}
                  {playerDeviceMappings.map((mapping, index) => (
                    <div key={index} className="p-3 border rounded space-y-3 bg-muted/40">
                       <Label className="font-semibold">Player {index + 1}</Label>
                       {playerDeviceErrors[index] && (
                           <p className="text-sm text-red-500">Error: {playerDeviceErrors[index]}</p>
                       )}
                       <div className="grid grid-cols-2 gap-4">
                         <div className="space-y-1">
                            <Label htmlFor={`deviceId-${index}`}>Device ID</Label>
                            <Select
                              value={mapping.deviceId}
                              onValueChange={(value) => updatePlayerDeviceMapping(index, 'deviceId', value)}
                            >
                              <SelectTrigger id={`deviceId-${index}`} className={playerDeviceErrors[index]?.includes('Device') ? 'border-red-500' : ''}>
                                <SelectValue placeholder="Select device" />
                              </SelectTrigger>
                              <SelectContent>
                                {deviceIds.map(id => (
                                  <SelectItem key={id} value={id}>
                                    Device {id}
                                  </SelectItem>
                                ))}
                              </SelectContent>
                            </Select>
                         </div>
                         <div className="space-y-1">
                            <Label htmlFor={`playerSelect-${index}`}>Player Profile</Label>
                            <Select
                              value={mapping.playerId}
                              onValueChange={(value) => updatePlayerDeviceMapping(index, 'playerId', value)}
                            >
                              <SelectTrigger id={`playerSelect-${index}`} className={playerDeviceErrors[index]?.includes('Player') ? 'border-red-500' : ''}>
                                <SelectValue placeholder="Select player" />
                              </SelectTrigger>
                              <SelectContent>
                                {playersLoading ? (
                                  <SelectItem value="loading" disabled>Loading...</SelectItem>
                                ) : players.length === 0 ? (
                                  <SelectItem value="none" disabled>No players</SelectItem>
                                ) : (
                                  players.map(player => (
                                    <SelectItem key={player.id} value={player.id}>
                                      {player.name}
                                    </SelectItem>
                                  ))
                                )}
                              </SelectContent>
                            </Select>
                         </div>
                       </div>
                    </div>
                  ))}
                  
                  <div className="flex gap-2">
                    <Button
                      className="flex-1"
                      onClick={handleStartStreaming}
                      // Disable if name missing or any mapping is incomplete
                      disabled={!sessionName.trim() || playerDeviceMappings.some(m => !m.playerId || !m.deviceId) || playersLoading || !isConnected}
                    >
                      <Play className="mr-2 h-4 w-4" />
                      Start Session
                    </Button>
                    <Button
                      variant="outline"
                      onClick={handlePairDevices}
                      // Disable if any required device ID is missing
                      disabled={playerDeviceMappings.some(m => !m.deviceId) || !isConnected}
                    >
                      <Bluetooth className="mr-2 h-4 w-4" />
                      Pair
                    </Button>
                    <Button
                      variant="outline"
                      onClick={disconnectSerial}
                    >
                      Disconnect
                    </Button>
                  </div>
                </div>
              )}
              
              {isStreaming && (
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="font-medium">{sessionName}</h3>
                      <p className="text-sm text-muted-foreground">
                        Duration: {formatDuration(liveDuration)}
                      </p>
                      <p className="text-sm text-muted-foreground">
                        Data Points: {liveDataPoints}
                      </p>
                      {sessionType.includes('calibration') && calibrationTimeRemaining !== null && (
                        <p className="text-sm font-medium text-orange-500">
                          Auto-stop in: {formatRemainingTime(calibrationTimeRemaining)}
                        </p>
                      )}
                    </div>
                    <div className="flex gap-2">
                      <Button
                        variant="destructive"
                        onClick={handleStopStreaming}
                        disabled={isStopping}
                      >
                        <Pause className="mr-2 h-4 w-4" />
                        Stop Session
                      </Button>
                      <Button
                        variant="outline"
                        onClick={handleReset}
                      >
                        <RefreshCw className="mr-2 h-4 w-4" />
                        Reset
                      </Button>
                    </div>
                  </div>
                  
                  {/* Render a graph for each player/device mapping */}
                  <div className="space-y-4">
                    {playerDeviceMappings.map((mapping, index) => {
                      const player = players.find(p => p.id === mapping.playerId);
                      const playerName = player ? player.name : 'Unknown Player';
                      const deviceData = parsedSensorData[mapping.deviceId] || [];
                      const graphTitle = `${playerName} (Device ${mapping.deviceId})`;

                      return (
                        <div key={mapping.deviceId || index} className="p-3 border rounded bg-muted/20">
                          <h4 className="text-sm font-semibold mb-2">{graphTitle}</h4>
                          <LiveDataGraph
                            data={deviceData}
                            maxPoints={100} 
                            // Pass title if LiveDataGraph supports it, otherwise handled by h4 above
                            // title={graphTitle} 
                          />
                        </div>
                      );
                    })}
                  </div>
                  {/* <LiveDataGraph data={parsedSensorData} maxPoints={100} /> Remove the single graph */}
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
} 