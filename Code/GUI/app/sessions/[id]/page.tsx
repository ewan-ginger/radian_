"use client";

import React, { useEffect, useState, useMemo } from 'react';
import { useParams } from 'next/navigation';
import { MainLayout } from '@/components/layout/MainLayout';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { OrientationGraph } from '@/components/data/OrientationGraph';
import { AccelerationGraph } from '@/components/data/AccelerationGraph';
import { GyroscopeGraph } from '@/components/data/GyroscopeGraph';
import { MagnetometerGraph } from '@/components/data/MagnetometerGraph';
import { Badge } from '@/components/ui/badge';
import { Button } from "@/components/ui/button";
import { Activity, Calendar, Clock, Database, User, Wifi, Users, BarChart2, Maximize, ZoomOut, ChevronDown } from 'lucide-react';
import { useSessionData } from '@/hooks/useSessionData';
import { getSensorDataBySession } from '@/lib/services/sensor-data-service';
import { getSessionPlayersBySessionId } from '@/lib/services/session-player-service';
import { getTrainingDataBySession } from '@/lib/services/training-data-service';
import { SensorDataEntity, SessionPlayerEntity, SessionEntity, SessionType, getRequiredPlayers, TrainingSensorDataEntity } from '@/types/database.types';
import { formatDistanceToNow, format } from 'date-fns';
import { formatSessionType } from '@/lib/utils';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { LacrosseStickAnimationLoader } from '@/components/data/LacrosseStickAnimationLoader';

interface ActionGroup {
  label: string;
  startTime: number; // Relative seconds from session start
  endTime: number; // Relative seconds from session start
  metric: number | null;
  count: number; // Number of data points in the group
  playerName: string | null;
  playerId: string | null;
}

export default function SessionDetailPage() {
  const params = useParams();
  const sessionId = params.id as string;
  const { sessions } = useSessionData();
  const [session, setSession] = useState<SessionEntity | null>(null);
  const [sessionPlayers, setSessionPlayers] = useState<SessionPlayerEntity[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [sensorData, setSensorData] = useState<SensorDataEntity[]>([]);
  const [trainingData, setTrainingData] = useState<TrainingSensorDataEntity[]>([]);
  const [dataType, setDataType] = useState('orientation');
  
  const [graphStartTime, setGraphStartTime] = useState<number | null>(null);
  const [graphEndTime, setGraphEndTime] = useState<number | null>(null);
  const [openActionId, setOpenActionId] = useState<string | null>(null); // State to track the currently open action
  
  const isCalibrationSession = useMemo(() => {
    if (!session?.session_type) return false;
    return session.session_type.toLowerCase().includes('calibration');
  }, [session]);

  useEffect(() => {
    const fetchPageData = async () => {
      if (!sessionId) return;
      setIsLoading(true);
      setGraphStartTime(null);
      setGraphEndTime(null);
      try {
        const sessionData = sessions.find(s => s.id === sessionId);
        setSession(sessionData || null);
        
        if (sessionData) {
          const playersData = await getSessionPlayersBySessionId(sessionId);
          setSessionPlayers(playersData);
          console.log('Fetched Session Players:', playersData);
          
          const rawSensorData = await getSensorDataBySession(sessionId);
          console.log('Fetched raw sensor data:', {
            count: rawSensorData.length,
            firstFew: rawSensorData.slice(0, 3),
            lastFew: rawSensorData.slice(-3)
          });
          setSensorData(rawSensorData);

          if (sessionData.session_type?.toLowerCase().includes('calibration')) {
            console.log('Calibration session detected, fetching training data...');
            const labeledData = await getTrainingDataBySession(sessionId);
            console.log('Fetched training data:', {
              count: labeledData.length,
              firstFew: labeledData.slice(0, 3),
              lastFew: labeledData.slice(-3)
            });
            setTrainingData(labeledData);
          } else {
            console.log('Not a calibration session, skipping training data fetch.');
            setTrainingData([]);
          }

        } else {
            console.warn(`Session ${sessionId} not found in useSessionData list.`);
            setSessionPlayers([]);
            setSensorData([]);
            setTrainingData([]);
        }

      } catch (error) {
        console.error('Error fetching session page data:', error);
        setSession(null);
        setSessionPlayers([]);
        setSensorData([]);
        setTrainingData([]);
      } finally {
        setIsLoading(false);
      }
    };
    
    if (sessionId && sessions.length > 0) {
      fetchPageData();
    } else if (sessionId && sessions.length === 0) {
        console.log('Sessions list empty, page might load incompletely.');
    }
  }, [sessionId, sessions]);
  
  const formatDuration = (duration: string | null) => {
    if (!duration) {
      if (session) {
        const startTime = new Date(session.start_time);
        const endTime = session.end_time ? new Date(session.end_time) : new Date();
        const durationSeconds = Math.floor((endTime.getTime() - startTime.getTime()) / 1000);
        
        const hours = Math.floor(durationSeconds / 3600);
        const minutes = Math.floor((durationSeconds % 3600) / 60);
        const seconds = durationSeconds % 60;
        
        if (hours > 0) {
          return `${hours} hours ${minutes} minutes`;
        } else if (minutes > 0) {
          return `${minutes} minutes ${seconds} seconds`;
        } else {
          return `${seconds} seconds`;
        }
      }
      return 'N/A';
    }
    
    const timeFormatMatch = duration.match(/^(\d{2}):(\d{2}):(\d{2})$/);
    if (timeFormatMatch) {
      const hours = parseInt(timeFormatMatch[1]);
      const minutes = parseInt(timeFormatMatch[2]);
      const seconds = parseInt(timeFormatMatch[3]);
      
      if (hours > 0) {
        return `${hours} hours ${minutes} minutes`;
      } else if (minutes > 0) {
        return `${minutes} minutes ${seconds} seconds`;
      } else {
        return `${seconds} seconds`;
      }
    }
    
    try {
      const hours = duration.match(/(\d+)\s+hour/i)?.[1] || '0';
      const minutes = duration.match(/(\d+)\s+minute/i)?.[1] || '0';
      const seconds = duration.match(/(\d+)\s+second/i)?.[1] || '0';
      
      if (parseInt(hours) > 0) {
        return `${hours} hours ${minutes} minutes`;
      } else if (parseInt(minutes) > 0) {
        return `${minutes} minutes ${seconds} seconds`;
      } else {
        return `${seconds} seconds`;
      }
    } catch (e) {
      if (typeof duration === 'string' && !isNaN(Number(duration))) {
        const totalSeconds = parseInt(duration);
        const hours = Math.floor(totalSeconds / 3600);
        const minutes = Math.floor((totalSeconds % 3600) / 60);
        const seconds = totalSeconds % 60;
        
        if (hours > 0) {
          return `${hours} hours ${minutes} minutes`;
        } else if (minutes > 0) {
          return `${minutes} minutes ${seconds} seconds`;
        } else {
          return `${seconds} seconds`;
        }
      }
      
      return 'N/A';
    }
  };
  
  const dataByDevice = useMemo(() => {
    const groupedData: { [deviceId: string]: SensorDataEntity[] } = {};
    
    sessionPlayers.forEach(player => {
        if (player.device_id) {
            groupedData[player.device_id] = [];
        }
    });

    sensorData.forEach(item => {
      if (item.device_id && groupedData[item.device_id]) {
        groupedData[item.device_id].push(item);
      }
    });

    console.log("Grouped sensor data by device:", groupedData);
    return groupedData;
  }, [sensorData, sessionPlayers]);

  const processedActions = useMemo(() => {
    if (!trainingData || trainingData.length === 0) {
      return [];
    }

    // 1. Create Player ID -> Name lookup map
    const playerMap = new Map<string, string>();
    sessionPlayers.forEach(p => {
      if (p.player_id && p.playerName) {
        playerMap.set(p.player_id, p.playerName);
      }
    });

    // 2. Group trainingData by player_id (including null player_id)
    const dataByPlayer = new Map<string | null, TrainingSensorDataEntity[]>();
    // Ensure data is sorted by timestamp primarily BEFORE grouping
    const sortedTrainingData = [...trainingData].sort((a,b) => (a.timestamp ?? 0) - (b.timestamp ?? 0));

    sortedTrainingData.forEach(point => {
      const key = point.player_id; // Use player_id directly (can be null)
      if (!dataByPlayer.has(key)) {
        dataByPlayer.set(key, []);
      }
      // Since data is pre-sorted, pushing maintains order within each player group
      dataByPlayer.get(key)?.push(point);
    });

    let allActionGroups: ActionGroup[] = [];

    // 3. Process each player's data group
    dataByPlayer.forEach((playerData, pId) => {
      // Get player name once for this group
      const groupPlayerName = pId ? playerMap.get(pId) || `Player ID: ${pId.substring(0, 6)}` : 'Unknown Player';
      
      const playerGroups: ActionGroup[] = [];
      let currentGroup: ActionGroup | null = null;

      // Iterate through this player's sorted data
      for (const point of playerData) {
        const currentTimestamp = point.timestamp ?? 0;

        if (point.label === null || point.label === undefined) {
          // Null label: finalize current group for this player
          if (currentGroup !== null) {
            playerGroups.push(currentGroup);
            currentGroup = null; 
          }
          continue; // Move to next point for this player
        }

        // Non-null label for this player
        if (currentGroup === null) {
          // Start a new group for this player
          currentGroup = {
            label: point.label,
            startTime: currentTimestamp,
            endTime: currentTimestamp,
            metric: point.metric, 
            count: 1,
            playerName: groupPlayerName, // Assign the determined player name
            playerId: pId // Assign playerId
          };
        } else if (point.label === currentGroup.label) {
          // Continue the current group for this player
          currentGroup.endTime = currentTimestamp;
          currentGroup.count += 1;
          // metric is taken from the first point of the group
        } else {
          // Label changed for this player
          // Finalize the previous group and start a new one
          playerGroups.push(currentGroup);
          currentGroup = {
            label: point.label,
            startTime: currentTimestamp,
            endTime: currentTimestamp,
            metric: point.metric,
            count: 1,
            playerName: groupPlayerName, // Assign the determined player name
            playerId: pId // Assign playerId
          };
        }
      } // End loop through this player's points

      // Add the last group for this player if it exists
      if (currentGroup) {
        playerGroups.push(currentGroup);
      }
      
      // Add this player's processed groups to the combined list
      allActionGroups = allActionGroups.concat(playerGroups);

    }); // End loop through each player's data

    // 4. Sort combined results by start time
    allActionGroups.sort((a, b) => a.startTime - b.startTime);

    console.log("Processed Action Groups (Per Player, Sorted):", allActionGroups);
    return allActionGroups;
    
  }, [trainingData, sessionPlayers]);

  const filteredDataByDevice = useMemo(() => {
    const groupedData: { [deviceId: string]: SensorDataEntity[] } = {};
    
    sessionPlayers.forEach(player => {
        if (player.device_id) {
            groupedData[player.device_id] = [];
        }
    });

    sensorData.forEach(item => {
      if (item.device_id && groupedData[item.device_id]) {
        const isWithinRange = 
          (graphStartTime === null || (item.timestamp ?? -Infinity) >= graphStartTime) &&
          (graphEndTime === null || (item.timestamp ?? Infinity) <= graphEndTime);

        if (isWithinRange) {
            groupedData[item.device_id].push(item);
        }
      }
    });

    console.log("Filtered & Grouped sensor data by device:", groupedData);
    return groupedData;
  }, [sensorData, sessionPlayers, graphStartTime, graphEndTime]);

  const transformDataForGraph = (data: SensorDataEntity[], type: string) => {
    if (data.length > 0) {
      console.log(`First few ${type} data points:`, data.slice(0, 3));
    } else {
      console.log(`No ${type} data points available`);
      return [];
    }

    const validData = data.filter(item => 
      item.timestamp !== undefined && 
      item.timestamp !== null
    );
    
    if (validData.length === 0) {
      console.log(`No valid ${type} data points with timestamps`);
      return [];
    }
    
    console.log(`Valid ${type} data points: ${validData.length} out of ${data.length}`);

    const sortedData = [...validData].sort((a, b) => {
      const aTime = typeof a.timestamp === 'number' ? a.timestamp : Number(a.timestamp);
      const bTime = typeof b.timestamp === 'number' ? b.timestamp : Number(b.timestamp);
      return aTime - bTime;
    });
    
    const transformedData = sortedData.map((item) => {
      const timestamp = typeof item.timestamp === 'number' 
        ? item.timestamp 
        : Number(item.timestamp);
      
      switch (type) {
        case 'orientation': return { timestamp, x: item.orientation_x || 0, y: item.orientation_y || 0, z: item.orientation_z || 0 };
        case 'accelerometer': return { timestamp, x: item.accelerometer_x || 0, y: item.accelerometer_y || 0, z: item.accelerometer_z || 0 };
        case 'gyroscope': return { timestamp, x: item.gyroscope_x || 0, y: item.gyroscope_y || 0, z: item.gyroscope_z || 0 };
        case 'magnetometer': return { timestamp, x: item.magnetometer_x || 0, y: item.magnetometer_y || 0, z: item.magnetometer_z || 0 };
        default: return { timestamp, x: 0, y: 0, z: 0 };
      }
    });
    
    if (transformedData.length > 0) {
      console.log(`First few transformed ${type} data points:`, transformedData.slice(0, 3));
      console.log(`Last few transformed ${type} data points:`, transformedData.slice(-3));
    }
    
    console.log(`Transformed ${type} data count: ${transformedData.length}`);
    return transformedData;
  };
  
  const handleActionClick = (action: ActionGroup) => {
      console.log(`Zooming graph to action: ${action.label} [${action.startTime.toFixed(2)}s - ${action.endTime.toFixed(2)}s]`);
      const buffer = 0.5;
      setGraphStartTime(Math.max(0, action.startTime - buffer)); 
      setGraphEndTime(action.endTime + buffer);
  };

  const handleResetZoom = () => {
      console.log("Resetting graph zoom");
      setGraphStartTime(null);
      setGraphEndTime(null);
  };
  
  if (isLoading) {
    return (
      <MainLayout>
        <div className="flex items-center justify-center h-full">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary"></div>
        </div>
      </MainLayout>
    );
  }
  
  if (!session) {
    return (
      <MainLayout>
        <div className="text-center py-12">
          <h2 className="text-2xl font-bold">Session not found</h2>
          <p className="text-muted-foreground mt-2">The session you're looking for doesn't exist or has been deleted.</p>
        </div>
      </MainLayout>
    );
  }
  
  const numberOfPlayers = sessionPlayers.length || getRequiredPlayers(session?.session_type);
  
  return (
    <MainLayout>
      <div className="container mx-auto p-4 md:p-6 lg:p-8 space-y-6">
        <Card className="overflow-hidden">
          <CardHeader className="bg-muted/30 p-4 md:p-6">
            <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-2">
              <CardTitle className="text-xl md:text-2xl font-bold tracking-tight">{session.name || `Session Details`}</CardTitle>
              <Badge variant={isCalibrationSession ? "default" : "secondary"} className="whitespace-nowrap shrink-0">
                {formatSessionType(session.session_type || 'unknown')}
              </Badge>
            </div>
            {session.name && <CardDescription className="mt-1">ID: {sessionId}</CardDescription>} 
          </CardHeader>
          <CardContent className="p-4 md:p-6 grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-4">
            <div className="space-y-3">
              <div className="flex items-center space-x-3">
                <Calendar className="h-5 w-5 text-orange-500 flex-shrink-0" />
                <div className="text-sm">
                  <span className="font-medium">Date & Time</span>
                  <p className="text-muted-foreground">{format(new Date(session.start_time), 'PPP p')}</p>
                </div>
              </div>
              <div className="flex items-center space-x-3">
                <Clock className="h-5 w-5 text-blue-500 flex-shrink-0" />
                 <div className="text-sm">
                   <span className="font-medium">Duration</span>
                   <p className="text-muted-foreground">{formatDuration(session.duration as string | null)}</p>
                 </div>
              </div>
            </div>
            
            <div className="space-y-3">
               <div className="flex items-center space-x-3">
                 <Users className="h-5 w-5 text-green-500 flex-shrink-0" />
                 <div className="text-sm">
                   <span className="font-medium">Participants</span>
                   <p className="text-muted-foreground">
                       {sessionPlayers.length} Player{sessionPlayers.length !== 1 ? 's' : ''} / Device{sessionPlayers.length !== 1 ? 's' : ''}
                   </p>
                 </div>
              </div>
               <div className="flex items-center space-x-3">
                 <Database className="h-5 w-5 text-muted-foreground flex-shrink-0" />
                 <div className="text-sm">
                   <span className="font-medium">Data Points</span>
                    <p className="text-muted-foreground">{sensorData.length} Raw</p>
                 </div>
              </div>
            </div>
          </CardContent>
           {sessionPlayers.length > 0 && (
                <CardFooter className="bg-muted/30 p-4 md:p-6 border-t">
                  <div>
                     <h4 className="text-sm font-semibold mb-2">Players & Devices:</h4>
                     <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-x-4 gap-y-1 text-sm">
                       {sessionPlayers.map(p => (
                           <div key={p.id} className="flex items-center space-x-1.5 text-muted-foreground truncate">
                               <User className="h-3.5 w-3.5 flex-shrink-0" />
                               <span className="font-medium truncate">{p.playerName || 'Unknown Player'}</span>
                               <span className="truncate text-xs">(Dev ID: {p.device_id || 'N/A'})</span>
                           </div>
                       ))}
                     </div>
                   </div>
                </CardFooter>
             )}
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-4">
            <CardTitle>Sensor Data Visualization</CardTitle>
            {graphStartTime !== null && (
                <Button variant="outline" size="sm" onClick={handleResetZoom} className="ml-auto">
                    <ZoomOut className="h-4 w-4 mr-1.5" />
                    Reset Zoom
                </Button>
            )}
          </CardHeader>
          <CardContent>
            {sensorData.length > 0 ? (
              <Tabs value={dataType} onValueChange={setDataType} className="w-full">
                <TabsList className="grid w-full grid-cols-2 md:grid-cols-4 mb-4 mt-2">
                  <TabsTrigger value="orientation">Orientation</TabsTrigger>
                  <TabsTrigger value="accelerometer">Accelerometer</TabsTrigger>
                  <TabsTrigger value="gyroscope">Gyroscope</TabsTrigger>
                  <TabsTrigger value="magnetometer">Magnetometer</TabsTrigger>
                </TabsList>
                 {Object.entries(filteredDataByDevice).map(([deviceId, deviceData]) => (
                   <div key={deviceId} className="mb-6 border rounded-lg p-4">
                    <h3 className="text-lg font-semibold mb-2 flex items-center">
                      <User className="h-4 w-4 mr-2 opacity-80"/>
                      Device ID: {deviceId}
                      {sessionPlayers.find(p => p.device_id === deviceId)?.playerName
                        ? <span className="ml-1 font-normal text-muted-foreground">({sessionPlayers.find(p => p.device_id === deviceId)?.playerName})</span>
                        : ''}
                    </h3>
                    {deviceData.length > 0 ? (
                      <>
                        <TabsContent value="orientation">
                          <OrientationGraph data={transformDataForGraph(deviceData, 'orientation')} title="Orientation (°)" />
                        </TabsContent>
                        <TabsContent value="accelerometer">
                          <AccelerationGraph data={transformDataForGraph(deviceData, 'accelerometer')} title="Accelerometer (m/s²)" />
                        </TabsContent>
                        <TabsContent value="gyroscope">
                          <GyroscopeGraph data={transformDataForGraph(deviceData, 'gyroscope')} title="Gyroscope (rad/s)" />
                        </TabsContent>
                        <TabsContent value="magnetometer">
                          <MagnetometerGraph data={transformDataForGraph(deviceData, 'magnetometer')} title="Magnetometer (μT)" />
                        </TabsContent>
                      </>
                    ) : (
                      <p className="text-muted-foreground text-center py-4 text-sm">
                         No data available for this device {graphStartTime !== null ? 'in the selected time range' : ''}.
                      </p>
                    )}
                   </div>
                 ))}
                 {Object.values(filteredDataByDevice).every(arr => arr.length === 0) && sensorData.length > 0 && (
                     <p className="text-muted-foreground text-center py-8">
                        No sensor data available {graphStartTime !== null ? 'in the selected time range' : 'for this session'}.
                     </p>
                 )}
              </Tabs>
            ) : (
              <p className="text-muted-foreground text-center py-8">No raw sensor data recorded for this session.</p>
            )}
          </CardContent>
        </Card>

        {isCalibrationSession && processedActions.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle>Detected Actions</CardTitle>
              <CardDescription>
                Click a row to view the 3D animation for that action. Or click the time interval to zoom graph.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow><TableHead className="w-[150px]">Player</TableHead><TableHead className="w-[150px]">Action</TableHead><TableHead>Time (Seconds)</TableHead><TableHead className="text-right">Metric</TableHead><TableHead className="w-[40px]"></TableHead></TableRow>
                </TableHeader>
                <TableBody>
                  {processedActions.map((action, index) => {
                    const actionKey = `${action.label}-${index}-${action.startTime}-${action.playerId || 'no-player'}`;
                    const isAnimationVisible = openActionId === actionKey;

                    const startSec = action.startTime.toFixed(2);
                    const endSec = action.endTime.toFixed(2);
                    const interval = `${startSec}s - ${endSec}s`;

                    let metricDisplay = 'N/A';
                    if (action.metric !== null) {
                      if (action.label === 'faceoff') {
                        const metricMs = (action.metric * 1000) * -1; 
                        metricDisplay = `${metricMs >= 0 ? '+' : ''}${metricMs.toFixed(0)} ms`; 
                      } else if (action.label === 'save') {
                        const metricMs = action.metric * 1;
                        metricDisplay = `${metricMs >= 0 ? '+' : ''}${metricMs.toFixed(0)} ms`; 
                      } else if (action.label === 'groundball') {
                        const metricMs = (action.metric * 1000); 
                        metricDisplay = `${metricMs.toFixed(0)} ms`;
                      }else if (action.label === 'catch') {
                        const metricM = (action.metric); 
                        metricDisplay = `${metricM.toFixed(2)} m`;
                      } else  {
                        metricDisplay = `${action.metric.toFixed(2)} mph`;
                      }
                    }

                    const toggleAnimation = () => {
                        setOpenActionId(isAnimationVisible ? null : actionKey);
                    };

                    return (
                      <React.Fragment key={actionKey}>
                        <TableRow 
                          className="cursor-pointer hover:bg-muted/50 data-[state=open]:bg-muted/50"
                          onClick={toggleAnimation} 
                          aria-expanded={isAnimationVisible}
                          aria-controls={`animation-content-${actionKey}`}
                          data-state={isAnimationVisible ? 'open' : 'closed'}
                        >
                          <TableCell className="font-medium truncate" title={action.playerName || 'Unknown'}>
                            {action.playerName || 'Unknown'}
                          </TableCell>
                          <TableCell className="font-medium capitalize">{action.label}</TableCell>
                          <TableCell 
                            onClick={(e) => { 
                                e.stopPropagation(); // Prevent row click from toggling animation
                                handleActionClick(action); 
                            }}
                            className="hover:underline"
                            title="Click to zoom graph"
                          >
                            {interval}
                          </TableCell>
                          <TableCell className="text-right">
                            {metricDisplay}
                          </TableCell>
                          <TableCell className="w-[40px] text-center">
                            <ChevronDown className={`h-4 w-4 text-muted-foreground transition-transform duration-200 ${isAnimationVisible ? 'rotate-180' : ''}`} />
                          </TableCell>
                        </TableRow>
                        {isAnimationVisible && (
                          <TableRow id={`animation-content-${actionKey}`}>
                            <TableCell colSpan={5}>
                              <LacrosseStickAnimationLoader
                                action={action}
                                sessionId={sessionId}
                                allSensorData={sensorData}
                                sessionPlayers={sessionPlayers}
                                modelPath="/lacrosse_stick.glb" 
                              />
                            </TableCell>
                          </TableRow>
                        )}
                      </React.Fragment>
                    );
                  })}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        )}
        {isCalibrationSession && processedActions.length === 0 && !isLoading && (
             <Card>
                 <CardContent className="pt-6">
                    <p className="text-muted-foreground text-center">No actions detected in this calibration session.</p>
                 </CardContent>
             </Card>
         )}

      </div>
    </MainLayout>
  );
} 