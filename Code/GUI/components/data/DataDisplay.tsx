"use client";

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { OrientationGraph } from './OrientationGraph';
import { AccelerationGraph } from './AccelerationGraph';
import { GyroscopeGraph } from './GyroscopeGraph';
import { MagnetometerGraph } from './MagnetometerGraph';
import { DataControls } from './DataControls';

// Sample data generator for demonstration
const generateSampleData = (count: number, noiseLevel = 5) => {
  const data = [];
  const startTime = 0;
  
  for (let i = 0; i < count; i++) {
    const timestamp = startTime + i * 0.02; // Use 0.02s intervals for 50Hz
    data.push({
      timestamp,
      x: Math.sin(i * 0.1) * 45 + (Math.random() - 0.5) * noiseLevel,
      y: Math.cos(i * 0.1) * 30 + (Math.random() - 0.5) * noiseLevel,
      z: Math.sin(i * 0.05) * 15 + (Math.random() - 0.5) * noiseLevel,
    });
  }
  
  return data;
};

export function DataDisplay() {
  const [isRecording, setIsRecording] = useState(false);
  const [dataType, setDataType] = useState('orientation');
  const [dataRate, setDataRate] = useState(50);
  const [visiblePoints, setVisiblePoints] = useState(100);
  const [orientationData, setOrientationData] = useState(generateSampleData(50));
  const [accelerometerData, setAccelerometerData] = useState(generateSampleData(50));
  const [gyroscopeData, setGyroscopeData] = useState(generateSampleData(50));
  const [magnetometerData, setMagnetometerData] = useState(generateSampleData(50));
  
  // Simulate data streaming when recording
  useEffect(() => {
    if (!isRecording) return;
    
    // Convert timestamps to seconds for better x-axis display
    const startTime = Date.now() / 1000;
    
    const interval = setInterval(() => {
      const currentTime = Date.now() / 1000;
      const elapsedTime = currentTime - startTime;
      
      const newPoint = {
        timestamp: elapsedTime,
        x: Math.sin(Date.now() * 0.001) * 45 + (Math.random() - 0.5) * 5,
        y: Math.cos(Date.now() * 0.001) * 30 + (Math.random() - 0.5) * 5,
        z: Math.sin(Date.now() * 0.0005) * 15 + (Math.random() - 0.5) * 5,
      };
      
      setOrientationData(prev => [...prev, newPoint]);
      setAccelerometerData(prev => [...prev, {
        ...newPoint,
        x: newPoint.x * 0.2,
        y: newPoint.y * 0.2,
        z: newPoint.z * 0.2,
      }]);
      setGyroscopeData(prev => [...prev, {
        ...newPoint,
        x: newPoint.x * 0.1,
        y: newPoint.y * 0.1,
        z: newPoint.z * 0.1,
      }]);
      setMagnetometerData(prev => [...prev, {
        ...newPoint,
        x: newPoint.x * 0.05,
        y: newPoint.y * 0.05,
        z: newPoint.z * 0.05,
      }]);
    }, 1000 / dataRate);
    
    return () => clearInterval(interval);
  }, [isRecording, dataRate]);
  
  const handleStartRecording = () => setIsRecording(true);
  const handleStopRecording = () => setIsRecording(false);
  
  const handleReset = () => {
    setOrientationData(generateSampleData(50));
    setAccelerometerData(generateSampleData(50));
    setGyroscopeData(generateSampleData(50));
    setMagnetometerData(generateSampleData(50));
  };
  
  const handleSave = () => {
    // In a real app, this would save data to Supabase or export to a file
    console.log('Saving data...');
    alert('Data saved successfully!');
  };
  
  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>Data Visualization</CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        <DataControls 
          isRecording={isRecording}
          onStartRecording={handleStartRecording}
          onStopRecording={handleStopRecording}
          onReset={handleReset}
          onSave={handleSave}
          dataRate={dataRate}
          onDataRateChange={setDataRate}
          visiblePoints={visiblePoints}
          onVisiblePointsChange={setVisiblePoints}
          dataType={dataType}
          onDataTypeChange={setDataType}
        />
        
        <Tabs defaultValue="orientation" value={dataType} onValueChange={setDataType}>
          <TabsList className="grid grid-cols-4">
            <TabsTrigger value="orientation">Orientation</TabsTrigger>
            <TabsTrigger value="accelerometer">Accelerometer</TabsTrigger>
            <TabsTrigger value="gyroscope">Gyroscope</TabsTrigger>
            <TabsTrigger value="magnetometer">Magnetometer</TabsTrigger>
          </TabsList>
          
          <TabsContent value="orientation">
            <OrientationGraph 
              data={orientationData} 
              title="Orientation (degrees)" 
              maxPoints={visiblePoints}
            />
          </TabsContent>
          
          <TabsContent value="accelerometer">
            <AccelerationGraph 
              data={accelerometerData} 
              title="Accelerometer (m/s²)" 
              maxPoints={visiblePoints}
            />
          </TabsContent>
          
          <TabsContent value="gyroscope">
            <GyroscopeGraph 
              data={gyroscopeData} 
              title="Gyroscope (rad/s)" 
              maxPoints={visiblePoints}
            />
          </TabsContent>
          
          <TabsContent value="magnetometer">
            <MagnetometerGraph 
              data={magnetometerData} 
              title="Magnetometer (μT)" 
              maxPoints={visiblePoints}
            />
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
} 