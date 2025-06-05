"use client";

import { useState, useEffect } from 'react';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer 
} from 'recharts';
import { Card, CardContent } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

interface SensorData {
  timestamp: number;
  orientation_x: number;
  orientation_y: number;
  orientation_z: number;
  accelerometer_x: number;
  accelerometer_y: number;
  accelerometer_z: number;
  gyroscope_x: number;
  gyroscope_y: number;
  gyroscope_z: number;
  magnetometer_x: number;
  magnetometer_y: number;
  magnetometer_z: number;
}

interface LiveDataGraphProps {
  data: SensorData[];
  maxPoints?: number;
}

export function LiveDataGraph({ data, maxPoints = 100 }: LiveDataGraphProps) {
  const [activeTab, setActiveTab] = useState('orientation');
  const [visibleData, setVisibleData] = useState<any[]>([]);

  // Update visible data when data changes
  useEffect(() => {
    if (data.length <= maxPoints) {
      setVisibleData(data);
    } else {
      // Show only the most recent data points
      setVisibleData(data.slice(data.length - maxPoints));
    }
  }, [data, maxPoints]);

  // Format the timestamp for display
  const formatTimestamp = (timestamp: number) => {
    // Ensure timestamp is a number
    const numericTimestamp = Number(timestamp);
    if (isNaN(numericTimestamp)) {
      console.warn('Invalid timestamp value:', timestamp);
      return '0.0';
    }
    return numericTimestamp.toFixed(1);
  };

  // Format the timestamp for tooltip
  const formatTooltipTimestamp = (timestamp: number) => {
    return `Time: ${timestamp}s`;
  };

  // Use the data directly without transformation
  const getTransformedData = () => {
    return visibleData;
  };

  return (
    <Card className="w-full mt-6">
      <CardContent className="pt-6">
        <Tabs defaultValue="orientation" value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="grid grid-cols-4">
            <TabsTrigger value="orientation">Orientation</TabsTrigger>
            <TabsTrigger value="accelerometer">Accelerometer</TabsTrigger>
            <TabsTrigger value="gyroscope">Gyroscope</TabsTrigger>
            <TabsTrigger value="magnetometer">Magnetometer</TabsTrigger>
          </TabsList>
          
          <TabsContent value="orientation">
            <div className="h-[500px] mt-4">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart 
                  data={getTransformedData()}
                  margin={{ top: 20, right: 30, left: 20, bottom: 40 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="timestamp"
                    type="number"
                    domain={['dataMin', 'dataMax']}
                    tickFormatter={formatTimestamp}
                    label={{ value: 'Time (s)', position: 'bottom', offset: 15 }}
                  />
                  <YAxis 
                    label={{ 
                      value: 'Degrees', 
                      angle: -90, 
                      position: 'insideLeft',
                      offset: 10
                    }}
                    width={60}
                  />
                  <Tooltip 
                    formatter={(value: number) => [value.toFixed(2), '']}
                    labelFormatter={formatTooltipTimestamp}
                  />
                  <Legend 
                    verticalAlign="top"
                    height={50}
                    wrapperStyle={{
                      paddingTop: '20px',
                      paddingBottom: '20px'
                    }}
                  />
                  <Line type="monotone" dataKey="orientation_x" name="X" stroke="#8884d8" dot={false} />
                  <Line type="monotone" dataKey="orientation_y" name="Y" stroke="#82ca9d" dot={false} />
                  <Line type="monotone" dataKey="orientation_z" name="Z" stroke="#ffc658" dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </TabsContent>
          
          <TabsContent value="accelerometer">
            <div className="h-[500px] mt-4">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart 
                  data={getTransformedData()}
                  margin={{ top: 20, right: 30, left: 20, bottom: 40 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="timestamp"
                    type="number"
                    domain={['dataMin', 'dataMax']}
                    tickFormatter={formatTimestamp}
                    label={{ value: 'Time (s)', position: 'bottom', offset: 15 }}
                  />
                  <YAxis 
                    label={{ 
                      value: 'm/s²', 
                      angle: -90, 
                      position: 'insideLeft',
                      offset: 10
                    }}
                    width={60}
                  />
                  <Tooltip 
                    formatter={(value: number) => [value.toFixed(2), '']}
                    labelFormatter={formatTooltipTimestamp}
                  />
                  <Legend 
                    verticalAlign="top"
                    height={50}
                    wrapperStyle={{
                      paddingTop: '20px',
                      paddingBottom: '20px'
                    }}
                  />
                  <Line type="monotone" dataKey="accelerometer_x" name="X" stroke="#8884d8" dot={false} />
                  <Line type="monotone" dataKey="accelerometer_y" name="Y" stroke="#82ca9d" dot={false} />
                  <Line type="monotone" dataKey="accelerometer_z" name="Z" stroke="#ffc658" dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </TabsContent>
          
          <TabsContent value="gyroscope">
            <div className="h-[500px] mt-4">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart 
                  data={getTransformedData()}
                  margin={{ top: 20, right: 30, left: 20, bottom: 40 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="timestamp"
                    type="number"
                    domain={['dataMin', 'dataMax']}
                    tickFormatter={formatTimestamp}
                    label={{ value: 'Time (s)', position: 'bottom', offset: 15 }}
                  />
                  <YAxis 
                    label={{ 
                      value: '°/s', 
                      angle: -90, 
                      position: 'insideLeft',
                      offset: 10
                    }}
                    width={60}
                  />
                  <Tooltip 
                    formatter={(value: number) => [value.toFixed(2), '']}
                    labelFormatter={formatTooltipTimestamp}
                  />
                  <Legend 
                    verticalAlign="top"
                    height={50}
                    wrapperStyle={{
                      paddingTop: '20px',
                      paddingBottom: '20px'
                    }}
                  />
                  <Line type="monotone" dataKey="gyroscope_x" name="X" stroke="#8884d8" dot={false} />
                  <Line type="monotone" dataKey="gyroscope_y" name="Y" stroke="#82ca9d" dot={false} />
                  <Line type="monotone" dataKey="gyroscope_z" name="Z" stroke="#ffc658" dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </TabsContent>
          
          <TabsContent value="magnetometer">
            <div className="h-[500px] mt-4">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart 
                  data={getTransformedData()}
                  margin={{ top: 20, right: 30, left: 20, bottom: 40 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="timestamp"
                    type="number"
                    domain={['dataMin', 'dataMax']}
                    tickFormatter={formatTimestamp}
                    label={{ value: 'Time (s)', position: 'bottom', offset: 15 }}
                  />
                  <YAxis 
                    label={{ 
                      value: 'µT', 
                      angle: -90, 
                      position: 'insideLeft',
                      offset: 10
                    }}
                    width={60}
                  />
                  <Tooltip 
                    formatter={(value: number) => [value.toFixed(2), '']}
                    labelFormatter={formatTooltipTimestamp}
                  />
                  <Legend 
                    verticalAlign="top"
                    height={50}
                    wrapperStyle={{
                      paddingTop: '20px',
                      paddingBottom: '20px'
                    }}
                  />
                  <Line type="monotone" dataKey="magnetometer_x" name="X" stroke="#8884d8" dot={false} />
                  <Line type="monotone" dataKey="magnetometer_y" name="Y" stroke="#82ca9d" dot={false} />
                  <Line type="monotone" dataKey="magnetometer_z" name="Z" stroke="#ffc658" dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
} 