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
  ResponsiveContainer, 
  Brush
} from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface OrientationData {
  timestamp: number;
  x: number;
  y: number;
  z: number;
}

interface OrientationGraphProps {
  data: OrientationData[];
  title: string;
}

export function OrientationGraph({ 
  data, 
  title, 
}: OrientationGraphProps) {
  // const [visibleData, setVisibleData] = useState<OrientationData[]>([]);

  // Format the timestamp for display
  const formatTimestamp = (timestamp: number) => {
    // Ensure timestamp is a number
    const numericTimestamp = Number(timestamp);
    if (isNaN(numericTimestamp)) {
      console.warn('Invalid timestamp value:', timestamp);
      return '0.00';
    }
    return numericTimestamp.toFixed(2);
  };

  // Format the timestamp for tooltip
  const formatTooltipTimestamp = (timestamp: number) => {
    // Ensure timestamp is a number
    const numericTimestamp = Number(timestamp);
    if (isNaN(numericTimestamp)) {
      console.warn('Invalid tooltip timestamp value:', timestamp);
      return 'Time: 0.00s';
    }
    return `Time: ${numericTimestamp.toFixed(2)}s`;
  };

  return (
    <Card className="w-full overflow-hidden">
      <CardContent className="pt-6">
        <div className="w-full h-[400px] overflow-x-auto">
          <div style={{ width: Math.max(100, data.length * 0.5) + '%', height: '100%', minWidth: '600px' }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={data}
                margin={{
                  top: 30,
                  right: 30,
                  left: 30,
                  bottom: 40,
                }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="timestamp" 
                  label={{ value: 'Time (s)', position: 'insideBottomRight', offset: -10 }}
                  tickFormatter={formatTimestamp}
                  domain={['dataMin', 'dataMax']}
                  type="number"
                  allowDecimals={true}
                  allowDataOverflow={false}
                />
                <YAxis 
                  label={{ value: 'Orientation (degrees)', angle: -90, position: 'insideLeft', offset: -15, dy: 50 }}
                  width={60}
                />
                <Tooltip 
                  formatter={(value, name) => [`${value.toFixed(2)}°`, name]}
                  labelFormatter={formatTooltipTimestamp}
                />
                <Legend />
                <Line 
                  type="monotone" 
                  dataKey="x" 
                  stroke="#8884d8" 
                  name="X-Axis" 
                  dot={false}
                  activeDot={{ r: 8 }}
                />
                <Line 
                  type="monotone" 
                  dataKey="y" 
                  stroke="#82ca9d" 
                  name="Y-Axis" 
                  dot={false}
                  activeDot={{ r: 8 }}
                />
                <Line 
                  type="monotone" 
                  dataKey="z" 
                  stroke="#ff7300" 
                  name="Z-Axis" 
                  dot={false}
                  activeDot={{ r: 8 }}
                />
                <Brush 
                  dataKey="timestamp" 
                  height={30} 
                  stroke="#8884d8" 
                  tickFormatter={formatTimestamp} 
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </CardContent>
    </Card>
  );
} 