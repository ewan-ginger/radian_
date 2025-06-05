'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Usb, Play, Pause, RefreshCw, Settings, Info } from "lucide-react";
import { useDevice } from '@/context/DeviceContext';
import { ConnectionStatus } from '@/types/serial';
import { 
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";

export function DeviceSection() {
  const {
    isSupported,
    status,
    error,
    deviceInfo,
    isStreaming,
    connect,
    disconnect,
    startStreaming,
    stopStreaming,
    setSampleRate,
    resetDevice,
  } = useDevice();

  const [sampleRate, setSampleRateValue] = useState<number>(50);

  // Update sample rate when device info changes
  useEffect(() => {
    if (deviceInfo && deviceInfo.sampleRate) {
      setSampleRateValue(deviceInfo.sampleRate);
    }
  }, [deviceInfo]);

  // Handle sample rate change
  const handleSampleRateChange = async (value: number) => {
    setSampleRateValue(value);
    await setSampleRate(value);
  };

  // Handle connect/disconnect button click
  const handleConnectionToggle = async () => {
    if (status === ConnectionStatus.CONNECTED) {
      await disconnect();
    } else {
      try {
        console.log('Attempting to connect to device...');
        console.log('Web Serial API supported:', isSupported);
        
        // Connect using our hook (which will trigger the port selection dialog)
        await connect();
      } catch (err) {
        console.error('Error connecting to device:', err);
      }
    }
  };

  // Handle start/stop streaming button click
  const handleStreamingToggle = async () => {
    if (isStreaming) {
      await stopStreaming();
    } else {
      await startStreaming();
    }
  };

  // Handle reset button click
  const handleReset = async () => {
    await resetDevice();
  };

  // Render connection status badge
  const renderStatusBadge = () => {
    switch (status) {
      case ConnectionStatus.CONNECTED:
        return <Badge className="bg-green-500">Connected</Badge>;
      case ConnectionStatus.CONNECTING:
        return <Badge className="bg-yellow-500">Connecting...</Badge>;
      case ConnectionStatus.ERROR:
        return <Badge className="bg-red-500">Error</Badge>;
      default:
        return <Badge className="bg-gray-500">Disconnected</Badge>;
    }
  };

  if (!isSupported) {
    return (
      <Card className="w-full">
        <CardHeader>
          <CardTitle>Device Connection</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-center p-6">
            <Usb className="h-16 w-16 text-muted-foreground" />
          </div>
          <p className="text-center text-sm text-muted-foreground">
            Web Serial API is not supported in your browser. Please use a browser that supports Web Serial API, such as Chrome or Edge.
          </p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="w-full">
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle>Device Connection</CardTitle>
        {renderStatusBadge()}
      </CardHeader>
      <CardContent className="space-y-6">
        {status !== ConnectionStatus.CONNECTED ? (
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
              onClick={handleConnectionToggle}
              disabled={status === ConnectionStatus.CONNECTING}
            >
              {status === ConnectionStatus.CONNECTING ? 'Connecting...' : 'Connect Device'}
            </Button>
          </>
        ) : (
          <>
            <div className="flex flex-col space-y-4">
              <div className="flex items-center justify-between">
                <div className="space-y-1">
                  <h3 className="text-sm font-medium">Device Info</h3>
                  <p className="text-xs text-muted-foreground">
                    {deviceInfo ? `ID: ${deviceInfo.deviceId}` : 'Loading device info...'}
                  </p>
                  {deviceInfo && (
                    <p className="text-xs text-muted-foreground">
                      Firmware: {deviceInfo.firmwareVersion}
                    </p>
                  )}
                </div>
                {deviceInfo && (
                  <div className="flex items-center gap-2">
                    <div className="text-xs text-muted-foreground">Battery:</div>
                    <div className="h-2 w-8 bg-muted rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-green-500" 
                        style={{ width: `${deviceInfo.batteryLevel}%` }}
                      />
                    </div>
                    <div className="text-xs">{deviceInfo.batteryLevel}%</div>
                  </div>
                )}
              </div>

              <div className="space-y-2">
                <Label htmlFor="sample-rate">Sample Rate: {sampleRate}Hz</Label>
                <Slider 
                  id="sample-rate"
                  min={10} 
                  max={100} 
                  step={10} 
                  value={[sampleRate]} 
                  onValueChange={(value) => handleSampleRateChange(value[0])}
                  disabled={isStreaming}
                />
              </div>

              <div className="flex flex-wrap gap-2 pt-4">
                <Button 
                  className="flex items-center gap-2"
                  onClick={handleStreamingToggle}
                >
                  {isStreaming ? (
                    <>
                      <Pause className="h-4 w-4" />
                      Stop Streaming
                    </>
                  ) : (
                    <>
                      <Play className="h-4 w-4" />
                      Start Streaming
                    </>
                  )}
                </Button>
                
                <Button 
                  variant="outline" 
                  className="flex items-center gap-2"
                  onClick={handleReset}
                  disabled={isStreaming}
                >
                  <RefreshCw className="h-4 w-4" />
                  Reset Device
                </Button>
                
                <Button 
                  variant="outline" 
                  className="flex items-center gap-2"
                  onClick={handleConnectionToggle}
                >
                  <Usb className="h-4 w-4" />
                  Disconnect
                </Button>
              </div>
            </div>
          </>
        )}

        {error && (
          <div className="text-sm text-destructive mt-4 p-4 bg-destructive/10 rounded-md">
            <h4 className="font-medium mb-1">Connection Error</h4>
            <p>{error.message}</p>
            <p className="text-xs mt-2">
              If you're having trouble connecting, try:
              <ul className="list-disc pl-5 mt-1">
                <li>Unplugging and reconnecting your device</li>
                <li>Restarting your browser</li>
                <li>Using a different USB port</li>
                <li>Checking if the device is recognized by your operating system</li>
              </ul>
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );
} 