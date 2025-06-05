"use client";

import { Button } from "@/components/ui/button";
import { 
  Play, 
  Pause, 
  RefreshCw, 
  Save, 
  ZoomIn, 
  ZoomOut 
} from "lucide-react";
import { 
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Label } from "@/components/ui/label";

interface DataControlsProps {
  isRecording: boolean;
  onStartRecording: () => void;
  onStopRecording: () => void;
  onReset: () => void;
  onSave: () => void;
  dataRate: number;
  onDataRateChange: (value: number) => void;
  visiblePoints: number;
  onVisiblePointsChange: (value: number) => void;
  dataType: string;
  onDataTypeChange: (value: string) => void;
}

export function DataControls({
  isRecording,
  onStartRecording,
  onStopRecording,
  onReset,
  onSave,
  dataRate,
  onDataRateChange,
  visiblePoints,
  onVisiblePointsChange,
  dataType,
  onDataTypeChange
}: DataControlsProps) {
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap gap-2">
        {isRecording ? (
          <Button 
            variant="outline" 
            className="flex items-center gap-2"
            onClick={onStopRecording}
          >
            <Pause className="h-4 w-4" />
            Pause
          </Button>
        ) : (
          <Button 
            className="flex items-center gap-2"
            onClick={onStartRecording}
          >
            <Play className="h-4 w-4" />
            Start
          </Button>
        )}
        
        <Button 
          variant="outline" 
          className="flex items-center gap-2"
          onClick={onReset}
        >
          <RefreshCw className="h-4 w-4" />
          Reset
        </Button>
        
        <Button 
          variant="outline" 
          className="flex items-center gap-2"
          onClick={onSave}
        >
          <Save className="h-4 w-4" />
          Save
        </Button>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="space-y-2">
          <Label htmlFor="data-type">Data Type</Label>
          <Select value={dataType} onValueChange={onDataTypeChange}>
            <SelectTrigger id="data-type">
              <SelectValue placeholder="Select data type" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="orientation">Orientation</SelectItem>
              <SelectItem value="accelerometer">Accelerometer</SelectItem>
              <SelectItem value="gyroscope">Gyroscope</SelectItem>
              <SelectItem value="magnetometer">Magnetometer</SelectItem>
            </SelectContent>
          </Select>
        </div>
        
        <div className="space-y-2">
          <Label htmlFor="data-rate">Data Rate: {dataRate}Hz</Label>
          <Slider 
            id="data-rate"
            min={10} 
            max={100} 
            step={10} 
            value={[dataRate]} 
            onValueChange={(value) => onDataRateChange(value[0])}
          />
        </div>
      </div>
      
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <Label htmlFor="visible-points">Visible Points: {visiblePoints}</Label>
          <div className="flex items-center gap-1">
            <Button 
              variant="outline" 
              size="icon" 
              onClick={() => onVisiblePointsChange(Math.max(50, visiblePoints - 50))}
            >
              <ZoomIn className="h-4 w-4" />
            </Button>
            <Button 
              variant="outline" 
              size="icon" 
              onClick={() => onVisiblePointsChange(Math.min(500, visiblePoints + 50))}
            >
              <ZoomOut className="h-4 w-4" />
            </Button>
          </div>
        </div>
        <Slider 
          id="visible-points"
          min={50} 
          max={500} 
          step={50} 
          value={[visiblePoints]} 
          onValueChange={(value) => onVisiblePointsChange(value[0])}
        />
      </div>
    </div>
  );
} 