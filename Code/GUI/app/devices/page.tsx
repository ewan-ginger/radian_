"use client";

import { MainLayout } from "@/components/layout/MainLayout";
import { SimpleDeviceConnection } from "@/components/devices/SimpleDeviceConnection";
import { RecordingProvider } from "@/context/RecordingContext";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Laptop } from "lucide-react";

export default function DevicesPage() {
  return (
    <RecordingProvider>
      <MainLayout>
        <div>
          
          <SimpleDeviceConnection />
        </div>
      </MainLayout>
    </RecordingProvider>
  );
} 