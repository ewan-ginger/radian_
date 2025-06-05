import { MainLayout } from "@/components/layout/MainLayout";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { BarChart3 } from "lucide-react";

export const metadata = {
  title: 'Data Visualization - Radian',
  description: 'Visualize player movement data in real-time',
};

export default function VisualizationPage() {
  return (
    <MainLayout>
      <div className="space-y-6">
        <div className="flex items-center gap-2">
          <BarChart3 className="h-6 w-6 text-red-500" />
          <h1 className="text-3xl font-bold tracking-tight">Data Visualization</h1>
        </div>
        
        <p className="text-lg text-muted-foreground">
          Visualize player movement data in real-time with interactive graphs.
        </p>
        
        <Card>
          <CardHeader>
            <CardTitle>Orientation Data</CardTitle>
            <CardDescription>
              Visualize player orientation in real-time
            </CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-muted-foreground">
              Data visualization components will be implemented in Step 10.
            </p>
          </CardContent>
        </Card>
      </div>
    </MainLayout>
  );
} 