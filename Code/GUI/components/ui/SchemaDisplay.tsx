'use client';

import { useState } from 'react';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { completeSchemaSQL } from '@/lib/supabase/schema';
import { Copy, Check } from 'lucide-react';
import { toast } from 'sonner';

export function SchemaDisplay() {
  const [copied, setCopied] = useState(false);

  const copyToClipboard = async () => {
    try {
      await navigator.clipboard.writeText(completeSchemaSQL);
      setCopied(true);
      toast.success('SQL schema copied to clipboard');
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy text: ', err);
      toast.error('Failed to copy SQL schema');
    }
  };

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>Database Schema</CardTitle>
        <CardDescription>
          Execute this SQL in the Supabase SQL Editor to create the required tables
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="relative">
          <pre className="bg-muted p-4 rounded-md overflow-auto max-h-96 text-sm">
            <code>{completeSchemaSQL}</code>
          </pre>
          <Button
            variant="outline"
            size="icon"
            className="absolute top-2 right-2"
            onClick={copyToClipboard}
          >
            {copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
          </Button>
        </div>
      </CardContent>
      <CardFooter className="flex flex-col items-start gap-2">
        <p className="text-sm text-muted-foreground">
          This SQL script will create the following tables:
        </p>
        <ul className="list-disc list-inside text-sm text-muted-foreground">
          <li><code>players</code> - Stores information about players</li>
          <li><code>sessions</code> - Stores information about recording sessions</li>
          <li><code>sensor_data</code> - Stores sensor readings from ESP32 devices</li>
        </ul>
      </CardFooter>
    </Card>
  );
} 