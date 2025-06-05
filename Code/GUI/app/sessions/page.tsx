"use client";

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { MainLayout } from '@/components/layout/MainLayout';
import { Button } from '@/components/ui/button';
import { Plus } from 'lucide-react';
import { useSessionData } from '@/hooks/useSessionData';
import Link from 'next/link';

export default function SessionsPage() {
  const router = useRouter();
  const { sessions, isLoading } = useSessionData();
  
  useEffect(() => {
    if (!isLoading && sessions.length > 0) {
      // Redirect to the most recent session
      router.push(`/sessions/${sessions[0].id}`);
    }
  }, [sessions, isLoading, router]);
  
  if (isLoading) {
    return (
      <MainLayout>
        <div className="flex items-center justify-center h-full">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary"></div>
        </div>
      </MainLayout>
    );
  }
  
  return (
    <MainLayout>
      <div className="flex flex-col items-center justify-center h-full max-w-md mx-auto text-center">
        <h1 className="text-3xl font-bold tracking-tight mb-4">No Sessions Found</h1>
        <p className="text-muted-foreground mb-8">
          You don't have any recording sessions yet. Start by connecting a device and recording some data.
        </p>
        <Link href="/devices">
          <Button size="lg">
            <Plus className="mr-2 h-5 w-5" />
            New Session
          </Button>
        </Link>
      </div>
    </MainLayout>
  );
} 