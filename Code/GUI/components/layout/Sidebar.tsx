"use client";

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { usePathname, useRouter } from 'next/navigation';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { PanelLeft, PanelRight, Plus, Activity, Search, Trash2, Edit2, Users } from 'lucide-react';
import { useSessionData } from '@/hooks/useSessionData';
import { formatDistanceToNow } from 'date-fns';
import { ThemeToggle } from "@/components/ui/ThemeToggle";
import { SupabaseStatusPill } from "@/components/ui/SupabaseStatusPill";
import { Input } from "@/components/ui/input";
import { deleteSession, updateSession } from '@/lib/services/session-service';
import { toast } from "sonner";
import Image from 'next/image';

export function Sidebar() {
  const router = useRouter();
  const [isOpen, setIsOpen] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [editingSessionId, setEditingSessionId] = useState<string | null>(null);
  const [editingName, setEditingName] = useState('');
  const { sessions, isLoading, fetchSessions } = useSessionData();
  const pathname = usePathname();

  // Filter sessions based on search query
  const filteredSessions = sessions?.filter(session => {
    const searchLower = searchQuery.toLowerCase();
    const sessionName = (session.name || `Session ${session.id.substring(0, 8)}`).toLowerCase();
    return sessionName.includes(searchLower);
  }) || [];

  // Handle session deletion
  const handleDeleteSession = async (e: React.MouseEvent, sessionId: string) => {
    e.preventDefault();
    e.stopPropagation();

    if (confirm('Are you sure you want to delete this session? This action cannot be undone.')) {
      try {
        await deleteSession(sessionId);
        await fetchSessions();
        toast.success('Session deleted successfully');

        // If we're on the deleted session's page, redirect to /devices
        if (pathname.includes(sessionId)) {
          router.push('/devices');
        } else {
          // Reload the current page to refresh all data
          window.location.reload();
        }
      } catch (error) {
        console.error('Error deleting session:', error);
        toast.error('Failed to delete session');
      }
    }
  };

  // Handle session name edit
  const handleEditClick = (e: React.MouseEvent, sessionId: string, currentName: string) => {
    e.preventDefault();
    e.stopPropagation();
    setEditingSessionId(sessionId);
    setEditingName(currentName);
  };

  const handleNameSubmit = async (e: React.FormEvent, sessionId: string) => {
    e.preventDefault();
    if (!editingName.trim()) {
      toast.error('Session name cannot be empty');
      return;
    }

    try {
      await updateSession(sessionId, { name: editingName.trim() });
      await fetchSessions();
      setEditingSessionId(null);
      toast.success('Session name updated successfully');
      // Reload the page to refresh all data
      window.location.reload();
    } catch (error) {
      console.error('Error updating session name:', error);
      toast.error('Failed to update session name');
    }
  };

  // Log sessions to see duration values
  useEffect(() => {
    if (sessions.length > 0) {
      console.log('Sessions with durations:', sessions.map(s => ({
        id: s.id,
        name: s.name,
        duration: s.duration
      })));
    }
  }, [sessions]);

  // Format the duration from interval string to human readable format
  const formatDuration = (duration: string | null) => {
    if (!duration) {
      // If no duration is set, calculate it from start_time to now
      return 'N/A';
    }
    
    console.log('Formatting duration:', duration);
    
    // Check if duration is in HH:MM:SS format
    const timeFormatMatch = duration.match(/^(\d{2}):(\d{2}):(\d{2})$/);
    if (timeFormatMatch) {
      const hours = parseInt(timeFormatMatch[1]);
      const minutes = parseInt(timeFormatMatch[2]);
      const seconds = parseInt(timeFormatMatch[3]);
      
      console.log('Parsed time format duration:', { hours, minutes, seconds });
      
      if (hours > 0) {
        return `${hours}h ${minutes}m`;
      } else if (minutes > 0) {
        return `${minutes}m ${seconds}s`;
      } else {
        return `${seconds}s`;
      }
    }
    
    // Try to extract hours, minutes, seconds from PostgreSQL interval format
    try {
      // This is a simple parser for interval format like "1 hour 30 minutes 45 seconds"
      const hours = duration.match(/(\d+)\s+hour/i)?.[1] || '0';
      const minutes = duration.match(/(\d+)\s+minute/i)?.[1] || '0';
      const seconds = duration.match(/(\d+)\s+second/i)?.[1] || '0';
      
      console.log('Parsed text format duration:', { hours, minutes, seconds });
      
      if (parseInt(hours) > 0) {
        return `${hours}h ${minutes}m`;
      } else if (parseInt(minutes) > 0) {
        return `${minutes}m ${seconds}s`;
      } else {
        return `${seconds}s`;
      }
    } catch (e) {
      console.error('Error parsing duration:', e);
      
      // Alternative approach: if the duration is just a number of seconds
      if (typeof duration === 'string' && !isNaN(Number(duration))) {
        const totalSeconds = parseInt(duration);
        const hours = Math.floor(totalSeconds / 3600);
        const minutes = Math.floor((totalSeconds % 3600) / 60);
        const seconds = totalSeconds % 60;
        
        if (hours > 0) {
          return `${hours}h ${minutes}m`;
        } else if (minutes > 0) {
          return `${minutes}m ${seconds}s`;
        } else {
          return `${seconds}s`;
        }
      }
      
      return 'N/A';
    }
  };

  // Calculate duration for sessions without a duration field
  const calculateDuration = (session: any) => {
    if (session.duration) {
      return formatDuration(session.duration);
    }
    
    // If no duration but has end_time, calculate from start_time to end_time
    if (session.end_time) {
      const startTime = new Date(session.start_time);
      const endTime = new Date(session.end_time);
      const durationSeconds = Math.floor((endTime.getTime() - startTime.getTime()) / 1000);
      
      const hours = Math.floor(durationSeconds / 3600);
      const minutes = Math.floor((durationSeconds % 3600) / 60);
      const seconds = durationSeconds % 60;
      
      if (hours > 0) {
        return `${hours}h ${minutes}m`;
      } else if (minutes > 0) {
        return `${minutes}m ${seconds}s`;
      } else {
        return `${seconds}s`;
      }
    }
    
    // If no end_time, show as ongoing
    return 'Ongoing';
  };

  return (
    <div className={`relative h-screen border-r transition-all duration-300 ${isOpen ? 'w-64' : 'w-16'}`}>
      <div className="flex flex-col h-full">
        {/* Sidebar Header with Logo and Toggle */}
        <div className="p-4 border-b flex items-center justify-between">
          {isOpen ? (
            <>
              {/* Logo */}
              <Image 
                src="/radian_logo.png" 
                alt="Radian Logo" 
                width={30} 
                height={30} 
                className="mr-2"
              />
              <span className="font-roboto-mono text-[#474C59] font-bold text-xl">Radian_</span>
              <Button 
                variant="ghost" 
                size="icon" 
                onClick={() => setIsOpen(false)}
                className="ml-auto"
              >
                <PanelLeft size={18} />
              </Button>
            </>
          ) : (
            <Button 
              variant="ghost" 
              size="icon" 
              onClick={() => setIsOpen(true)}
              className="mx-auto"
            >
              <PanelRight size={18} />
            </Button>
          )}
        </div>
        
        {/* New Session Button */}
        <div className="p-4 border-b">
          {isOpen ? (
            <Link href="/devices">
              <Button className="w-full">
                <Plus className="mr-2 h-4 w-4" />
                New Session
              </Button>
            </Link>
          ) : (
            <Link href="/devices">
              <Button size="icon" className="w-full">
                <Plus className="h-4 w-4" />
              </Button>
            </Link>
          )}
        </div>
        
        {/* Player Profiles Button */}
        <div className="p-4 border-b">
          {isOpen ? (
            <Link href="/players">
              <div className="w-full flex items-center px-3 py-2 text-sm font-medium rounded-md hover:bg-accent transition-colors">
                <Users className="mr-2 h-4 w-4" />
                Player Profiles
              </div>
            </Link>
          ) : (
            <Link href="/players">
              <div className="w-full flex justify-center items-center p-2 rounded-md hover:bg-accent transition-colors">
                <Users className="h-4 w-4" />
              </div>
            </Link>
          )}
        </div>
        
        {/* Sessions List */}
        <ScrollArea className="flex-1">
          <div className="p-2 space-y-1">
            {isOpen && (
              <>
                <h3 className="px-2 py-1 text-sm font-medium text-muted-foreground">Recent Sessions</h3>
                <div className="px-2 py-1">
                  <div className="relative">
                    <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
                    <Input
                      placeholder="Search sessions..."
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      className="pl-8"
                    />
                  </div>
                </div>
              </>
            )}
            
            {isLoading ? (
              <div className="flex items-center justify-center py-8">
                <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-primary"></div>
              </div>
            ) : filteredSessions.length === 0 ? (
              isOpen && (
                <div className="px-2 py-4 text-sm text-muted-foreground text-center">
                  {searchQuery ? 'No matching sessions found' : 'No sessions found'}
                </div>
              )
            ) : (
              filteredSessions.map((session) => {
                const isEditing = editingSessionId === session.id;
                const sessionName = session.name || `Session ${session.id.substring(0, 8)}`;
                const isActive = pathname === `/sessions/${session.id}`;
                
                return (
                  <Link
                    key={session.id}
                    href={`/sessions/${session.id}`}
                    className={`relative group flex items-center justify-between p-2 hover:bg-accent rounded-md ${
                      isActive ? 'bg-accent' : ''
                    }`}
                  >
                    {isOpen ? (
                      <div className="flex-1 min-w-0">
                        {isEditing ? (
                          <form onSubmit={(e) => handleNameSubmit(e, session.id)} className="flex gap-2">
                            <Input
                              value={editingName}
                              onChange={(e) => setEditingName(e.target.value)}
                              onClick={(e) => e.stopPropagation()}
                              className="h-8"
                              autoFocus
                            />
                            <Button
                              type="submit"
                              size="sm"
                              variant="ghost"
                              onClick={(e) => e.stopPropagation()}
                            >
                              Save
                            </Button>
                          </form>
                        ) : (
                          <>
                            <div className="font-medium truncate">{sessionName}</div>
                            <div className="text-sm text-muted-foreground">
                              {formatDistanceToNow(new Date(session.start_time), { addSuffix: true })}
                            </div>
                            {session.session_type && (
                              <div className="text-xs text-muted-foreground flex items-center mt-1">
                                <Activity className="h-3 w-3 mr-1" />
                                {session.session_type.replace(/_/g, ' ')}
                              </div>
                            )}
                            
                            {session.players && session.players.length > 0 && (
                              <div className="text-xs text-muted-foreground flex items-center mt-1">
                                <Users className="h-3 w-3 mr-1" />
                                {session.players.length === 1 
                                  ? session.players[0].playerName 
                                  : `${session.players.length} players`
                                }
                              </div>
                            )}
                            <div className="flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity absolute right-2 top-1/2 -translate-y-1/2">
                              <Button
                                variant="ghost"
                                size="icon"
                                className="h-8 w-8"
                                onClick={(e) => handleEditClick(e, session.id, sessionName)}
                              >
                                <Edit2 className="h-4 w-4" />
                              </Button>
                              <Button
                                variant="ghost"
                                size="icon"
                                className="h-8 w-8 text-destructive"
                                onClick={(e) => handleDeleteSession(e, session.id)}
                              >
                                <Trash2 className="h-4 w-4" />
                              </Button>
                            </div>
                          </>
                        )}
                      </div>
                    ) : (
                      <div className="flex justify-center w-full">
                        <Activity className={`h-5 w-5 ${isActive ? 'text-primary' : ''}`} />
                      </div>
                    )}
                  </Link>
                );
              })
            )}
          </div>
        </ScrollArea>
        
        {/* Sidebar Footer with Theme Toggle and Supabase Status */}
        <div className="p-4 border-t">
          {isOpen ? (
            <div className="flex items-center justify-between">
              <SupabaseStatusPill />
              <ThemeToggle />
            </div>
          ) : (
            <div className="flex justify-center">
              <ThemeToggle />
            </div>
          )}
        </div>
      </div>
    </div>
  );
} 