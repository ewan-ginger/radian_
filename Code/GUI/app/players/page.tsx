'use client';

import { useState } from 'react';
import { MainLayout } from "@/components/layout/MainLayout";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { PlayerProfilesTable } from "@/components/players/PlayerProfilesTable";
import { PlayerProfileForm } from "@/components/players/PlayerProfileForm";
import { usePlayerData } from "@/hooks/usePlayerData";
import { PlayerEntity } from "@/types/database.types";
import { PlayerInsert } from "@/types/supabase";
import { Users, Plus } from "lucide-react";
import { toast } from "sonner";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";

export default function PlayersPage() {
  const { players, isLoading, error, createPlayer, updatePlayer, deletePlayer, fetchPlayers } = usePlayerData();
  const [isAddDialogOpen, setIsAddDialogOpen] = useState(false);
  const [isEditDialogOpen, setIsEditDialogOpen] = useState(false);
  const [isDeleteDialogOpen, setIsDeleteDialogOpen] = useState(false);
  const [selectedPlayer, setSelectedPlayer] = useState<PlayerEntity | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Handle form submission for adding a new player
  const handleAddPlayer = async (data: PlayerInsert) => {
    try {
      setIsSubmitting(true);
      await createPlayer(data);
      toast.success('Player added successfully!');
      setIsAddDialogOpen(false);
      await fetchPlayers();
    } catch (error) {
      toast.error('Failed to add player');
      console.error('Error adding player:', error);
    } finally {
      setIsSubmitting(false);
    }
  };

  // Handle form submission for updating a player
  const handleUpdatePlayer = async (data: PlayerInsert) => {
    if (!selectedPlayer?.id) return;
    
    try {
      setIsSubmitting(true);
      await updatePlayer(selectedPlayer.id, data);
      toast.success('Player updated successfully!');
      setIsEditDialogOpen(false);
      setSelectedPlayer(null);
      await fetchPlayers();
    } catch (error) {
      toast.error('Failed to update player');
      console.error('Error updating player:', error);
    } finally {
      setIsSubmitting(false);
    }
  };

  // Handle player deletion
  const handleDeletePlayer = async () => {
    if (!selectedPlayer?.id) return;
    
    try {
      setIsSubmitting(true);
      await deletePlayer(selectedPlayer.id);
      toast.success('Player deleted successfully!');
      setIsDeleteDialogOpen(false);
      setSelectedPlayer(null);
      await fetchPlayers();
    } catch (error) {
      toast.error('Failed to delete player');
      console.error('Error deleting player:', error);
    } finally {
      setIsSubmitting(false);
    }
  };

  // Open edit dialog with selected player
  const handleEditClick = (player: PlayerEntity) => {
    setSelectedPlayer(player);
    setIsEditDialogOpen(true);
  };

  // Open delete confirmation dialog
  const handleDeleteClick = (player: PlayerEntity) => {
    setSelectedPlayer(player);
    setIsDeleteDialogOpen(true);
  };

  return (
    <MainLayout>
      <div className="space-y-6">
        <div className="flex justify-between items-center">
          <div className="flex items-center gap-2">
            <Users className="h-6 w-6 text-green-500" />
            <h1 className="text-3xl font-bold tracking-tight">Player Profiles</h1>
          </div>
          <Button onClick={() => setIsAddDialogOpen(true)}>
            <Plus className="mr-2 h-4 w-4" />
            Add Player
          </Button>
        </div>
        
        <p className="text-lg text-muted-foreground">
          Manage player profiles for the lacrosse team.
        </p>
        
        <Card>
          <CardHeader>
            <CardTitle>Player Profiles</CardTitle>
            <CardDescription>
              View, edit and manage player profiles
            </CardDescription>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <div className="flex justify-center py-8">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
              </div>
            ) : error ? (
              <div className="text-center py-8 text-destructive">
                {error}
              </div>
            ) : (
              <PlayerProfilesTable
                players={players}
                onEdit={handleEditClick}
                onDelete={handleDeleteClick}
              />
            )}
          </CardContent>
        </Card>
      </div>

      {/* Add Player Dialog */}
      <Dialog open={isAddDialogOpen} onOpenChange={setIsAddDialogOpen}>
        <DialogContent className="sm:max-w-[500px]">
          <DialogHeader>
            <DialogTitle>Add New Player</DialogTitle>
            <DialogDescription>
              Create a new player profile with the form below.
            </DialogDescription>
          </DialogHeader>
          <PlayerProfileForm
            onSubmit={handleAddPlayer}
            onCancel={() => setIsAddDialogOpen(false)}
            isSubmitting={isSubmitting}
          />
        </DialogContent>
      </Dialog>

      {/* Edit Player Dialog */}
      <Dialog open={isEditDialogOpen} onOpenChange={setIsEditDialogOpen}>
        <DialogContent className="sm:max-w-[500px]">
          <DialogHeader>
            <DialogTitle>Edit Player</DialogTitle>
            <DialogDescription>
              Update the player profile information.
            </DialogDescription>
          </DialogHeader>
          {selectedPlayer && (
            <PlayerProfileForm
              player={selectedPlayer}
              onSubmit={handleUpdatePlayer}
              onCancel={() => setIsEditDialogOpen(false)}
              isSubmitting={isSubmitting}
            />
          )}
        </DialogContent>
      </Dialog>

      {/* Delete Confirmation Dialog */}
      <AlertDialog open={isDeleteDialogOpen} onOpenChange={setIsDeleteDialogOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Are you sure?</AlertDialogTitle>
            <AlertDialogDescription>
              This action will permanently delete the player profile for{' '}
              <span className="font-bold">{selectedPlayer?.name}</span>.
              This action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={isSubmitting}>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={handleDeletePlayer}
              disabled={isSubmitting}
              className="bg-destructive hover:bg-destructive/90"
            >
              {isSubmitting ? 'Deleting...' : 'Delete Player'}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </MainLayout>
  );
} 