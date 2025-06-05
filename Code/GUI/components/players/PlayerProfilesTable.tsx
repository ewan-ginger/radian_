'use client';

import { useState } from 'react';
import { 
  Table, 
  TableBody, 
  TableCaption, 
  TableCell, 
  TableHead, 
  TableHeader, 
  TableRow 
} from "@/components/ui/table";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Edit2, Trash2 } from "lucide-react";
import { PlayerEntity } from '@/types/database.types';
import { 
  DropdownMenu, 
  DropdownMenuContent, 
  DropdownMenuItem, 
  DropdownMenuLabel, 
  DropdownMenuSeparator, 
  DropdownMenuTrigger 
} from "@/components/ui/dropdown-menu";
import { ChevronDown, ChevronUp, ChevronsUpDown } from "lucide-react";

export type SortOrder = 'asc' | 'desc' | 'none';
export type SortField = 'name' | 'stick_type' | 'position' | 'strong_hand';

interface PlayerProfilesTableProps {
  players: PlayerEntity[];
  onEdit: (player: PlayerEntity) => void;
  onDelete: (player: PlayerEntity) => void;
}

export function PlayerProfilesTable({ players, onEdit, onDelete }: PlayerProfilesTableProps) {
  const [searchTerm, setSearchTerm] = useState('');
  const [sortField, setSortField] = useState<SortField>('name');
  const [sortOrder, setSortOrder] = useState<SortOrder>('asc');

  // Handle search filter
  const filteredPlayers = players.filter(player => {
    if (!searchTerm) return true;
    
    const search = searchTerm.toLowerCase();
    return (
      player.name.toLowerCase().includes(search) ||
      player.stick_type.toLowerCase().includes(search) ||
      player.position.toLowerCase().includes(search) ||
      player.strong_hand.toLowerCase().includes(search)
    );
  });

  // Handle sorting
  const sortedPlayers = [...filteredPlayers].sort((a, b) => {
    if (sortOrder === 'none') return 0;
    
    const aValue = a[sortField];
    const bValue = b[sortField];
    
    const comparison = aValue.localeCompare(bValue);
    return sortOrder === 'asc' ? comparison : -comparison;
  });

  const handleSort = (field: SortField) => {
    if (field === sortField) {
      // Cycle through sort orders: asc -> desc -> none -> asc
      const orders: SortOrder[] = ['asc', 'desc', 'none'];
      const currentIndex = orders.indexOf(sortOrder);
      const nextOrder = orders[(currentIndex + 1) % orders.length];
      setSortOrder(nextOrder);
    } else {
      setSortField(field);
      setSortOrder('asc');
    }
  };

  // Function to render sort indicator
  const renderSortIndicator = (field: SortField) => {
    if (field !== sortField) return <ChevronsUpDown className="ml-2 h-4 w-4" />;
    if (sortOrder === 'asc') return <ChevronUp className="ml-2 h-4 w-4" />;
    if (sortOrder === 'desc') return <ChevronDown className="ml-2 h-4 w-4" />;
    return <ChevronsUpDown className="ml-2 h-4 w-4" />;
  };

  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <Input
          placeholder="Search players..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="max-w-sm"
        />
        <div className="text-sm text-muted-foreground">
          {filteredPlayers.length} player{filteredPlayers.length !== 1 ? 's' : ''}
        </div>
      </div>
      
      <div className="rounded-md border">
        <Table>
          <TableCaption>List of player profiles</TableCaption>
          <TableHeader>
            <TableRow>
              <TableHead className="cursor-pointer" onClick={() => handleSort('name')}>
                <div className="flex items-center">
                  Name {renderSortIndicator('name')}
                </div>
              </TableHead>
              <TableHead className="cursor-pointer" onClick={() => handleSort('stick_type')}>
                <div className="flex items-center">
                  Stick Type {renderSortIndicator('stick_type')}
                </div>
              </TableHead>
              <TableHead className="cursor-pointer" onClick={() => handleSort('position')}>
                <div className="flex items-center">
                  Position {renderSortIndicator('position')}
                </div>
              </TableHead>
              <TableHead className="cursor-pointer" onClick={() => handleSort('strong_hand')}>
                <div className="flex items-center">
                  Strong Hand {renderSortIndicator('strong_hand')}
                </div>
              </TableHead>
              <TableHead className="text-right">Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {sortedPlayers.length === 0 ? (
              <TableRow>
                <TableCell colSpan={5} className="h-24 text-center">
                  {searchTerm ? 'No players found matching your search' : 'No players found'}
                </TableCell>
              </TableRow>
            ) : (
              sortedPlayers.map((player) => (
                <TableRow key={player.id}>
                  <TableCell className="font-medium">{player.name}</TableCell>
                  <TableCell>
                    {player.stick_type ? player.stick_type.split('-').map(word => 
                      word.charAt(0).toUpperCase() + word.slice(1)
                    ).join(' ') : ''}
                  </TableCell>
                  <TableCell className="capitalize">
                    {player.position}
                  </TableCell>
                  <TableCell className="capitalize">
                    {player.strong_hand}
                  </TableCell>
                  <TableCell className="text-right">
                    <div className="flex justify-end space-x-2">
                      <Button
                        variant="ghost"
                        size="icon"
                        onClick={() => onEdit(player)}
                      >
                        <Edit2 className="h-4 w-4" />
                      </Button>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="text-destructive"
                        onClick={() => onDelete(player)}
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>
                  </TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </div>
    </div>
  );
} 