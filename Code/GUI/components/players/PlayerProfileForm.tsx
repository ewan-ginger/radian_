'use client';

import { useState, useEffect } from 'react';
import { zodResolver } from '@hookform/resolvers/zod';
import { useForm } from 'react-hook-form';
import * as z from 'zod';
import { 
  Form, 
  FormControl, 
  FormDescription, 
  FormField, 
  FormItem, 
  FormLabel, 
  FormMessage 
} from '@/components/ui/form';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group';
import { 
  Select, 
  SelectContent, 
  SelectItem, 
  SelectTrigger, 
  SelectValue 
} from '@/components/ui/select';
import { PlayerEntity } from '@/types/database.types';
import { Label } from '@/components/ui/label';
import { PlayerInsert } from '@/types/supabase';

// Define form validation schema
const playerFormSchema = z.object({
  name: z
    .string()
    .min(2, { message: 'Name must be at least 2 characters' })
    .max(50, { message: 'Name must be 50 characters or less' }),
  stick_type: z.enum(['short-stick', 'long-stick', 'goalie-stick'], {
    required_error: 'Please select a stick type'
  }),
  position: z.enum(['attack', 'midfield', 'faceoff', 'defense', 'goalie'], {
    required_error: 'Please select a position'
  }),
  strong_hand: z.enum(['left', 'right'], {
    required_error: 'Please select a strong hand preference'
  })
});

type PlayerFormValues = z.infer<typeof playerFormSchema>;

interface PlayerProfileFormProps {
  player?: PlayerEntity; // Optional for editing an existing player
  onSubmit: (data: PlayerInsert) => Promise<void>;
  onCancel: () => void;
  isSubmitting: boolean;
}

export function PlayerProfileForm({ 
  player, 
  onSubmit, 
  onCancel,
  isSubmitting 
}: PlayerProfileFormProps) {
  // Set up form with validation
  const form = useForm<PlayerFormValues>({
    resolver: zodResolver(playerFormSchema),
    defaultValues: {
      name: player?.name || '',
      stick_type: player?.stick_type || 'short-stick',
      position: player?.position || 'midfield',
      strong_hand: player?.strong_hand || 'right'
    }
  });

  const isEditMode = !!player;

  // Function to handle form submission
  const handleSubmit = async (values: PlayerFormValues) => {
    try {
      await onSubmit({
        ...values,
        id: player?.id, // Include id if in edit mode
        device_id: player?.device_id // Preserve device_id if it exists
      });
      form.reset(); // Reset form after submission
    } catch (error) {
      console.error('Error submitting form:', error);
    }
  };

  return (
    <Form {...form}>
      <form onSubmit={form.handleSubmit(handleSubmit)} className="space-y-6">
        <FormField
          control={form.control}
          name="name"
          render={({ field }) => (
            <FormItem>
              <FormLabel>Name</FormLabel>
              <FormControl>
                <Input placeholder="Player name" {...field} />
              </FormControl>
              <FormDescription>
                Enter the player's full name
              </FormDescription>
              <FormMessage />
            </FormItem>
          )}
        />

        <FormField
          control={form.control}
          name="stick_type"
          render={({ field }) => (
            <FormItem>
              <FormLabel>Stick Type</FormLabel>
              <Select
                onValueChange={field.onChange}
                defaultValue={field.value}
              >
                <FormControl>
                  <SelectTrigger>
                    <SelectValue placeholder="Select stick type" />
                  </SelectTrigger>
                </FormControl>
                <SelectContent>
                  <SelectItem value="short-stick">Short Stick</SelectItem>
                  <SelectItem value="long-stick">Long Stick</SelectItem>
                  <SelectItem value="goalie-stick">Goalie Stick</SelectItem>
                </SelectContent>
              </Select>
              <FormDescription>
                Select the type of stick the player uses
              </FormDescription>
              <FormMessage />
            </FormItem>
          )}
        />

        <FormField
          control={form.control}
          name="position"
          render={({ field }) => (
            <FormItem>
              <FormLabel>Position</FormLabel>
              <Select
                onValueChange={field.onChange}
                defaultValue={field.value}
              >
                <FormControl>
                  <SelectTrigger>
                    <SelectValue placeholder="Select position" />
                  </SelectTrigger>
                </FormControl>
                <SelectContent>
                  <SelectItem value="attack">Attack</SelectItem>
                  <SelectItem value="midfield">Midfield</SelectItem>
                  <SelectItem value="faceoff">Faceoff</SelectItem>
                  <SelectItem value="defense">Defense</SelectItem>
                  <SelectItem value="goalie">Goalie</SelectItem>
                </SelectContent>
              </Select>
              <FormDescription>
                Select the player's primary position
              </FormDescription>
              <FormMessage />
            </FormItem>
          )}
        />

        <FormField
          control={form.control}
          name="strong_hand"
          render={({ field }) => (
            <FormItem className="space-y-3">
              <FormLabel>Strong Hand</FormLabel>
              <FormControl>
                <RadioGroup
                  onValueChange={field.onChange}
                  defaultValue={field.value}
                  className="flex space-x-6"
                >
                  <FormItem className="flex items-center space-x-2">
                    <FormControl>
                      <RadioGroupItem value="right" />
                    </FormControl>
                    <FormLabel className="font-normal cursor-pointer">
                      Right
                    </FormLabel>
                  </FormItem>
                  <FormItem className="flex items-center space-x-2">
                    <FormControl>
                      <RadioGroupItem value="left" />
                    </FormControl>
                    <FormLabel className="font-normal cursor-pointer">
                      Left
                    </FormLabel>
                  </FormItem>
                </RadioGroup>
              </FormControl>
              <FormDescription>
                Select the player's dominant hand
              </FormDescription>
              <FormMessage />
            </FormItem>
          )}
        />

        <div className="flex justify-end space-x-2">
          <Button 
            type="button" 
            variant="outline" 
            onClick={onCancel}
            disabled={isSubmitting}
          >
            Cancel
          </Button>
          <Button 
            type="submit" 
            disabled={isSubmitting}
          >
            {isSubmitting ? 'Saving...' : isEditMode ? 'Update Player' : 'Add Player'}
          </Button>
        </div>
      </form>
    </Form>
  );
} 