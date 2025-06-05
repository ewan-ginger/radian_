export type Json =
  | string
  | number
  | boolean
  | null
  | { [key: string]: Json | undefined }
  | Json[];

export interface Database {
  public: {
    Tables: {
      players: {
        Row: {
          id: string;
          name: string;
          device_id: string | null;
          created_at: string;
          stick_type: 'short-stick' | 'long-stick' | 'goalie-stick';
          position: 'attack' | 'midfield' | 'defense' | 'goalie' | 'faceoff';
          strong_hand: 'left' | 'right';
          updated_at: string;
        };
        Insert: {
          id?: string;
          name: string;
          device_id?: string | null;
          created_at?: string;
          stick_type: 'short-stick' | 'long-stick' | 'goalie-stick';
          position: 'attack' | 'midfield' | 'defense' | 'goalie' | 'faceoff';
          strong_hand: 'left' | 'right';
          updated_at?: string;
        };
        Update: {
          id?: string;
          name?: string;
          device_id?: string | null;
          created_at?: string;
          stick_type?: 'short-stick' | 'long-stick' | 'goalie-stick';
          position?: 'attack' | 'midfield' | 'defense' | 'goalie' | 'faceoff';
          strong_hand?: 'left' | 'right';
          updated_at?: string;
        };
      };
      sessions: {
        Row: {
          id: string;
          name: string | null;
          start_time: string;
          end_time: string | null;
          duration: string | null;
          session_type: string | null;
          created_at: string;
        };
        Insert: {
          id?: string;
          name?: string | null;
          start_time?: string;
          end_time?: string | null;
          duration?: string | null;
          session_type?: string | null;
          created_at?: string;
        };
        Update: {
          id?: string;
          name?: string | null;
          start_time?: string;
          end_time?: string | null;
          duration?: string | null;
          session_type?: string | null;
          created_at?: string;
        };
      };
      session_players: {
        Row: {
          id: string;
          session_id: string;
          player_id: string | null;
          device_id: string | null;
          created_at: string;
        };
        Insert: {
          id?: string;
          session_id: string;
          player_id?: string | null;
          device_id?: string | null;
          created_at?: string;
        };
        Update: {
          id?: string;
          session_id?: string;
          player_id?: string | null;
          device_id?: string | null;
          created_at?: string;
        };
      };
      sensor_data: {
        Row: {
          id: string;
          session_id: string;
          device_id: string | null;
          timestamp: number;
          accelerometer_x: number | null;
          accelerometer_y: number | null;
          accelerometer_z: number | null;
          gyroscope_x: number | null;
          gyroscope_y: number | null;
          gyroscope_z: number | null;
          magnetometer_x: number | null;
          magnetometer_y: number | null;
          magnetometer_z: number | null;
          orientation_x: number | null;
          orientation_y: number | null;
          orientation_z: number | null;
          battery_level: number | null;
          created_at: string;
        };
        Insert: {
          id?: string;
          session_id: string;
          device_id?: string | null;
          timestamp: number;
          accelerometer_x?: number | null;
          accelerometer_y?: number | null;
          accelerometer_z?: number | null;
          gyroscope_x?: number | null;
          gyroscope_y?: number | null;
          gyroscope_z?: number | null;
          magnetometer_x?: number | null;
          magnetometer_y?: number | null;
          magnetometer_z?: number | null;
          orientation_x?: number | null;
          orientation_y?: number | null;
          orientation_z?: number | null;
          battery_level?: number | null;
          created_at?: string;
        };
        Update: {
          id?: string;
          session_id?: string;
          device_id?: string | null;
          timestamp?: number;
          accelerometer_x?: number | null;
          accelerometer_y?: number | null;
          accelerometer_z?: number | null;
          gyroscope_x?: number | null;
          gyroscope_y?: number | null;
          gyroscope_z?: number | null;
          magnetometer_x?: number | null;
          magnetometer_y?: number | null;
          magnetometer_z?: number | null;
          orientation_x?: number | null;
          orientation_y?: number | null;
          orientation_z?: number | null;
          battery_level?: number | null;
          created_at?: string;
        };
      };
    };
    Views: {
      [_ in never]: never;
    };
    Functions: {
      [_ in never]: never;
    };
    Enums: {
      [_ in never]: never;
    };
  };
}

// Helper types for common database entities
export type Player = Database['public']['Tables']['players']['Row'];
export type Session = Database['public']['Tables']['sessions']['Row'];
export type SessionPlayer = Database['public']['Tables']['session_players']['Row'];
export type SensorData = Database['public']['Tables']['sensor_data']['Row'];

// Helper types for inserting data
export type PlayerInsert = Database['public']['Tables']['players']['Insert'];
export type SessionInsert = Database['public']['Tables']['sessions']['Insert'];
export type SessionPlayerInsert = Database['public']['Tables']['session_players']['Insert'];
export type SensorDataInsert = Database['public']['Tables']['sensor_data']['Insert'];

// Helper types for updating data
export type PlayerUpdate = Database['public']['Tables']['players']['Update'];
export type SessionUpdate = Database['public']['Tables']['sessions']['Update'];
export type SessionPlayerUpdate = Database['public']['Tables']['session_players']['Update'];
export type SensorDataUpdate = Database['public']['Tables']['sensor_data']['Update']; 