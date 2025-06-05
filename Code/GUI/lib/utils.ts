import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"

/**
 * A utility function for merging class names with Tailwind CSS
 * @param inputs Class names or conditional class names
 * @returns Merged class names string
 */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

/**
 * Format seconds into a readable duration string (HH:MM:SS)
 * @param seconds Duration in seconds
 * @returns Formatted duration string
 */
export function formatDuration(seconds: number): string {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const remainingSeconds = seconds % 60;
  
  const pad = (num: number) => num.toString().padStart(2, '0');
  
  if (hours > 0) {
    return `${pad(hours)}:${pad(minutes)}:${pad(remainingSeconds)}`;
  }
  
  return `${pad(minutes)}:${pad(remainingSeconds)}`;
}

/**
 * Format a SessionType string into a human-readable format
 * @param type SessionType string (e.g., 'pass_calibration')
 * @returns Formatted string (e.g., 'Pass Calibration')
 */
export function formatSessionType(type: string | undefined | null): string {
  if (!type) return 'Unknown Type';
  
  // Replace underscores with spaces and capitalize each word
  return type
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}
