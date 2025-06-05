"use client";

import { ThemeToggle } from "@/components/ui/ThemeToggle";
import { SupabaseStatusPill } from "@/components/ui/SupabaseStatusPill";

export function Header() {
  return (
    <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container flex h-14 items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="font-bold text-xl">Radian</span>
        </div>
        
        <div className="flex items-center gap-4">
          <SupabaseStatusPill />
          <ThemeToggle />
        </div>
      </div>
    </header>
  );
} 