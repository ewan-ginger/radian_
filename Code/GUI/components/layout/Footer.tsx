"use client";

import Link from "next/link";
import { Github } from "lucide-react";

export function Footer() {
  const currentYear = new Date().getFullYear();
  
  return (
    <footer className="border-t bg-background">
      <div className="container flex h-14 items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="font-bold">Radian</span>
          <span className="text-xs text-muted-foreground">© {currentYear}</span>
        </div>
        
        <Link 
          href="https://github.com/yourusername/radian" 
          target="_blank" 
          rel="noopener noreferrer"
          className="flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground transition-colors"
        >
          <Github className="h-4 w-4" />
          <span>GitHub</span>
        </Link>
      </div>
    </footer>
  );
} 