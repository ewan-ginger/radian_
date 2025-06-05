import { MainLayout } from "@/components/layout/MainLayout";
import { SchemaDisplay } from "@/components/ui/SchemaDisplay";
import { SupabaseStatus } from "@/components/ui/SupabaseStatus";
import { Database } from "lucide-react";

export const metadata = {
  title: 'Database Setup - Radian',
  description: 'Set up the database schema for the Radian Sports Analytics Dashboard',
};

export default function DatabasePage() {
  return (
    <MainLayout>
      <div className="space-y-6">
        <div className="flex items-center gap-2">
          <Database className="h-6 w-6 text-purple-500" />
          <h1 className="text-3xl font-bold tracking-tight">Database Setup</h1>
        </div>
        
        <p className="text-lg text-muted-foreground">
          Set up the database schema for the Radian Sports Analytics Dashboard
        </p>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          <div className="lg:col-span-2">
            <SchemaDisplay />
          </div>
          
          <div className="space-y-8">
            <SupabaseStatus />
            
            <div className="bg-card rounded-lg border p-6">
              <h2 className="text-xl font-semibold mb-4">Setup Instructions</h2>
              <ol className="space-y-4 list-decimal list-inside text-muted-foreground">
                <li>
                  <span className="font-medium text-foreground">Create a Supabase project</span>
                  <p className="mt-1 ml-6 text-sm">
                    Go to <a href="https://supabase.com" target="_blank" rel="noopener noreferrer" className="underline">supabase.com</a> and create a new project.
                  </p>
                </li>
                <li>
                  <span className="font-medium text-foreground">Update environment variables</span>
                  <p className="mt-1 ml-6 text-sm">
                    Copy your Supabase URL and anon key from the project settings and update the .env.local file.
                  </p>
                </li>
                <li>
                  <span className="font-medium text-foreground">Execute the SQL schema</span>
                  <p className="mt-1 ml-6 text-sm">
                    Copy the SQL schema from the left and execute it in the Supabase SQL Editor.
                  </p>
                </li>
                <li>
                  <span className="font-medium text-foreground">Verify the tables</span>
                  <p className="mt-1 ml-6 text-sm">
                    Check the Table Editor in Supabase to verify that the tables were created successfully.
                  </p>
                </li>
              </ol>
            </div>
          </div>
        </div>
      </div>
    </MainLayout>
  );
} 