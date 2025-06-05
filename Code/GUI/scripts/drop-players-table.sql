-- Drop the unused players table from the database

-- First check if the table exists before attempting to drop it
DO $$ 
BEGIN
    IF EXISTS (SELECT FROM pg_tables WHERE schemaname = 'public' AND tablename = 'players') THEN
        -- Drop the table
        DROP TABLE public.players;
        RAISE NOTICE 'The players table has been dropped.';
    ELSE
        RAISE NOTICE 'The players table does not exist. No action needed.';
    END IF;
END $$; 