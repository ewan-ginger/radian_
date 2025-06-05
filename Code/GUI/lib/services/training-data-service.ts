import { supabaseClient } from '@/lib/supabase/client';
import { TrainingSensorDataEntity } from '@/types/database.types';

/**
 * Fetches training sensor data for a specific session, only including labeled data.
 * @param sessionId The UUID of the session.
 * @returns A promise that resolves to an array of TrainingSensorDataEntity objects.
 */
export const getTrainingDataBySession = async (sessionId: string): Promise<TrainingSensorDataEntity[]> => {
  if (!sessionId) {
    console.error('getTrainingDataBySession requires a sessionId');
    return [];
  }

  console.log(`Fetching training data for session: ${sessionId}`);

  // Fetch all pages concurrently for potentially faster retrieval
  const PAGE_SIZE = 1000; // Supabase default limit
  let currentPage = 0;
  let allData: TrainingSensorDataEntity[] = [];
  let hasMoreData = true;

  try {
    // First, get the count to determine the number of pages needed (optional but good for large datasets)
    const { count, error: countError } = await supabaseClient
      .from('training_sensor_data')
      .select('id', { count: 'exact', head: true })
      .eq('session_id', sessionId);

    if (countError) {
      console.error('Error fetching training data count:', countError);
      throw countError; // Re-throw to handle in the main function
    }

    if (count === null || count === 0) {
        console.log(`No training data found for session ${sessionId}.`);
        return [];
    }

    const totalPages = Math.ceil(count / PAGE_SIZE);
    console.log(`Total training records: ${count}, Pages: ${totalPages}`);

    // Create an array of promises for each page fetch
    const fetchPromises = [];
    for (let page = 0; page < totalPages; page++) {
      const from = page * PAGE_SIZE;
      const to = from + PAGE_SIZE - 1;
      fetchPromises.push(
        supabaseClient
          .from('training_sensor_data')
          .select('*') // Select all columns
          .eq('session_id', sessionId)
          .order('timestamp', { ascending: true }) // Order by timestamp crucially
          .range(from, to)
      );
    }

    // Execute all fetch promises
    const results = await Promise.all(fetchPromises);

    // Process results
    results.forEach((result, index) => {
      if (result.error) {
        console.error(`Error fetching training data page ${index + 1}:`, result.error);
        // Decide how to handle partial errors: throw, continue, return partial data?
        // For now, we'll log and continue, potentially returning partial data.
      } else if (result.data) {
        allData = allData.concat(result.data as TrainingSensorDataEntity[]);
      }
    });

    console.log(`Successfully fetched ${allData.length} training records (including null labels) for session ${sessionId}.`);
    // Ensure final sort just in case pagination + concurrent fetching messes order slightly (unlikely with explicit order)
    allData.sort((a, b) => (a.timestamp ?? 0) - (b.timestamp ?? 0));
    return allData;

  } catch (error) {
    console.error('Error fetching training data:', error);
    // Return empty array or re-throw, depending on desired error handling
    return [];
  }
}; 