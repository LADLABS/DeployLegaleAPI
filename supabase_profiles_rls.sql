-- Enable Row Level Security
ALTER TABLE profiles ENABLE ROW LEVEL SECURITY;

-- Drop existing policies if they exist
DROP POLICY IF EXISTS "Allow profile reads" ON profiles;
DROP POLICY IF EXISTS "Allow profile updates" ON profiles;

-- Policy 1: Allow authenticated users to read profiles
CREATE POLICY "Allow profile reads"
ON profiles
FOR SELECT
USING (
  auth.role() = 'authenticated' OR
  auth.role() = 'service_role'
);

-- Policy 2: Allow service role to update profiles
CREATE POLICY "Allow profile updates"
ON profiles
FOR UPDATE
TO service_role
USING (true)
WITH CHECK (true);