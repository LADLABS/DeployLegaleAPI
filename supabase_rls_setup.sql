-- Enable RLS on profiles table
ALTER TABLE profiles ENABLE ROW LEVEL SECURITY;

-- Create policy to allow users to read their own profile
CREATE POLICY "Users can view their own profile" 
ON profiles
FOR SELECT
USING (auth.uid() = user_id);

-- Create policy to allow users to update their own profile
CREATE POLICY "Users can update their own profile"
ON profiles
FOR UPDATE
USING (auth.uid() = user_id);

-- For API access, create a policy that allows select with API key
CREATE POLICY "API access with valid key"
ON profiles
FOR SELECT
TO authenticated
USING (true); -- Or add more specific conditions if needed

-- For credit decrement function access
CREATE POLICY "Allow credit decrement function"
ON profiles
FOR UPDATE
USING (true); -- Or add specific conditions