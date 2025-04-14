# LegalAPIg

This project is a Python API designed to integrate with Supabase for user management and credits, and with an OpenAPI endpoint for querying data. It features concurrent processing, HTTP/2 support, throttling, and comprehensive logging.

## Features

-   **Supabase Integration**: User management, credit tracking, and secure data storage.
-   **Concurrent Processing**: Asynchronous handling of requests for optimal performance.
-   **HTTP/2 Support**: Efficient request handling using FastAPI.
-   **Throttling**: Rate limiting to prevent abuse.
-   **AI Model Integration**: Uses the `openai` library to interact with a configurable generative AI endpoint (e.g., Google Gemini via its OpenAI-compatible endpoint). Dynamic query construction based on user input and templates.
-   **Logging**: User request logging and error logging.

## Setup

1. Clone the repository:
    ```bash
    git clone <repository_url>
    ```
2. Navigate to the project directory:
    ```bash
    cd LegalAPIg
    ```
Create virtual env
  
   python3 -m venv myenv

source myenv/bin/activate


3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Create a `.env` file in the `LegalAPIg` directory with the following content:
    ```
    # Supabase Credentials
    SUPABASE_URL="your_supabase_url"
    SUPABASE_KEY="your_supabase_key"

    # AI Model Configuration (using OpenAI library)
    OPENAI_API_KEY="your_api_key_for_the_endpoint" # API key for the target endpoint
    GEMINI_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/" # Base URL of the OpenAI-compatible endpoint (defaults to Google Gemini)
    GEMINI_MODEL="gemini-2.0-flash" # Model name to use at the endpoint (defaults to gemini-2.0-flash)
    ```
    Replace the placeholder values with your actual Supabase credentials and the appropriate API key, base URL, and model name for your target AI endpoint.

## Usage

To run the API locally:

```bash
cd api
uvicorn main:app --reload
```

This will start the API server on `http://127.0.0.1:8000`.

## Example Request

```bash
curl -X POST http://127.0.0.1:8000/legal \
     -H "Content-Type: application/json" \
     -d '{
       "user_id": "564bac7a-60f9-4056-99c7-428ff45d8887",
       "api_key": "872fe3e3-3e92-45dc-9588-8b87e231b344",
       "query": "Comment contester une amende pour stationnement?"
     }'
```

Replace `user_id` with your actual user ID and `api_key` with your API key, and update the query as needed.

## Deployment

This project is configured for deployment on Vercel. To deploy:

1. Install the Vercel CLI:
    ```bash
    npm install -g vercel
    ```
2. Deploy the project:
    ```bash
    vercel
    ```

## Project Structure

-   `api/`: Contains the main API code (`main.py`) and the prompt template (`prompt_template.pt`).
-   `logs/`: Directory for storing log files (created automatically).
-   `.env`: Environment variables for Supabase credentials and AI model configuration.
-   `requirements.txt`: Python dependencies.
-   `readme.md`: Project documentation (this file).
-   `vercel.json`: Vercel deployment configuration.


---Supabase Decrement

--Decrement Credit function
create or replace function decrement_credits(user_uuid uuid) 
returns void as $$
  update profiles 
  set credits = credits - 1 
  where user_id = user_uuid and credits > 0;
$$ language sql;


-------------------

Okay, Row Level Security (RLS) is a very common reason for this behavior. The SQL function might be correct, but the security policies prevent the API (running as a specific role, often authenticated) from actually performing the UPDATE operation defined within that function.

Please go to your Supabase Dashboard:

Navigate to Authentication -> Policies.
Find the profiles table in the list.
Look for policies related to the UPDATE operation.
You need an UPDATE policy that allows the role your API uses to modify the credits field for the relevant user. A common policy allows users to update their own profile. It might look something like this:

Policy Name: Allow authenticated users to update their own profile (or similar)
Target Roles: authenticated (or the specific role your API key corresponds to)
Operation: UPDATE
USING expression: (auth.uid() = user_id)
WITH CHECK expression: (auth.uid() = user_id)

