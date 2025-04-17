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

3. Create virtual env:
    ```bash
    python3 -m venv myenv
    source myenv/bin/activate
    ```

4. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

5. Create a `.env` file in the `LegalAPIg` directory with the following content:
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

## API Endpoints

### `/legal` Endpoint

This endpoint processes legal queries using the configured AI model.

**Request:**

```bash
# For deployed version
curl -X POST https://qlegal2.vercel.app/legal \
     -H "Content-Type: application/json" \
     -d '{
       "user_id": "564bac7a-60f9-4056-99c7-428ff45d8887",
       "api_key": "872fe3e3-3e92-45dc-9588-8b87e231b344",
       "query": "Comment contester une amende pour stationnement?"
     }'

# For local development
curl -X POST http://127.0.0.1:8000/legal \
     -H "Content-Type: application/json" \
     -d '{
       "user_id": "564bac7a-60f9-4056-99c7-428ff45d8887",
       "api_key": "872fe3e3-3e92-45dc-9588-8b87e231b344",
       "query": "Comment contester une amende pour stationnement?"
     }'
```

Replace `user_id` with your actual user ID and `api_key` with your API key, and update the query as needed.

**Successful Response (Status Code 200):**

```json
{
  "response": "Pour contester une amende de stationnement, voici les étapes à suivre:\n\n1. Délais:\n- Vous avez 30 jours à partir de la date de l'infraction pour contester\n- Le délai commence à la date inscrite sur l'avis d'infraction\n\n2. Options de contestation:\n- En ligne sur le site de la cour municipale\n- Par la poste en envoyant le formulaire de contestation\n- En personne au bureau de la cour municipale\n\n3. Documents nécessaires:\n- L'avis d'infraction original\n- Preuves justifiant la contestation (photos, reçus, etc.)\n- Pièce d'identité valide\n\n4. Procédure:\n- Remplir le formulaire de plaidoyer de non-culpabilité\n- Joindre les preuves pertinentes\n- Conserver une copie des documents envoyés\n\n5. Suivi:\n- Vous recevrez un avis de convocation pour l'audience\n- Préparez votre défense et vos arguments\n\nImportant: Le dépôt d'une contestation suspend l'obligation de paiement jusqu'au jugement."
}
```

### `/status` Endpoint

This endpoint provides a simple health check to confirm the API is running.

**Request:**

```bash
# For deployed version
curl https://questionlegale.vercel.app/status

# For local development
curl http://127.0.0.1:8000/status
```

**Successful Response (Status Code 200):**

```json
{
  "status": "ok"
}
```

## Deployment on Vercel

1. Install the Vercel CLI:
    ```bash
    npm install -g vercel
    ```

2. Configure Environment Variables:
   - Go to your Vercel dashboard
   - Select your project
   - Navigate to Settings > Environment Variables
   - Add all the variables from your `.env` file

3. Deploy the project:
    ```bash
    vercel
    ```

4. For production deployment:
    ```bash
    vercel --prod
    ```

The deployment process will automatically:
- Detect your Python project
- Install dependencies from requirements.txt
- Configure the ASGI server for FastAPI
- Set up the routes based on vercel.json

## Project Structure

-   `api/`: Contains the main API code
    - `main.py`: Main FastAPI application
    - `prompt_template.pt`: Template for AI model prompts
-   `.env`: Environment variables
-   `requirements.txt`: Python dependencies
-   `readme.md`: Project documentation
-   `vercel.json`: Vercel deployment configuration
-   `supabase_profiles_rls.sql`: Supabase RLS policies
-   `supabase_rls_setup.sql`: Supabase setup scripts

## Error Handling

The API includes comprehensive error handling for:
- Invalid/missing credentials
- Insufficient credits
- Rate limiting
- AI model errors
- Database connection issues

All errors return appropriate HTTP status codes and descriptive messages.

---
