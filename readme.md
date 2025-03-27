# LegalAPIg

This project is a Python API designed to integrate with Supabase for user management and credits, and with an OpenAPI endpoint for querying data. It features concurrent processing, HTTP/2 support, throttling, and comprehensive logging.

## Features

-   **Supabase Integration**: User management, credit tracking, and secure data storage.
-   **Concurrent Processing**: Asynchronous handling of requests for optimal performance.
-   **HTTP/2 Support**: Efficient request handling using FastAPI.
-   **Throttling**: Rate limiting to prevent abuse.
-   **OpenAPI Integration**: Dynamic query construction based on user input and templates.
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
    SUPABASE_URL="your_supabase_url"
    SUPABASE_KEY="your_supabase_key"
    OPENAI_API_KEY="your_openai_api_key"
    ```
    Replace the placeholder values with your actual Supabase and OpenAI API keys.

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
       "username": "ladorure@outlook.com    ",
       "password": "Qwerty11",
       "api_key": "872fe3e3-3e92-45dc-9588-8b87e231b344",
       "query": "Comment contester une amende pour stationnement?"
     }'
```

Replace `your_email@example.com`, `your_password`, `your_api_key` with your actual credentials and API key, and update the query as needed.

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

-   `api/`: Contains the main API code (`main.py`).
-   `logs/`: Directory for storing error logs.
-   `.env`: Environment variables for Supabase and OpenAI API keys.
-   `requirements.txt`: Python dependencies.
-   `readme.md`: Project documentation.
-   `vercel.json`: Vercel deployment configuration.
