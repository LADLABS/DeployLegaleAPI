import asyncio
import logging
import os
import time
from http import HTTPStatus

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from ratelimit import RateLimitDecorator, sleep_and_retry, limits
from supabase import Client, create_client

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/error.log"),
    ],
)
logger = logging.getLogger(__name__)

# Supabase setup
supabase_url: str = os.getenv("SUPABASE_URL")
supabase_key: str = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# OpenAI setup
openai_api_key: str = os.getenv("OPENAI_API_KEY")
openai_api_url: str = os.getenv("OPENAI_API_URL")
openai_model: str = os.getenv("OPENAI_MODEL")

if not openai_api_key:
    logger.error("OPENAI_API_KEY is not set")
    raise EnvironmentError("OPENAI_API_KEY is not set")

if not openai_api_url:
    logger.error("OPENAI_API_URL is not set")
    raise EnvironmentError("OPENAI_API_URL is not set")

if not openai_model:
    logger.error("OPENAI_MODEL is not set")
    raise EnvironmentError("OPENAI_MODEL is not set")

# FastAPI setup
app = FastAPI()

# Rate limiting setup
MAX_REQUESTS_PER_DAY = int(os.getenv("MAX_REQUESTS_PER_DAY", 1000))
MAX_REQUESTS_PER_MINUTE = int(os.getenv("MAX_REQUESTS_PER_MINUTE", 100))

daily_limit = RateLimitDecorator(limit=limits(calls=MAX_REQUESTS_PER_DAY, period=24 * 60 * 60), sleep_and_retry=True)
minute_limit = RateLimitDecorator(limit=limits(calls=MAX_REQUESTS_PER_MINUTE, period=60), sleep_and_retry=True)

@daily_limit
@minute_limit
@app.post("/legal")
async def legal_endpoint(request: Request):
    """
    Endpoint to handle legal requests.
    """

async def check_user_key(api_key: str) -> bool:
    """
    Checks if the given API key is valid by querying the Supabase database, using the profiles table.
    """
    try:
        response = await supabase.from_("profiles").select("*").eq("api_key", api_key).execute()
        if response.data and len(response.data) > 0:
            return True
        return False
    except Exception as e:
        logger.error(f"Error checking user key: {e}")
        return False

async def query_openai(prompt: str) -> str:
    """
    Queries the OpenAI API with the given prompt.
    """
    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json",
    }
    data = {
        "model": openai_model,
        "messages": [{"role": "user", "content": prompt}],
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                openai_api_url, headers=headers, json=data
            )
            response.raise_for_status()
            response_text = response.json()["choices"][0]["message"]["content"]
            return response_text
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error querying OpenAI: {e}")
            if e.response.status_code == HTTPStatus.TOO_MANY_REQUESTS:
                retry_after = int(e.response.headers.get("Retry-After", 60))
                logger.warning(f"Rate limited by OpenAI. Retrying after {retry_after} seconds.")
                await asyncio.sleep(retry_after)
                return await query_openai(prompt)  # Retry the request
            else:
                raise HTTPException(
                    status_code=e.response.status_code,
                    detail=f"Error querying OpenAI: {e}",
                )
        except Exception as e:
            logger.error(f"Error querying OpenAI: {e}")
            raise HTTPException(status_code=500, detail=f"Error querying OpenAI: {e}")

@app.post("/legal")
@rate_limit
async def legal_endpoint(request: Request):
    """
    Endpoint to handle legal requests.
    """
    start_time = time.time()
    try:
        data = await request.json()
        api_key = data.get("api_key")
        user_input = data.get("query")

        if not api_key or not user_input:
            raise HTTPException(
                status_code=400, detail="API key and query must be provided"
            )

        if not await check_user_key(api_key):
            raise HTTPException(status_code=403, detail="Invalid API key")

        prompt = PROMPT_TEMPLATE.format(context="", question=user_input)
        response_text = await query_openai(prompt)
        logger.info(f"Legal request processed successfully in {time.time() - start_time:.2f} seconds")
        return {"response": response_text}

    except HTTPException as e:
        logger.error(f"HTTP Exception: {e}")
        raise e
    except Exception as e:
        logger.error(f"Exception: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")
