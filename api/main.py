import asyncio
import logging
import os
import time
from http import HTTPStatus
from pathlib import Path
from string import Template

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from ratelimit import sleep_and_retry, limits
from supabase import Client, create_client

# Load prompt template
script_dir = Path(__file__).parent
with open(script_dir / "prompt_template.pt", "r") as f:
    PROMPT_TEMPLATE = Template(f.read())

load_dotenv()

# Create logs directory and files if they don't exist
logs_dir = Path("logs")
logs_dir.mkdir(parents=True, exist_ok=True)

log_files = {
    'error': logs_dir / 'error.log',
    'access': logs_dir / 'access.log',
    'info': logs_dir / 'info.log',
    'debug': logs_dir / 'debug.log'
}

# Create log files if they don't exist
for log_file in log_files.values():
    log_file.touch(exist_ok=True)

# Configure logging
# Main logger for error and info
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)

# Error logger
error_handler = logging.FileHandler(str(log_files['error']))
error_handler.setLevel(logging.ERROR)
error_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logging.getLogger().addHandler(error_handler)

# Info logger
info_handler = logging.FileHandler(str(log_files['info']))
info_handler.setLevel(logging.INFO)
info_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logging.getLogger().addHandler(info_handler)

# Access logger for API requests
access_logger = logging.getLogger('access')
access_logger.setLevel(logging.INFO)
access_handler = logging.FileHandler(str(log_files['access']))
access_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
access_logger.addHandler(access_handler)

# Debug logger
debug_logger = logging.getLogger('debug')
debug_logger.setLevel(logging.DEBUG)
debug_handler = logging.FileHandler(str(log_files['debug']))
debug_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
debug_logger.addHandler(debug_handler)

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
@limits(calls=MAX_REQUESTS_PER_DAY, period=24 * 60 * 60)
@limits(calls=MAX_REQUESTS_PER_MINUTE, period=60)
@sleep_and_retry
async def legal_endpoint(request: Request):
    """
    Endpoint to handle legal requests.
    """
    start_time = time.time()
    client_host = request.client.host if request.client else "unknown"
    access_logger.info(f"Request from {client_host} to /legal endpoint")
    try:
        data = await request.json()
        debug_logger.debug(f"Request data: {data}")
        api_key = data.get("api_key")
        user_input = data.get("query")

        if not api_key or not user_input:
            raise HTTPException(
                status_code=400, detail="API key and query must be provided"
            )

        if not await check_user_key(api_key):
            raise HTTPException(status_code=403, detail="Invalid API key")

        prompt = PROMPT_TEMPLATE.substitute(context="", question=user_input)
        debug_logger.debug(f"Generated prompt: {prompt}")
        response_text = await query_openai(prompt)
        debug_logger.debug(f"OpenAI response: {response_text}")
        
        processing_time = time.time() - start_time
        logger.info(f"Legal request processed successfully in {processing_time:.2f} seconds")
        access_logger.info(f"Request completed in {processing_time:.2f}s with status 200")
        return {"response": response_text}

    except HTTPException as e:
        error_msg = f"HTTP Exception: {e}"
        logger.error(error_msg)
        access_logger.info(f"Request failed with status {e.status_code}")
        debug_logger.error(f"HTTP Exception details: {e.detail}")
        raise e
    except Exception as e:
        error_msg = f"Exception: {e}"
        logger.error(error_msg)
        access_logger.info("Request failed with status 500")
        debug_logger.error(f"Unexpected error details: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")
