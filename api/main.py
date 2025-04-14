import asyncio
import logging
import os
import time
from http import HTTPStatus
from pathlib import Path
from string import Template

import httpx
from dotenv import load_dotenv
import contextlib
import inspect # Add inspect module
from fastapi import FastAPI, HTTPException, Request
from ratelimit import sleep_and_retry, limits
from supabase import Client, create_client, AsyncClient, create_async_client
from supabase.lib.client_options import ClientOptions
from gotrue.errors import AuthApiError # Corrected import based on search
from openai import AsyncOpenAI, OpenAIError # Add OpenAI import

from fastapi import BackgroundTasks # Import BackgroundTasks
# Load prompt template
script_dir = Path(__file__).parent
with open(script_dir / "prompt_template.pt", "r") as f:
    PROMPT_TEMPLATE = Template(f.read())

load_dotenv()

# Helper function to get boolean value from environment variable
def get_bool_env(var_name, default=True):
    value = os.getenv(var_name, str(default))
    return value.lower() in ("1", "true", "yes", "on")

# Logging control variables
LOG_INFO_ENABLED = get_bool_env("LOG_INFO_ENABLED", False)
LOG_DEBUG_ENABLED = get_bool_env("LOG_DEBUG_ENABLED", False)
LOG_WARNING_ENABLED = get_bool_env("LOG_WARNING_ENABLED", False)
LOG_ERROR_ENABLED = get_bool_env("LOG_ERROR_ENABLED", True)

# Load environment variables (already loaded earlier, but ensure it's done before use)
# load_dotenv() # This is typically done once at the start

# Get logging enable flags from environment variables
log_info_enabled = get_bool_env("LOG_INFO_ENABLED", True)
log_debug_enabled = get_bool_env("LOG_DEBUG_ENABLED", True)
log_warning_enabled = get_bool_env("LOG_WARNING_ENABLED", True) # Read for potential future use
log_error_enabled = get_bool_env("LOG_ERROR_ENABLED", True)

# Create logs directory relative to this script file
script_dir = Path(__file__).parent # Ensure logs are relative to main.py
logs_dir = script_dir / "logs"
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
# Basic config - set level low to allow handlers to filter
logging.basicConfig(
    level=logging.DEBUG, # Set to lowest level, handlers will filter
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(), # Keep console output
    ],
    # Force=True might be needed if basicConfig was called elsewhere implicitly
    # force=True
)

# Get root logger to add handlers conditionally
root_logger = logging.getLogger()

# Use logging.info for setup messages AFTER basicConfig is set
logging.info("--- Logging Configuration Start ---")

# Error logger
if log_error_enabled:
    error_handler = logging.FileHandler(str(log_files['error']))
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    root_logger.addHandler(error_handler)
    logging.info("Error logging to file enabled.")
else:
    logging.info("Error logging to file disabled.")

# Info logger (also handles WARNING if enabled)
if log_info_enabled:
    info_handler = logging.FileHandler(str(log_files['info']))
    # This handler will capture INFO, WARNING by default if level is INFO
    info_handler.setLevel(logging.INFO)
    info_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    root_logger.addHandler(info_handler)
    logging.info("Info/Warning logging to file enabled.")
else:
    logging.info("Info/Warning logging to file disabled.")

# Access logger for API requests (remains unconditional)
access_logger = logging.getLogger('access')
access_logger.setLevel(logging.INFO)
access_handler = logging.FileHandler(str(log_files['access']))
access_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
access_logger.addHandler(access_handler)
access_logger.propagate = False # Prevent access logs going to root handlers
logging.info("Access logging to file enabled.")

# Debug logger (separate logger)
debug_logger = logging.getLogger('debug')
if log_debug_enabled:
    debug_logger.setLevel(logging.DEBUG) # Set level on the specific logger
    debug_handler = logging.FileHandler(str(log_files['debug']))
    debug_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
    debug_logger.addHandler(debug_handler)
    debug_logger.propagate = False # Prevent debug logs going to root handlers
    logging.info("Debug logging to file enabled.")
else:
    debug_logger.setLevel(logging.CRITICAL + 1) # Effectively disable the logger
    # Remove handlers if they exist from previous runs? Might be overkill.
    logging.info("Debug logging to file disabled.")

logging.info("--- Logging Configuration End ---")

logger = logging.getLogger(__name__)

# Supabase setup
supabase_url: str = os.getenv("SUPABASE_URL")
supabase_key: str = os.getenv("SUPABASE_KEY")
# supabase: AsyncClient = create_async_client(supabase_url, supabase_key) # Moved to lifespan

# OpenAI Client Setup (using OpenAI library)
openai_api_key: str = os.getenv("OPENAI_API_KEY") # Use OPENAI_API_KEY
gemini_base_url: str = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/") # Keep base URL for Gemini endpoint
gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash") # Keep model for Gemini endpoint

if not openai_api_key: # Check openai_api_key
    if LOG_ERROR_ENABLED:
        logger.error("OPENAI_API_KEY environment variable is not set") # Update error message
    raise EnvironmentError("OPENAI_API_KEY is not set") # Update exception message

if not gemini_base_url:
    if LOG_ERROR_ENABLED:
        logger.error("GEMINI_BASE_URL environment variable is not set")
    raise EnvironmentError("GEMINI_BASE_URL is not set")

if not gemini_model:
    if LOG_ERROR_ENABLED:
        logger.error("GEMINI_MODEL environment variable is not set")
    raise EnvironmentError("GEMINI_MODEL is not set")

# Lifespan context manager for startup/shutdown events
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize Supabase client on startup
    supabase_url: str = os.getenv("SUPABASE_URL")
    supabase_key: str = os.getenv("SUPABASE_KEY")  # Use anon/public key only
    if not supabase_url or not supabase_key:
        if LOG_ERROR_ENABLED:
            logger.error("SUPABASE_URL or SUPABASE_KEY environment variables not set.")
        raise EnvironmentError("Supabase URL or Key not configured.")
    try:
        # Verify environment variables
        if not supabase_url or not supabase_key:
            raise ValueError("Supabase URL or Key not configured")
            
        if LOG_INFO_ENABLED:
            logger.info(f"Initializing Supabase client with URL: {supabase_url[:15]}...")
        
        # Create and await the async client with proper storage
        from gotrue._async.storage import AsyncMemoryStorage
        client_options = ClientOptions(
            headers={"Accept": "application/json"},
            storage=AsyncMemoryStorage()
        )
        supabase_client = await create_async_client(supabase_url, supabase_key, options=client_options) # Key variable name is still supabase_key here
        
        # Verify client creation
        if not isinstance(supabase_client, AsyncClient):
            raise TypeError(f"Expected AsyncClient, got {type(supabase_client)}")
            
        # Verify auth module exists
        if not hasattr(supabase_client, 'auth') or supabase_client.auth is None:
            raise RuntimeError("Supabase auth module not initialized")
            
        # Store client
        app.state.supabase = supabase_client
        if LOG_INFO_ENABLED:
            logger.info(f"Supabase client initialized successfully. Type: {type(supabase_client)}")
        if LOG_DEBUG_ENABLED:
            logger.debug(f"Client methods: {[m for m in dir(supabase_client) if not m.startswith('_')]}")

        # Initialize OpenAI client (for Gemini)
        if LOG_INFO_ENABLED:
            logger.info(f"Initializing OpenAI client for Gemini with base URL: {gemini_base_url}")
        openai_client = AsyncOpenAI(
            api_key=openai_api_key, # Use openai_api_key variable
            base_url=gemini_base_url
        )
        app.state.openai = openai_client
        if LOG_INFO_ENABLED:
            logger.info("OpenAI client for Gemini initialized successfully.")

        yield
    except Exception as e:
        if LOG_ERROR_ENABLED:
            logger.error(f"Failed to initialize Supabase client: {e}")
        raise
    finally:
        # Explicit cleanup of Supabase client
        if hasattr(app.state, 'supabase') and app.state.supabase is not None:
            try:
                if hasattr(app.state.supabase, 'aclose'):
                    await app.state.supabase.aclose()
                    if LOG_INFO_ENABLED:
                        logger.info("Supabase client closed successfully")
                else:
                    if LOG_WARNING_ENABLED:
                        logger.warning("Supabase client missing aclose method")
            except Exception as e:
                if LOG_ERROR_ENABLED:
                    logger.error(f"Error closing Supabase client: {e}")

        # Explicit cleanup of OpenAI client (if needed)
        if hasattr(app.state, 'openai') and app.state.openai is not None:
            try:
                # The openai library >= 1.0 manages connections automatically via httpx.
                # Explicit close is generally not needed unless specific resource cleanup is required.
                # await app.state.openai.close() # Uncomment if explicit close becomes necessary
                if LOG_INFO_ENABLED:
                    logger.info("OpenAI client cleanup check complete.")
            except Exception as e:
                if LOG_ERROR_ENABLED:
                    logger.error(f"Error during OpenAI client cleanup: {e}")

        if LOG_INFO_ENABLED:
            logger.info("Lifespan cleanup finished")

# FastAPI setup
app = FastAPI(lifespan=lifespan)

# Rate limiting setup
MAX_REQUESTS_PER_DAY = int(os.getenv("MAX_REQUESTS_PER_DAY", 1000))
MAX_REQUESTS_PER_MINUTE = int(os.getenv("MAX_REQUESTS_PER_MINUTE", 100))


async def check_user_key(request: Request, user_id: str, api_key: str) -> str:
    """
    Verifies the provided user_id and API key are valid and the user has sufficient credits.
    Assumes 'profiles' table has 'user_id', 'api_key', and 'credits'.
    Accesses Supabase client via request.app.state.supabase.
    """
    # Verify Supabase client is properly initialized and accessible
    if not hasattr(request.app.state, 'supabase'):
        if LOG_ERROR_ENABLED:
            logger.error("Supabase client missing from app.state")
        raise HTTPException(status_code=500, detail="Authentication service not initialized")
        
    supabase_client = request.app.state.supabase
    if supabase_client is None:
        if LOG_ERROR_ENABLED:
            logger.error("Supabase client is None in check_user_key")
        raise HTTPException(status_code=500, detail="Authentication service unavailable")
        
    try:
        # Verify API Key and Credits for the provided user_id
        if LOG_INFO_ENABLED:
            logger.info(f"Verifying API key {api_key} and credits for user ID: {user_id}")
        
        # Use service role client for profile access
        profile_response = await request.app.state.supabase \
            .from_("profiles") \
            .select("api_key, credits") \
            .eq("user_id", user_id) \
            .execute()

        try:
            if LOG_DEBUG_ENABLED:
                logger.debug(f"Supabase query for user {user_id} returned: {profile_response}")
                logger.debug(f"Response data: {profile_response.data}")
                logger.debug(f"Response count: {profile_response.count}")
                
            if not profile_response.data or not isinstance(profile_response.data, list):
                if LOG_ERROR_ENABLED:
                    logger.error(f"No profile data found for user {user_id}. Full response: {profile_response}")
                raise HTTPException(status_code=404, detail="User profile not found")

            try:
                user_profile = profile_response.data[0]  # Get first profile record
                if not isinstance(user_profile, dict):
                    raise ValueError("Profile data is not a dictionary")
                    
                # Check if the provided API key matches the one in the profile
                if user_profile.get("api_key") != api_key:
                    if LOG_WARNING_ENABLED:
                        logger.warning(f"API key mismatch for user ID: {user_id}. Provided: {api_key}, Expected: {user_profile.get('api_key')}")
                    raise HTTPException(status_code=403, detail="Invalid API key for this user")

                # Check credits
                if user_profile.get("credits", 0) <= 0:
                    if LOG_WARNING_ENABLED:
                        logger.warning(f"User ID: {user_id} has insufficient credits ({user_profile.get('credits', 0)}).")
                    raise HTTPException(status_code=403, detail="Insufficient credits")
                    
            except (IndexError, ValueError, AttributeError) as e:
                if LOG_ERROR_ENABLED:
                    logger.error(f"Malformed profile data for user {user_id}: {str(e)}. Data: {profile_response.data}")
                raise HTTPException(status_code=500, detail="Invalid profile data format")

            if LOG_INFO_ENABLED:
                logger.info(f"API key {api_key} validated for user ID: {user_id}. Credits: {user_profile['credits']}")
            
            return user_id  # Return user_id on success

        except Exception as e:
            if LOG_ERROR_ENABLED:
                logger.error(f"Error processing profile for user {user_id}: {str(e)}")
            raise HTTPException(status_code=500, detail="Error processing user profile")

    except HTTPException as http_exc:
        # Re-raise specific HTTPExceptions (e.g., insufficient credits)
        raise http_exc

    except HTTPException as http_exc:
        # Re-raise specific HTTPExceptions (e.g., insufficient credits)
        raise http_exc
    except Exception as e:
        if LOG_ERROR_ENABLED:
            logger.error(f"Error during user verification for ID {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during verification")

async def query_openai(request: Request, prompt: str) -> str:
    """
    Queries the configured Generative AI model (via OpenAI client library) with the given prompt.
    Accesses the client via request.app.state.openai.
    """
    client: AsyncOpenAI = request.app.state.openai
    if not client:
        if LOG_ERROR_ENABLED:
            logger.error("OpenAI client not found in application state.")
        raise HTTPException(status_code=500, detail="AI service not initialized")

    try:
        if LOG_DEBUG_ENABLED:
            logger.debug(f"Sending prompt to model {gemini_model}: {prompt[:100]}...")
        response = await client.chat.completions.create(
            model=gemini_model,
            messages=[
                {"role": "system", "content": "You are a helpful legal assistant."}, # Added system prompt
                {"role": "user", "content": prompt}
            ],
            n=1 # As per user example
        )
        response_text = response.choices[0].message.content
        if not response_text:
            if LOG_WARNING_ENABLED:
                logger.warning("Received empty response from AI model.")
            # Decide how to handle empty response, maybe raise error or return default
            raise HTTPException(status_code=500, detail="AI model returned an empty response.")
        if LOG_DEBUG_ENABLED:
            logger.debug(f"Received response: {response_text[:100]}...")
        return response_text

    except OpenAIError as e: # Catch specific OpenAI errors
        if LOG_ERROR_ENABLED:
            logger.error(f"OpenAI API error: {e} (Type: {type(e).__name__})")
        status_code = 500 # Default internal server error
        detail = f"Error querying AI model: {e}"

        # Check for specific error types if needed (e.g., RateLimitError)
        if isinstance(e, openai.RateLimitError):
            status_code = HTTPStatus.TOO_MANY_REQUESTS
            detail = "AI model rate limit exceeded. Please try again later."
            # Note: Automatic retry logic removed for simplicity, can be added back if needed
            # retry_after = e.response.headers.get("Retry-After") # Check how openai lib exposes this
            # logger.warning(f"Rate limited by AI model. Retry after: {retry_after}")
            # await asyncio.sleep(int(retry_after) if retry_after else 60)
            # return await query_openai(request, prompt) # Recursive retry
        elif isinstance(e, openai.APIConnectionError):
            status_code = HTTPStatus.SERVICE_UNAVAILABLE
            detail = "Could not connect to the AI model service."
        elif isinstance(e, openai.AuthenticationError):
            status_code = HTTPStatus.UNAUTHORIZED
            detail = "AI service authentication failed. Check API key."
        # Add more specific error handling as needed

        raise HTTPException(status_code=status_code, detail=detail)

    except Exception as e:
        # Catch any other unexpected errors
        if LOG_ERROR_ENABLED:
            logger.error(f"Unexpected error querying AI model: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected internal server error: {e}")

async def consume_credit(request: Request, user_id: str):
    """
    Decrements the credit count for the given user_id by 1.
    Uses Supabase RPC for atomic operation to avoid race conditions.
    """
    supabase_client: AsyncClient = request.app.state.supabase
    if not supabase_client:
        if LOG_ERROR_ENABLED:
            logger.error(f"[consume_credit] Supabase client not found for user_id: {user_id}")
        return

    try:
        # First verify the user exists and has credits
        profile_res = await supabase_client.from_("profiles") \
            .select("credits") \
            .eq("user_id", user_id) \
            .single() \
            .execute()

        if not profile_res.data:
            if LOG_ERROR_ENABLED:
                logger.error(f"[consume_credit] No profile found for user_id: {user_id}")
            return

        current_credits = profile_res.data.get("credits", 0)
        if current_credits <= 0:
            if LOG_WARNING_ENABLED:
                logger.warning(f"[consume_credit] User {user_id} has no credits ({current_credits})")
            return

        # Use RPC for atomic decrement
        rpc_response = await supabase_client.rpc(
            "decrement_credits",
            {"user_uuid": user_id}
        ).execute()

        # If the RPC call executes without raising an exception,
        # and the function returns void, we assume success (HTTP 204).
        if LOG_INFO_ENABLED:
            logger.info(f"[consume_credit] Successfully executed decrement RPC for user_id: {user_id}")

    except Exception as e:
        # Log any exception during the RPC call or the preceding checks
        if LOG_ERROR_ENABLED:
            logger.error(f"[consume_credit] Error during credit consumption for user_id {user_id}: {e}", exc_info=True)

    except Exception as e:
        if LOG_ERROR_ENABLED:
            logger.error(f"[consume_credit] Error decrementing credits for user_id {user_id}: {e}", exc_info=True)


    except Exception as e:
        if LOG_ERROR_ENABLED:
            logger.error(f"[consume_credit] Error decrementing credits for user_id {user_id}: {e}", exc_info=True)

@app.post("/legal")
# @limits(calls=MAX_REQUESTS_PER_DAY, period=24 * 60 * 60)
# @limits(calls=MAX_REQUESTS_PER_MINUTE, period=60)
# @sleep_and_retry
async def legal_endpoint(request: Request, background_tasks: BackgroundTasks):
    """
    Endpoint to handle legal requests.
    """
    start_time = time.time()
    client_host = request.client.host if request.client else "unknown"
    access_logger.info(f"Request from {client_host} to /legal endpoint")
    try:
        data = await request.json()
        debug_logger.debug(f"Request data: {data}")
        user_id = data.get("user_id")
        api_key = data.get("api_key")
        user_input = data.get("query")

        if not user_id or not api_key or not user_input:
            raise HTTPException(
                status_code=400, detail="User ID, API key, and query must be provided"
            )

        # Verify user_id and API key
        user_id = await check_user_key(request, user_id, api_key)


        prompt = PROMPT_TEMPLATE.substitute(context="", question=user_input)
        debug_logger.debug(f"Generated prompt: {prompt}")
        # Pass request object to query_openai
        response_text = await query_openai(request, prompt)
        debug_logger.debug(f"AI response: {response_text}") # Update log message variable name

        # Add credit consumption to background tasks
        background_tasks.add_task(consume_credit, request, user_id)
        
        processing_time = time.time() - start_time
        if LOG_INFO_ENABLED:
            logger.info(f"Legal request processed successfully in {processing_time:.2f} seconds")
        access_logger.info(f"Request completed in {processing_time:.2f}s with status 200")
        return {"response": response_text}

    except HTTPException as e:
        error_msg = f"HTTP Exception: {e}"
        if LOG_ERROR_ENABLED:
            logger.error(error_msg)
        access_logger.info(f"Request failed with status {e.status_code}")
        debug_logger.error(f"HTTP Exception details: {e.detail}")
        raise e
    except Exception as e:
        error_msg = f"Exception: {e}"
        if LOG_ERROR_ENABLED:
            logger.error(error_msg)
        access_logger.info("Request failed with status 500")
        debug_logger.error(f"Unexpected error details: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")
