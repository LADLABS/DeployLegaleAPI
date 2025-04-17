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

# --- Logging Configuration FIRST ---
# Determine log level from environment variable, default to INFO
log_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_name, logging.INFO)

# Configure logging to output to stdout for Vercel
logging.basicConfig(
    level=log_level,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler()] # Vercel captures stdout
)
# Get the root logger AFTER basicConfig
logger = logging.getLogger(__name__)
logger.info("API server starting...")
logger.info(f"Environment: {os.getenv('VERCEL_ENV', 'development')}")
logger.info(f"Log level: {log_level_name}")
# --- End Logging Configuration ---

# Load prompt template with error handling
script_dir = Path(__file__).parent
prompt_template_path = script_dir / "prompt_template.pt"
try:
    with open(prompt_template_path, "r") as f:
        PROMPT_TEMPLATE = Template(f.read())
    logger.info(f"Prompt template loaded successfully from {prompt_template_path}")
except FileNotFoundError:
    logger.error(f"Prompt template file not found at {prompt_template_path}. Cannot proceed.")
    # Raising an error here will stop the app, which is correct behavior if the template is essential
    raise FileNotFoundError(f"Required prompt template file not found: {prompt_template_path}")
except Exception as e:
    logger.error(f"Error loading prompt template file {prompt_template_path}: {e}", exc_info=True)
    raise # Re-raise other unexpected errors

# Load .env file from the parent directory (project root)
dotenv_path = Path(__file__).parent.parent / '.env'
logger.info(f"Attempting to load .env file from: {dotenv_path}")
if dotenv_path.is_file():
    load_dotenv(dotenv_path=dotenv_path)
    logger.info(".env file found and loaded.")
else:
    logger.warning(".env file not found at the expected location. Relying on environment variables.")

# Helper function to get boolean value from environment variable
def get_bool_env(var_name, default=True):
    value = os.getenv(var_name, str(default))
    return value.lower() in ("1", "true", "yes", "on")

# Specific loggers can be obtained if needed, but basicConfig covers the root
access_logger = logging.getLogger('access') # Can still use specific loggers
debug_logger = logging.getLogger('debug')   # They will inherit the root config

# Supabase setup
supabase_url: str = os.getenv("SUPABASE_URL")
supabase_key: str = os.getenv("SUPABASE_KEY")
# supabase: AsyncClient = create_async_client(supabase_url, supabase_key) # Moved to lifespan

# OpenAI Client Setup (using OpenAI library)
openai_api_key: str = os.getenv("OPENAI_API_KEY") # Use OPENAI_API_KEY
gemini_base_url: str = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/") # Keep base URL for Gemini endpoint
gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash") # Keep model for Gemini endpoint

# Environment variable checks
logger.info("Checking required environment variables...")
required_vars = {
    "SUPABASE_URL": supabase_url,
    "SUPABASE_KEY": supabase_key,
    "OPENAI_API_KEY": openai_api_key,
    "GEMINI_BASE_URL": gemini_base_url,
    "GEMINI_MODEL": gemini_model
}
missing_vars = [name for name, value in required_vars.items() if not value]

if missing_vars:
    error_message = f"Missing required environment variables: {', '.join(missing_vars)}"
    logger.error(error_message)
    raise EnvironmentError(error_message)
else:
    logger.info("All required environment variables seem to be present.")

# Lifespan context manager for startup/shutdown events
@contextlib.asynccontextmanager
async def lifespan(app_instance: FastAPI): # Renamed app -> app_instance to avoid potential shadowing
    logger.info("Lifespan starting...")
    # Initialize Supabase client on startup
    supabase_url: str = os.getenv("SUPABASE_URL")
    supabase_key: str = os.getenv("SUPABASE_KEY")  # Use anon/public key only
    if not supabase_url or not supabase_key:
        logger.error("SUPABASE_URL or SUPABASE_KEY environment variables not set.")
        raise EnvironmentError("Supabase URL or Key not configured.")
    try:
        # Verify environment variables
        if not supabase_url or not supabase_key:
            raise ValueError("Supabase URL or Key not configured")
            
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
        app_instance.state.supabase = supabase_client # Use app_instance
        logger.info(f"Supabase client initialized successfully. Type: {type(supabase_client)}")
        # Removed debug log for brevity in production logs, can be added back if needed
        # logger.debug(f"Client methods: {[m for m in dir(supabase_client) if not m.startswith('_')]}")

        # Initialize OpenAI client (for Gemini)
        logger.info(f"Initializing OpenAI client for Gemini with base URL: {gemini_base_url}")
        openai_client = AsyncOpenAI(
            api_key=openai_api_key, # Use openai_api_key variable
            base_url=gemini_base_url
        )
        app_instance.state.openai = openai_client # Use app_instance
        logger.info("OpenAI client for Gemini initialized successfully.")

        yield
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client: {e}", exc_info=True) # Add exc_info
        raise
    finally:
        # Explicit cleanup of Supabase client
        if hasattr(app_instance.state, 'supabase') and app_instance.state.supabase is not None: # Use app_instance
            try:
                if hasattr(app_instance.state.supabase, 'aclose'): # Use app_instance
                    await app_instance.state.supabase.aclose() # Use app_instance
                    logger.info("Supabase client closed successfully")
                else:
                    logger.warning("Supabase client missing aclose method")
            except Exception as e:
                logger.error(f"Error closing Supabase client: {e}", exc_info=True)

        # Explicit cleanup of OpenAI client (if needed)
        if hasattr(app_instance.state, 'openai') and app_instance.state.openai is not None: # Use app_instance
            try:
                # The openai library >= 1.0 manages connections automatically via httpx.
                # Explicit close is generally not needed unless specific resource cleanup is required.
                # await app.state.openai.close() # Uncomment if explicit close becomes necessary
                logger.info("OpenAI client cleanup check complete.")
            except Exception as e:
                logger.error(f"Error during OpenAI client cleanup: {e}", exc_info=True)

        logger.info("Lifespan cleanup finished")
    logger.info("Lifespan finished.")

# Log right before app creation
logger.info("Defining FastAPI app instance...")
try:
    app = FastAPI(lifespan=lifespan)
    logger.info("FastAPI app instance defined successfully.")
except Exception as e:
    logger.error(f"Error during FastAPI app instantiation: {e}", exc_info=True)
    raise # Re-raise the exception to ensure it stops execution

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
        logger.error("Supabase client missing from app.state")
        raise HTTPException(status_code=500, detail="Authentication service not initialized")
        
    supabase_client = request.app.state.supabase
    if supabase_client is None:
        logger.error("Supabase client is None in check_user_key")
        raise HTTPException(status_code=500, detail="Authentication service unavailable")
        
    try:
        # Verify API Key and Credits for the provided user_id
        logger.info(f"Verifying API key {'*' * (len(api_key)-4) + api_key[-4:]} and credits for user ID: {user_id}") # Mask API key
        
        # Use service role client for profile access
        profile_response = await request.app.state.supabase \
            .from_("profiles") \
            .select("api_key, credits") \
            .eq("user_id", user_id) \
            .execute()

        try:
            # Removed debug logs for brevity, can be re-enabled via LOG_LEVEL=DEBUG
            # logger.debug(f"Supabase query for user {user_id} returned: {profile_response}")
            # logger.debug(f"Response data: {profile_response.data}")
            # logger.debug(f"Response count: {profile_response.count}")
                
            if not profile_response.data or not isinstance(profile_response.data, list):
                logger.error(f"No profile data found for user {user_id}. Full response: {profile_response}")
                raise HTTPException(status_code=404, detail="User profile not found")

            try:
                user_profile = profile_response.data[0]  # Get first profile record
                if not isinstance(user_profile, dict):
                    raise ValueError("Profile data is not a dictionary")
                    
                # Check if the provided API key matches the one in the profile
                if user_profile.get("api_key") != api_key:
                    logger.warning(f"API key mismatch for user ID: {user_id}. Provided: {'*' * (len(api_key)-4) + api_key[-4:]}") # Mask API key
                    raise HTTPException(status_code=403, detail="Invalid API key for this user")

                # Check credits
                if user_profile.get("credits", 0) <= 0:
                    logger.warning(f"User ID: {user_id} has insufficient credits ({user_profile.get('credits', 0)}).")
                    raise HTTPException(status_code=403, detail="Insufficient credits")
                    
            except (IndexError, ValueError, AttributeError) as e:
                logger.error(f"Malformed profile data for user {user_id}: {str(e)}. Data: {profile_response.data}", exc_info=True)
                raise HTTPException(status_code=500, detail="Invalid profile data format")

            logger.info(f"API key validated for user ID: {user_id}. Credits: {user_profile['credits']}")
            
            return user_id  # Return user_id on success

        except Exception as e:
            logger.error(f"Error processing profile for user {user_id}: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail="Error processing user profile")

    except HTTPException as http_exc:
        # Re-raise specific HTTPExceptions (e.g., insufficient credits)
        raise http_exc
    except Exception as e:
        logger.error(f"Error during user verification for ID {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during verification")

async def query_openai(request: Request, prompt: str) -> str:
    """
    Queries the configured Generative AI model (via OpenAI client library) with the given prompt.
    Accesses the client via request.app.state.openai.
    """
    client: AsyncOpenAI = request.app.state.openai
    if not client:
        logger.error("OpenAI client not found in application state.")
        raise HTTPException(status_code=500, detail="AI service not initialized")

    try:
        logger.debug(f"Sending prompt to model {gemini_model}: {prompt[:100]}...") # Keep debug log, controlled by LOG_LEVEL
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
            logger.warning("Received empty response from AI model.")
            # Decide how to handle empty response, maybe raise error or return default
            raise HTTPException(status_code=500, detail="AI model returned an empty response.")
        logger.debug(f"Received response: {response_text[:100]}...") # Keep debug log
        return response_text

    except OpenAIError as e: # Catch specific OpenAI errors
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
        logger.error(f"Unexpected error querying AI model: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected internal server error: {e}")

async def consume_credit(request: Request, user_id: str):
    """
    Decrements the credit count for the given user_id by 1.
    Uses Supabase RPC for atomic operation to avoid race conditions.
    """
    supabase_client: AsyncClient = request.app.state.supabase
    if not supabase_client:
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
            logger.error(f"[consume_credit] No profile found for user_id: {user_id}")
            return

        current_credits = profile_res.data.get("credits", 0)
        if current_credits <= 0:
            logger.warning(f"[consume_credit] User {user_id} has no credits ({current_credits})")
            return

        # Use RPC for atomic decrement
        rpc_response = await supabase_client.rpc(
            "decrement_credits",
            {"user_uuid": user_id}
        ).execute()

        # If the RPC call executes without raising an exception,
        # and the function returns void, we assume success (HTTP 204).
        logger.info(f"[consume_credit] Successfully executed decrement RPC for user_id: {user_id}")

    except Exception as e:
        # Log any exception during the RPC call or the preceding checks
        logger.error(f"[consume_credit] Error during credit consumption for user_id {user_id}: {e}", exc_info=True)

@app.post("/legal")
async def legal_endpoint(request: Request, background_tasks: BackgroundTasks):
    """
    Endpoint to handle legal requests.
    """
    start_time = time.time()
    client_host = request.client.host if request.client else "unknown"
    access_logger.info(f"Request from {client_host} to /legal endpoint")
    try:
        data = await request.json()
        # Avoid logging potentially sensitive request data by default
        # logger.debug(f"Request data: {data}")
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
        logger.debug(f"Generated prompt (first 100 chars): {prompt[:100]}") # Log only start of prompt
        # Pass request object to query_openai
        response_text = await query_openai(request, prompt)
        logger.debug(f"AI response (first 100 chars): {response_text[:100]}") # Log only start of response

        # Add credit consumption to background tasks
        background_tasks.add_task(consume_credit, request, user_id)
        
        processing_time = time.time() - start_time
        logger.info(f"Legal request processed successfully in {processing_time:.2f} seconds for user {user_id}")
        access_logger.info(f"Request from {client_host} completed in {processing_time:.2f}s with status 200") # Use access_logger here
        return {"response": response_text}

    except HTTPException as e:
        # Log the exception status code and detail
        logger.error(f"HTTP Exception: Status={e.status_code}, Detail={e.detail}")
        access_logger.info(f"Request from {client_host} failed with status {e.status_code}") # Use access_logger
        raise e
    except Exception as e:
        # Log the full exception traceback for unexpected errors
        logger.error(f"Unexpected Exception: {e}", exc_info=True)
        access_logger.info(f"Request from {client_host} failed with status 500") # Use access_logger
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

@app.get("/status", status_code=HTTPStatus.OK)
async def get_status():
    """Simple health check endpoint."""
    logger.info("Health check endpoint /status accessed")
    return {"status": "ok"}

# Add a root endpoint for basic testing
@app.get("/")
async def read_root():
    logger.info("Root endpoint / accessed")
    return {"message": "API is running"}
