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
    # Check against various truthy string values
    return value.lower() in ('true', '1', 't', 'y', 'yes')

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
    logger.error("OPENAI_API_KEY environment variable is not set") # Update error message
    raise EnvironmentError("OPENAI_API_KEY is not set") # Update exception message

if not gemini_base_url:
    logger.error("GEMINI_BASE_URL environment variable is not set")
    raise EnvironmentError("GEMINI_BASE_URL is not set")

if not gemini_model:
    logger.error("GEMINI_MODEL environment variable is not set")
    raise EnvironmentError("GEMINI_MODEL is not set")

# Lifespan context manager for startup/shutdown events
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize Supabase client on startup
    supabase_url: str = os.getenv("SUPABASE_URL")
    supabase_key: str = os.getenv("SUPABASE_KEY")
    if not supabase_url or not supabase_key:
        logger.error("SUPABASE_URL or SUPABASE_KEY environment variables not set.")
        raise EnvironmentError("Supabase URL/Key not configured.")
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
        supabase_client = await create_async_client(supabase_url, supabase_key, options=client_options)
        
        # Verify client creation
        if not isinstance(supabase_client, AsyncClient):
            raise TypeError(f"Expected AsyncClient, got {type(supabase_client)}")
            
        # Verify auth module exists
        if not hasattr(supabase_client, 'auth') or supabase_client.auth is None:
            raise RuntimeError("Supabase auth module not initialized")
            
        # Store client
        app.state.supabase = supabase_client
        logger.info(f"Supabase client initialized successfully. Type: {type(supabase_client)}")
        logger.debug(f"Client methods: {[m for m in dir(supabase_client) if not m.startswith('_')]}")

        # Initialize OpenAI client (for Gemini)
        logger.info(f"Initializing OpenAI client for Gemini with base URL: {gemini_base_url}")
        openai_client = AsyncOpenAI(
            api_key=openai_api_key, # Use openai_api_key variable
            base_url=gemini_base_url
        )
        app.state.openai = openai_client
        logger.info("OpenAI client for Gemini initialized successfully.")

        yield
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client: {e}")
        raise
    finally:
        # Explicit cleanup of Supabase client
        if hasattr(app.state, 'supabase') and app.state.supabase is not None:
            try:
                if hasattr(app.state.supabase, 'aclose'):
                    await app.state.supabase.aclose()
                    logger.info("Supabase client closed successfully")
                else:
                    logger.warning("Supabase client missing aclose method")
            except Exception as e:
                logger.error(f"Error closing Supabase client: {e}")

        # Explicit cleanup of OpenAI client (if needed)
        if hasattr(app.state, 'openai') and app.state.openai is not None:
            try:
                # The openai library >= 1.0 manages connections automatically via httpx.
                # Explicit close is generally not needed unless specific resource cleanup is required.
                # await app.state.openai.close() # Uncomment if explicit close becomes necessary
                logger.info("OpenAI client cleanup check complete.")
            except Exception as e:
                logger.error(f"Error during OpenAI client cleanup: {e}")

        logger.info("Lifespan cleanup finished")

# FastAPI setup
app = FastAPI(lifespan=lifespan)

# Rate limiting setup
MAX_REQUESTS_PER_DAY = int(os.getenv("MAX_REQUESTS_PER_DAY", 1000))
MAX_REQUESTS_PER_MINUTE = int(os.getenv("MAX_REQUESTS_PER_MINUTE", 100))


async def check_user_key(request: Request, username: str, password: str, api_key: str) -> str:
    """
    Authenticates the user with username/password, then checks if the provided
    API key is valid for that user and if they have sufficient credits.
    Assumes 'profiles' table has 'user_id' (matching auth.users.id), 'api_key', and 'credits'.
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
        
    logger.debug(f"Supabase client type: {type(supabase_client)}")
    logger.debug(f"Supabase client methods: {[m for m in dir(supabase_client) if not m.startswith('_')]}")
    
    # Verify client is properly initialized and has required modules
    required_attrs = ['auth', 'from_']
    for attr in required_attrs:
        if not hasattr(supabase_client, attr):
            logger.error(f"Supabase client missing required attribute: {attr}")
            raise HTTPException(
                status_code=500,
                detail="Authentication service misconfigured"
            )
        if getattr(supabase_client, attr) is None:
            logger.error(f"Supabase client attribute {attr} is None")
            raise HTTPException(
                status_code=500,
                detail="Authentication service component unavailable"
            )

    try:
        # 1. Authenticate user
        logger.info(f"Attempting login for user: {username}")
        
        # Detailed auth module verification
        logger.debug(f"Checking supabase_client.auth before sign-in. Type: {type(getattr(supabase_client, 'auth', None))}")
        if not hasattr(supabase_client, 'auth') or supabase_client.auth is None:
            logger.error(f"supabase_client.auth is missing or None. Client details: {dir(supabase_client)}")
            raise HTTPException(
                status_code=500,
                detail="Authentication service unavailable. Please try again later."
            )
        
        # Verify auth module has required method
        if not hasattr(supabase_client.auth, 'sign_in_with_password'):
            logger.error(f"supabase_client.auth missing sign_in_with_password method. Available methods: {dir(supabase_client.auth)}")
            raise HTTPException(
                status_code=500,
                detail="Authentication service misconfigured. Please contact support."
            )
            
        logger.debug(f"Auth module verified. Proceeding with sign_in_with_password.")

        # Explicitly check if sign_in_with_password is callable before awaiting
        sign_in_method = getattr(supabase_client.auth, 'sign_in_with_password', None)
        if not callable(sign_in_method):
            logger.error(f"sign_in_with_password is not callable or is None. Type: {type(sign_in_method)}")
            raise HTTPException(
                status_code=500,
                detail="Authentication service component unavailable (sign_in method)."
            )
        logger.debug(f"sign_in_with_password method found and is callable. Proceeding with await.")

        try:
            # Call the method first to inspect its return value
            sign_in_call_result = sign_in_method({
                "email": username.strip(), # Trim whitespace from username
                "password": password,
            })

            # Log the type of the result
            logger.debug(f"sign_in_method returned object of type: {type(sign_in_call_result)}")

            # Check if the result is awaitable
            if not inspect.isawaitable(sign_in_call_result):
                logger.error(f"sign_in_method did not return an awaitable object. Got: {sign_in_call_result}")
                # Raise specific error indicating the non-awaitable return type
                raise HTTPException(
                    status_code=500,
                    detail=f"Authentication service error: sign_in method returned non-awaitable type '{type(sign_in_call_result).__name__}'."
                )

            # If the check passes, await the result
            logger.debug("sign_in_method returned an awaitable object. Proceeding with await.")
            auth_response = await sign_in_call_result

            user = auth_response.user
            if not user:
                # This case might not be reached if sign_in throws error, but good practice
                logger.warning(f"Authentication failed for user: {username} - No user object returned after await.")
                raise HTTPException(status_code=401, detail="Invalid credentials")

            logger.info(f"User {username} authenticated successfully. User ID: {user.id}")
        except AuthApiError as auth_error:
            logger.warning(f"Authentication failed for user {username}: {auth_error.message}")
            raise HTTPException(status_code=401, detail=f"Invalid credentials: {auth_error.message}")
        except HTTPException as http_exc: # Catch specific HTTPExceptions raised above
            raise http_exc
        except Exception as e:
            # Log the original exception type as well
            logger.error(f"Unexpected error during authentication for {username} ({type(e).__name__}): {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Authentication service error")

        # 2. Verify API Key and Credits for the authenticated user
        logger.info(f"Verifying API key {api_key} and credits for user ID: {user.id}")
        profile_response = await supabase_client.from_("profiles") \
            .select("api_key, credits") \
            .eq("user_id", user.id) \
            .single() \
            .execute()

        if profile_response.data:
            user_profile = profile_response.data
            # Check if the provided API key matches the one in the profile
            if user_profile.get("api_key") != api_key:
                logger.warning(f"API key mismatch for user {username} (ID: {user.id}). Provided: {api_key}, Expected: {user_profile.get('api_key')}")
                raise HTTPException(status_code=403, detail="Invalid API key for this user")

            # Check credits
            if user_profile.get("credits", 0) > 0:
                logger.info(f"API key {api_key} validated for user {username}. Credits: {user_profile['credits']}")
                return user.id # Return user_id on success
            else:
                logger.warning(f"User {username} (ID: {user.id}) has insufficient credits ({user_profile.get('credits', 0)}).")
                raise HTTPException(status_code=403, detail="Insufficient credits")
        else:
            logger.warning(f"No profile found for authenticated user {username} (ID: {user.id}).")
            # Decide if this is a 403 (forbidden, profile issue) or implies invalid API key logic
            raise HTTPException(status_code=403, detail="User profile not found or setup incorrectly")

    except AuthApiError as auth_error: # Use the imported AuthApiError
        logger.warning(f"Authentication failed for user {username}: {auth_error.message}")
        raise HTTPException(status_code=401, detail=f"Invalid credentials: {auth_error.message}")
    except HTTPException as http_exc:
        # Re-raise specific HTTPExceptions (e.g., insufficient credits)
        raise http_exc
    except Exception as e:
        logger.error(f"Error during user check/authentication for {username}: {e}")
        # Generic error for unexpected issues
        raise HTTPException(status_code=500, detail="Internal server error during authentication/authorization")

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
            logger.warning("Received empty response from AI model.")
            # Decide how to handle empty response, maybe raise error or return default
            raise HTTPException(status_code=500, detail="AI model returned an empty response.")
        logger.debug(f"Received response: {response_text[:100]}...")
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

    except Exception as e:
        logger.error(f"[consume_credit] Error decrementing credits for user_id {user_id}: {e}", exc_info=True)


    except Exception as e:
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
        username = data.get("username")
        password = data.get("password")
        api_key = data.get("api_key") # Keep api_key for now, might adjust logic later
        user_input = data.get("query")

        if not username or not password or not api_key or not user_input:
            raise HTTPException(
                status_code=400, detail="Username, password, API key, and query must be provided"
            )

        # Pass username and password along with api_key
        # check_user_key now returns user_id on success or raises HTTPException on failure
        user_id = await check_user_key(request, username, password, api_key)


        prompt = PROMPT_TEMPLATE.substitute(context="", question=user_input)
        debug_logger.debug(f"Generated prompt: {prompt}")
        # Pass request object to query_openai
        response_text = await query_openai(request, prompt)
        debug_logger.debug(f"AI response: {response_text}") # Update log message variable name

        # Add credit consumption to background tasks
        background_tasks.add_task(consume_credit, request, user_id)
        
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
