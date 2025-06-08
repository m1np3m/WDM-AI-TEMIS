"""
Retry utilities for handling network timeouts and connection errors.
"""

import time
import logging
from functools import wraps
from typing import Callable, Tuple, Type, Any

# Setup logger
logger = logging.getLogger(__name__)

# Default retry configuration
DEFAULT_MAX_RETRIES = 5
DEFAULT_BACKOFF_FACTOR = 1.5
DEFAULT_INITIAL_DELAY = 1.0

# Network-related exceptions to retry
NETWORK_EXCEPTIONS = (
    OSError,  # Includes WinError 10060 (connection timeout)
    ConnectionError,
    TimeoutError,
)

try:
    import requests
    NETWORK_EXCEPTIONS += (
        requests.exceptions.RequestException,
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
        requests.exceptions.HTTPError,
    )
except ImportError:
    pass

try:
    from google.api_core import exceptions as gcp_exceptions
    NETWORK_EXCEPTIONS += (
        gcp_exceptions.ServiceUnavailable,
        gcp_exceptions.DeadlineExceeded,
        gcp_exceptions.InternalServerError,
        gcp_exceptions.TooManyRequests,
    )
except ImportError:
    pass


def retry_on_failure(
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
    initial_delay: float = DEFAULT_INITIAL_DELAY,
    retry_on: Tuple[Type[Exception], ...] = NETWORK_EXCEPTIONS,
    reraise_on_final_failure: bool = True
):
    """
    Decorator to retry function calls on specific exceptions.
    
    Args:
        max_retries: Maximum number of retry attempts
        backoff_factor: Multiplier for delay between retries
        initial_delay: Initial delay in seconds
        retry_on: Tuple of exception types to retry on
        reraise_on_final_failure: Whether to reraise the exception after all retries fail
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(max_retries + 1):  # +1 for initial attempt
                try:
                    result = func(*args, **kwargs)
                    if attempt > 0:
                        logger.info(f"✅ {func.__name__} succeeded on attempt {attempt + 1}")
                    return result
                    
                except retry_on as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        delay = initial_delay * (backoff_factor ** attempt)
                        logger.warning(
                            f"🔄 {func.__name__} failed on attempt {attempt + 1}/{max_retries + 1}: {str(e)}\n"
                            f"   Retrying in {delay:.1f} seconds..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"❌ {func.__name__} failed after {max_retries + 1} attempts. Last error: {str(e)}"
                        )
                        
                except Exception as e:
                    # Don't retry on non-network exceptions
                    logger.error(f"❌ {func.__name__} failed with non-retryable error: {str(e)}")
                    raise e
            
            # All retries exhausted
            if reraise_on_final_failure and last_exception:
                raise last_exception
            else:
                logger.warning(f"⚠️  {func.__name__} failed after all retries, returning None")
                return None
                
        return wrapper
    return decorator


def retry_api_call(func: Callable, *args, **kwargs) -> Any:
    """
    Simple function-based retry for API calls.
    Use this for one-off retries without decorating the function.
    """
    retry_decorator = retry_on_failure()
    wrapped_func = retry_decorator(func)
    return wrapped_func(*args, **kwargs)


# Pre-configured decorators for common use cases
retry_network_call = retry_on_failure(
    max_retries=5,
    backoff_factor=2.0,
    initial_delay=1.0,
    retry_on=NETWORK_EXCEPTIONS
)

retry_api_call_gentle = retry_on_failure(
    max_retries=3,
    backoff_factor=1.5,
    initial_delay=0.5,
    retry_on=NETWORK_EXCEPTIONS
)

retry_api_call_aggressive = retry_on_failure(
    max_retries=7,
    backoff_factor=3.0,
    initial_delay=2.0,
    retry_on=NETWORK_EXCEPTIONS
) 