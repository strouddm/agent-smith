import os
import logging
import requests
from functools import wraps
from typing import List, Dict

from duckduckgo_search import DDGS

# Get logger instance
logger = logging.getLogger(__name__)

# --- Custom Errors & Decorators ---
class SEDError(Exception): pass

def retry_on_error(max_retries: int = 3, delay: float = 1.0):
    """Decorator to retry operations on failure."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {str(e)}")
                    if attempt == max_retries - 1:
                        logger.error(f"Function {func.__name__} failed after {max_retries} attempts.")
                        raise
                    import time
                    time.sleep(delay * (attempt + 1))
            return None
        return wrapper
    return decorator

# --- Tool Classes ---

class WebSearchTool:
    """Tool for performing general internet searches."""
    def __init__(self):
        self.ddgs = DDGS()

    @retry_on_error()
    def run(self, query: str, num_results: int = 8) -> List[Dict[str, str]]:
        logger.info(f"Performing web search for: '{query}'")
        results = self.ddgs.text(query, max_results=num_results)
        if not results:
            return [{"title": "No Results", "snippet": "No web results found.", "url": ""}]
        return [{"title": r.get("title"), "snippet": r.get("body"), "url": r.get("href")} for r in results]

class SEDSearchTool:
    """
    Tool for interacting with the SED document API.
    It now includes methods for both searching and retrieving full documents.
    """
    def __init__(self, api_key: str, base_url: str):
        if not api_key:
            raise ValueError("SED API key is not configured.")
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {"X-API-Key": self.api_key}

    @retry_on_error()
    def search(self, query: str, limit: int = 5) -> List[Dict[str, str]]:
        """Searches for documents and returns a list of summaries."""
        search_url = f"{self.base_url}/search"
        params = {"query": query, "limit": limit}
        logger.info(f"Querying SED API search endpoint for: '{query}'")
        try:
            response = requests.get(search_url, headers=self.headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            if not data or 'documents' not in data:
                return [{"docId": "N/A", "title": "No Results", "summary": "No documents found in SED."}]
            return data['documents']
        except requests.exceptions.RequestException as e:
            logger.error(f"SED API search request failed: {e}")
            raise SEDError(f"Failed to connect to SED API for search: {e}")


