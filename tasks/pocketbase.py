import requests
import logging
import json
import time
import urllib.parse
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

class PocketBaseClient:
    """A client for interacting with the PocketBase API."""

    def __init__(self, base_url, email=None, password=None, token=None, log_prefix="[PocketBaseClient]"):
        self.base_url = base_url.rstrip('/')
        self.email = email
        self.password = password
        self.token = token
        self.log_prefix = log_prefix
        self.session = requests.Session()
        
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            backoff_factor=2  # e.g., 2s, 4s, 8s
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        
        self.session.headers.update({"Content-Type": "application/json"})
        self.timeout = (15, 90)

        if self.token:
            logger.info(f"{self.log_prefix} Initialized with an existing token.")
            self.session.headers.update({"Authorization": self.token})

    def authenticate(self):
        """Authenticates with the PocketBase server and stores the token."""
        if not self.email or not self.password:
            raise ValueError("Authentication requires email and password.")

        auth_url = f"{self.base_url}/api/collections/users/auth-with-password"
        payload = {"identity": self.email, "password": self.password}
        
        try:
            logger.info(f"{self.log_prefix} Attempting authentication...")
            response = self.session.post(auth_url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            self.token = response.json().get('token')
            if not self.token:
                raise ConnectionError("Authentication successful but no token received.")
            self.session.headers.update({"Authorization": self.token})
            logger.info(f"{self.log_prefix} Successfully authenticated.")
            return True
        except requests.exceptions.RequestException as e:
            error_body = e.response.text if hasattr(e, 'response') and e.response else "No response body"
            error_message = f"Failed to authenticate: {e} | Body: {error_body}"
            logger.error(f"{self.log_prefix} {error_message}")
            raise ConnectionError(error_message) from e

    def _make_request(self, method, endpoint, **kwargs):
        """Makes a request, handling re-authentication if necessary."""
        if 'Authorization' not in self.session.headers:
            if self.email and self.password:
                self.authenticate()
            else:
                raise ConnectionError("PocketBase client is not authenticated.")

        kwargs.setdefault('timeout', self.timeout)
        url = f"{self.base_url}{endpoint}"

        try:
            response = self.session.request(method, url, **kwargs)
            if response.status_code == 401:
                logger.warning(f"{self.log_prefix} Token may have expired. Re-authenticating...")
                self.authenticate()
                response = self.session.request(method, url, **kwargs)

            response.raise_for_status()
            return response.json() if response.content else {}
        except requests.exceptions.RequestException as e:
            logger.error(f"{self.log_prefix} API request failed for {method} {endpoint}: {e}")
            if hasattr(e, 'response') and e.response is not None:
                response_text = e.response.text
                if "validation_not_unique" in response_text:
                    logger.warning(f"{self.log_prefix} Request failed, likely due to an already existing song.")
                else:
                    logger.error(f"{self.log_prefix} Response status: {e.response.status_code}, body: {response_text}")
            raise

    def _sanitize_for_filter(self, value):
        # Escape backslashes first, then quotes
        return value.replace('\\', '\\\\').replace('"', '\\"')

    def get_records_by_artists(self, artists, collection):
        """
        Fetches all records for a list of artists using a GET request with a shorter filter.
        """
        if not artists:
            return []
        
        endpoint = f"/api/collections/{collection}/records"
        
        # Chunk artists to keep URL length reasonable
        ARTIST_CHUNK_SIZE = 5
        all_records = []
        
        artist_chunks = [artists[i:i + ARTIST_CHUNK_SIZE] for i in range(0, len(artists), ARTIST_CHUNK_SIZE)]

        for chunk in artist_chunks:
            filter_parts = []
            for artist in chunk:
                sanitized_artist = self._sanitize_for_filter(artist)
                filter_parts.append(f'artist="{sanitized_artist}"')

            filter_query = " || ".join(filter_parts)
            
            # Use a large perPage value since we are filtering by a limited number of artists
            params = {
                'filter': filter_query,
                'perPage': 500, 
            }
            
            try:
                response_data = self._make_request('GET', endpoint, params=params)
                all_records.extend(response_data.get('items', []))
            except requests.exceptions.RequestException as e:
                logger.error(f"{self.log_prefix} FATAL: Could not retrieve records from '{collection}' for artists. Reason: {e}")
                raise e # Re-raise to fail the task
        
        return all_records

    def create_records_batch(self, records, collection):
        """
        Creates multiple records in a specified collection using a single batch API call.
        """
        if not records:
            return True

        batch_endpoint = "/api/batch"
        requests_payload = []
        for record in records:
            # Create a copy to modify
            processed_body = record.copy()
            
            # The 'embedding' field in the 'embedding' collection must be a JSON string.
            # The 'embedding' field does not exist in the 'score' collection, so this is safe.
            if 'embedding' in processed_body and isinstance(processed_body['embedding'], list):
                processed_body['embedding'] = json.dumps(processed_body['embedding'])

            requests_payload.append({
                "method": "POST",
                "url": f"/api/collections/{collection}/records",
                "body": processed_body
            })

        payload = {"requests": requests_payload}
        
        # This is a critical write operation, so we don't use the built-in retry
        # and instead handle it with specific logic in the calling task.
        # We re-raise the exception to allow the task to decide how to handle it.
        try:
            self._make_request('POST', batch_endpoint, json=payload)
            logger.info(f"{self.log_prefix} Successfully submitted batch request for {len(records)} records to collection '{collection}'.")
        except requests.exceptions.RequestException:
             # The detailed error is already logged in _make_request, just re-raise
            logger.error(f"{self.log_prefix} The entire batch request to collection '{collection}' failed.")
            raise

