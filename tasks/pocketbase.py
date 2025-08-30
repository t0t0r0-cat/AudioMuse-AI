import requests
import logging
import json
import numpy as np
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

class PocketBaseClient:
    """A client for interacting with the PocketBase API with built-in retries."""

    def __init__(self, base_url, email=None, password=None, token=None):
        self.base_url = base_url.rstrip('/')
        self.email = email
        self.password = password
        self.token = token
        self.session = requests.Session()
        
        retry_strategy = Retry(
            total=5,
            status_forcelist=[429, 500, 502, 503, 504],
            backoff_factor=2,
            respect_retry_after_header=True
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        self.session.headers.update({"Content-Type": "application/json"})
        self.timeout = (15, 90)

        if self.token:
            self.session.headers.update({"Authorization": self.token})
            logger.info("PocketBaseClient initialized with an existing token.")

    def authenticate(self):
        if not self.email or not self.password:
            raise ValueError("Authentication attempted without providing email and password.")
        
        logger.info("Attempting to authenticate with PocketBase to get a new token...")
        auth_url = f"{self.base_url}/api/collections/users/auth-with-password"
        payload = {"identity": self.email, "password": self.password}
        try:
            response = self.session.post(auth_url, json=payload, timeout=(10, 20))
            response.raise_for_status()
            self.token = response.json().get('token')
            if not self.token:
                raise ConnectionError("Authentication successful but no token received.")
            self.session.headers.update({"Authorization": self.token})
            logger.info("Successfully authenticated and received new token.")
            return True
        except requests.exceptions.RequestException as e:
            error_message = f"Failed to authenticate with PocketBase: {e}"
            if hasattr(e, 'response') and e.response is not None:
                error_message += f" | Status: {e.response.status_code}, Body: {e.response.text}"
            logger.error(error_message)
            raise ConnectionError(error_message) from e

    def _make_request(self, method, endpoint, **kwargs):
        if not self.token:
            logger.warning("No token found. Attempting to authenticate first.")
            if not (self.email and self.password and self.authenticate()):
                raise ConnectionError("PocketBase authentication required but failed.")
        
        request_kwargs = kwargs.copy()
        request_kwargs.setdefault('timeout', self.timeout)

        url = f"{self.base_url}{endpoint}"
        try:
            response = self.session.request(method, url, **request_kwargs)
            if response.status_code == 401:
                logger.warning("Token expired or invalid (401). Attempting re-authentication.")
                if self.email and self.password and self.authenticate():
                    response = self.session.request(method, url, **request_kwargs)
                else:
                    raise ConnectionError("Token expired and re-authentication is not possible.")

            response.raise_for_status()
            return {} if response.status_code == 204 else response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"PocketBase API request failed for {method} {endpoint}: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response status: {e.response.status_code}, body: {e.response.text}")
            raise

    def get_records_by_songs(self, songs, collection='embedding'):
        if not songs: return []
            
        def escape_pb_filter_value(value):
            if value is None: return ""
            return str(value).replace('\\', '\\\\').replace('"', '""').replace("'", "''")

        filter_parts = []
        for song in songs:
            artist = song.get('artist')
            title = song.get('title')
            if artist and title:
                escaped_artist = escape_pb_filter_value(artist)
                escaped_title = escape_pb_filter_value(title)
                filter_parts.append(f'(artist="{escaped_artist}" && title="{escaped_title}")')

        if not filter_parts: return []
        filter_query = " || ".join(filter_parts)

        endpoint = f"/api/collections/{collection}/records"
        params = {'filter': filter_query, 'perPage': len(songs) + 5}
        
        try:
            response_data = self._make_request('GET', endpoint, params=params)
            return response_data.get('items', [])
        except Exception as e:
            logger.error(f"FATAL: Could not retrieve records from '{collection}'. Failing task. Reason: {e}")
            raise e

    def get_single_record_by_artist_title(self, artist, title, collection='score'):
        records = self.get_records_by_songs([{'artist': artist, 'title': title}], collection=collection)
        return records[0] if records else None

    def create_records_batch(self, records, collection='embedding'):
        if not records: return True

        batch_endpoint = "/api/batch"
        requests_payload = [
            {"method": "POST", "url": f"/api/collections/{collection}/records", "body": record}
            for record in records
        ]
        payload = {"requests": requests_payload}

        try:
            response_data = self._make_request('POST', batch_endpoint, json=payload)
            if not isinstance(response_data, list):
                logger.error(f"Batch create failed. Expected a list in response, got {type(response_data)}")
                return False

            all_successful = True
            for i, res in enumerate(response_data):
                status_code = res.get('statusCode', 500)
                if not (200 <= status_code < 300):
                    all_successful = False
                    original_record = records[i]
                    error_details = res.get('data', {})
                    logger.error("---- POCKETBASE BATCH SUB-REQUEST FAILURE ---")
                    logger.error(f"    Collection: {collection}")
                    logger.error(f"    Record: '{original_record.get('title')}' by '{original_record.get('artist')}'")
                    logger.error(f"    Status: {status_code}")
                    logger.error(f"    Error Details: {json.dumps(error_details)}")
                    logger.error("---------------------------------------------")

            if not all_successful:
                logger.warning(f"Finished batch creation for '{collection}' with one or more failures.")
            return all_successful
        except Exception as e:
            logger.error(f"An exception occurred during the batch create API call for '{collection}': {e}")
            return False

