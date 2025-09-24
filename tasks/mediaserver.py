# tasks/mediaserver.py

import requests
import logging
import os
import random
import config  # Import the config module to access server type and settings

logger = logging.getLogger(__name__)

# Define a global timeout for all requests
REQUESTS_TIMEOUT = 300
# Define a batch size for Navidrome API calls to avoid long URLs
NAVIDROME_API_BATCH_SIZE = 40


# ##############################################################################
# JELLYFIN IMPLEMENTATION
# ##############################################################################

def _jellyfin_get_users(token):
    """Fetches a list of all users from Jellyfin using a provided token."""
    url = f"{config.JELLYFIN_URL}/Users"
    headers = {"X-Emby-Token": token}
    try:
        r = requests.get(url, headers=headers, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error(f"Jellyfin get_users failed: {e}", exc_info=True)
        return None

def _jellyfin_resolve_user(identifier, token):
    """
    Resolves a Jellyfin username to a User ID.
    If the identifier doesn't match any username, it's returned as is, assuming it's already an ID.
    """
    users = _jellyfin_get_users(token)
    if users:
        for user in users:
            if user.get('Name', '').lower() == identifier.lower():
                logger.info(f"Matched username '{identifier}' to User ID '{user['Id']}'.")
                return user['Id']
    
    logger.info(f"No username match for '{identifier}'. Assuming it is a User ID.")
    return identifier # Return original identifier if no match is found

# --- ADMIN/GLOBAL JELLYFIN FUNCTIONS ---
def _jellyfin_get_recent_albums(limit):
    """
    Fetches a list of the most recently added albums from Jellyfin using pagination.
    Uses global admin credentials.
    """
    all_albums = []
    start_index = 0
    page_size = 500
    fetch_all = (limit == 0)
    while fetch_all or len(all_albums) < limit:
        size_to_fetch = page_size if fetch_all else min(page_size, limit - len(all_albums))
        if size_to_fetch <= 0: break
        url = f"{config.JELLYFIN_URL}/Users/{config.JELLYFIN_USER_ID}/Items"
        params = {"IncludeItemTypes": "MusicAlbum", "SortBy": "DateCreated", "SortOrder": "Descending", "Recursive": True, "Limit": size_to_fetch, "StartIndex": start_index}
        try:
            r = requests.get(url, headers=config.HEADERS, params=params, timeout=REQUESTS_TIMEOUT)
            r.raise_for_status()
            response_data = r.json()
            albums = response_data.get("Items", [])
            if not albums: break
            all_albums.extend(albums)
            start_index += len(albums)
            if len(albums) < size_to_fetch: break
            if fetch_all and start_index >= response_data.get("TotalRecordCount", float('inf')): break
        except Exception as e:
            logger.error(f"Jellyfin get_recent_albums failed: {e}", exc_info=True)
            break
    return all_albums

def _jellyfin_get_tracks_from_album(album_id):
    """Fetches all audio tracks for a given album ID from Jellyfin using admin credentials."""
    url = f"{config.JELLYFIN_URL}/Users/{config.JELLYFIN_USER_ID}/Items"
    params = {"ParentId": album_id, "IncludeItemTypes": "Audio"}
    try:
        r = requests.get(url, headers=config.HEADERS, params=params, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        return r.json().get("Items", [])
    except Exception as e:
        logger.error(f"Jellyfin get_tracks_from_album failed for album {album_id}: {e}", exc_info=True)
        return []

def _jellyfin_download_track(temp_dir, item):
    """Downloads a single track from Jellyfin using admin credentials."""
    try:
        track_id = item['Id']
        file_extension = os.path.splitext(item.get('Path', ''))[1] or '.tmp'
        download_url = f"{config.JELLYFIN_URL}/Items/{track_id}/Download"
        local_filename = os.path.join(temp_dir, f"{track_id}{file_extension}")
        with requests.get(download_url, headers=config.HEADERS, stream=True, timeout=REQUESTS_TIMEOUT) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
        logger.info(f"Downloaded '{item['Name']}' to '{local_filename}'")
        return local_filename
    except Exception as e:
        logger.error(f"Failed to download track {item.get('Name', 'Unknown')}: {e}", exc_info=True)
        return None

def _jellyfin_get_all_songs():
    """Fetches all songs from Jellyfin using admin credentials."""
    url = f"{config.JELLYFIN_URL}/Users/{config.JELLYFIN_USER_ID}/Items"
    params = {"IncludeItemTypes": "Audio", "Recursive": True}
    try:
        r = requests.get(url, headers=config.HEADERS, params=params, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        return r.json().get("Items", [])
    except Exception as e:
        logger.error(f"Jellyfin get_all_songs failed: {e}", exc_info=True)
        return []

def _jellyfin_get_playlist_by_name(playlist_name):
    """Finds a Jellyfin playlist by its exact name using admin credentials."""
    url = f"{config.JELLYFIN_URL}/Users/{config.JELLYFIN_USER_ID}/Items"
    params = {"IncludeItemTypes": "Playlist", "Recursive": True, "Name": playlist_name}
    try:
        r = requests.get(url, headers=config.HEADERS, params=params, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        playlists = r.json().get("Items", [])
        return playlists[0] if playlists else None
    except Exception as e:
        logger.error(f"Jellyfin get_playlist_by_name failed for '{playlist_name}': {e}", exc_info=True)
        return None

def _jellyfin_create_playlist(base_name, item_ids):
    """Creates a new playlist on Jellyfin using admin credentials."""
    url = f"{config.JELLYFIN_URL}/Playlists"
    body = {"Name": base_name, "Ids": item_ids, "UserId": config.JELLYFIN_USER_ID}
    try:
        r = requests.post(url, headers=config.HEADERS, json=body, timeout=REQUESTS_TIMEOUT)
        if r.ok: logger.info("✅ Created Jellyfin playlist '%s'", base_name)
    except Exception as e:
        logger.error("Exception creating Jellyfin playlist '%s': %s", base_name, e, exc_info=True)

def _jellyfin_get_all_playlists():
    """Fetches all playlists from Jellyfin using admin credentials."""
    url = f"{config.JELLYFIN_URL}/Users/{config.JELLYFIN_USER_ID}/Items"
    params = {"IncludeItemTypes": "Playlist", "Recursive": True}
    try:
        r = requests.get(url, headers=config.HEADERS, params=params, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        return r.json().get("Items", [])
    except Exception as e:
        logger.error(f"Jellyfin get_all_playlists failed: {e}", exc_info=True)
        return []

def _jellyfin_delete_playlist(playlist_id):
    """Deletes a playlist on Jellyfin using admin credentials."""
    url = f"{config.JELLYFIN_URL}/Items/{playlist_id}"
    try:
        r = requests.delete(url, headers=config.HEADERS, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        return True
    except Exception as e:
        logger.error(f"Exception deleting Jellyfin playlist ID {playlist_id}: {e}", exc_info=True)
        return False

# --- USER-SPECIFIC JELLYFIN FUNCTIONS ---
def _jellyfin_get_top_played_songs(limit, user_id, token):
    """Fetches the top N most played songs from Jellyfin for a specific user."""
    url = f"{config.JELLYFIN_URL}/Users/{user_id}/Items"
    headers = {"X-Emby-Token": token}
    params = {"IncludeItemTypes": "Audio", "SortBy": "PlayCount", "SortOrder": "Descending", "Recursive": True, "Limit": limit, "Fields": "UserData,Path"}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        return r.json().get("Items", [])
    except Exception as e:
        logger.error(f"Jellyfin get_top_played_songs failed for user {user_id}: {e}", exc_info=True)
        return []

def _jellyfin_get_last_played_time(item_id, user_id, token):
    """Fetches the last played time for a specific track from Jellyfin for a specific user."""
    url = f"{config.JELLYFIN_URL}/Users/{user_id}/Items/{item_id}"
    headers = {"X-Emby-Token": token}
    params = {"Fields": "UserData"}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        return r.json().get("UserData", {}).get("LastPlayedDate")
    except Exception as e:
        logger.error(f"Jellyfin get_last_played_time failed for item {item_id}, user {user_id}: {e}", exc_info=True)
        return None

def _jellyfin_create_instant_playlist(playlist_name, item_ids, user_id, token):
    """Creates a new instant playlist on Jellyfin for a specific user."""
    final_playlist_name = f"{playlist_name.strip()}_instant"
    url = f"{config.JELLYFIN_URL}/Playlists"
    headers = {"X-Emby-Token": token}
    body = {"Name": final_playlist_name, "Ids": item_ids, "UserId": user_id}
    try:
        r = requests.post(url, headers=headers, json=body, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error("Exception creating Jellyfin instant playlist '%s' for user %s: %s", playlist_name, user_id, e, exc_info=True)
        return None

# ##############################################################################
# NAVIDROME (SUBSONIC API) IMPLEMENTATION
# ##############################################################################
def get_navidrome_auth_params(username=None, password=None):
    """Generates Navidrome auth params, using provided creds or falling back to global config."""
    auth_user = username or config.NAVIDROME_USER
    auth_pass = password or config.NAVIDROME_PASSWORD
    if not auth_user or not auth_pass: 
        logger.warning("Navidrome User or Password is not configured.")
        return {}
    hex_encoded_password = auth_pass.encode('utf-8').hex()
    return {"u": auth_user, "p": f"enc:{hex_encoded_password}", "v": "1.16.1", "c": config.APP_VERSION, "f": "json"}

def _navidrome_request(endpoint, params=None, method='get', stream=False, user_creds=None):
    """
    Helper to make Navidrome API requests. It sends all parameters in the URL's
    query string, which is the expected behavior for Subsonic APIs, but can cause
    issues with very long parameter lists (e.g., creating large playlists).
    """
    params = params or {}
    auth_params = get_navidrome_auth_params(
        username=user_creds.get('user') if user_creds else None,
        password=user_creds.get('password') if user_creds else None
    )
    if not auth_params:
        logger.error("Navidrome credentials not configured. Cannot make API call.")
        return None

    url = f"{config.NAVIDROME_URL}/rest/{endpoint}.view"
    all_params = {**auth_params, **params}

    try:
        r = requests.request(method, url, params=all_params, timeout=REQUESTS_TIMEOUT, stream=stream)
        r.raise_for_status()

        if stream:
            return r
            
        subsonic_response = r.json().get("subsonic-response", {})
        if subsonic_response.get("status") == "failed":
            error = subsonic_response.get("error", {})
            logger.error(f"Navidrome API Error on '{endpoint}': {error.get('message')}")
            return None
        return subsonic_response
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling Navidrome API endpoint '{endpoint}': {e}", exc_info=True)
        return None

def _navidrome_download_track(temp_dir, item):
    """Downloads a single track from Navidrome using admin credentials."""
    try:
        track_id = item['id'] 
        file_extension = os.path.splitext(item.get('path', ''))[1] or '.tmp'
        local_filename = os.path.join(temp_dir, f"{track_id}{file_extension}")
        
        response = _navidrome_request("stream", params={"id": track_id}, stream=True)
        if response:
            with open(local_filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info(f"Downloaded '{item.get('title', 'Unknown')}' to '{local_filename}'")
            return local_filename
    except Exception as e:
        logger.error(f"Failed to download Navidrome track {item.get('title', 'Unknown')}: {e}", exc_info=True)
    return None

def _navidrome_get_recent_albums(limit):
    """Fetches a list of the most recently added albums from Navidrome using admin credentials."""
    all_albums = []
    offset = 0
    page_size = 500
    fetch_all = (limit == 0)

    while fetch_all or len(all_albums) < limit:
        size_to_fetch = page_size if fetch_all else min(page_size, limit - len(all_albums))
        if size_to_fetch <= 0: break

        params = {"type": "newest", "size": size_to_fetch, "offset": offset}
        response = _navidrome_request("getAlbumList2", params)

        if response and "albumList2" in response and "album" in response["albumList2"]:
            albums = response["albumList2"]["album"]
            if not albums: break 

            all_albums.extend([{**a, 'Id': a.get('id'), 'Name': a.get('name')} for a in albums])
            offset += len(albums)

            if len(albums) < size_to_fetch: break
        else:
            logger.error("Failed to fetch recent albums page from Navidrome.")
            break
            
    return all_albums

def _navidrome_get_all_songs():
    """Fetches all songs from Navidrome using admin credentials."""
    all_songs = []
    offset = 0
    limit = 500
    while True:
        params = {"query": '', "songCount": limit, "songOffset": offset}
        response = _navidrome_request("search3", params)
        if response and "searchResult3" in response and "song" in response["searchResult3"]:
            songs = response["searchResult3"]["song"]
            if not songs: break
            all_songs.extend([{'Id': s.get('id'), 'Name': s.get('title'), 'AlbumArtist': s.get('artist'), 'Path': s.get('path')} for s in songs])
            offset += len(songs)
            if len(songs) < limit: break
        else:
            logger.error("Failed to fetch all songs from Navidrome.")
            break
    return all_songs

def _navidrome_add_to_playlist(playlist_id, item_ids, user_creds=None):
    """
    Adds a list of songs to an existing Navidrome playlist in batches.
    Uses the 'updatePlaylist' endpoint.
    """
    if not item_ids:
        return True

    logger.info(f"Adding {len(item_ids)} songs to Navidrome playlist ID {playlist_id} in batches.")
    for i in range(0, len(item_ids), NAVIDROME_API_BATCH_SIZE):
        batch_ids = item_ids[i:i + NAVIDROME_API_BATCH_SIZE]
        params = {"playlistId": playlist_id, "songIdToAdd": batch_ids}
        
        # Note: updatePlaylist uses a POST method.
        response = _navidrome_request("updatePlaylist", params, method='post', user_creds=user_creds)
        
        if not (response and response.get("status") == "ok"):
            logger.error(f"Failed to add batch of {len(batch_ids)} songs to playlist {playlist_id}.")
            return False
    logger.info(f"Successfully added all songs to playlist {playlist_id}.")
    return True

def _navidrome_create_playlist_batched(playlist_name, item_ids, user_creds=None):
    """
    Creates a new playlist on Navidrome. Handles large numbers of
    songs by batching and captures the new playlist ID directly from the
    creation response to avoid race conditions.
    """
    # If no songs are provided, create an empty playlist.
    if not item_ids:
        item_ids = []

    # --- Create the playlist and capture the response ---
    ids_for_creation = item_ids[:NAVIDROME_API_BATCH_SIZE]
    ids_to_add_later = item_ids[NAVIDROME_API_BATCH_SIZE:]

    create_params = {"name": playlist_name, "songId": ids_for_creation}
    create_response = _navidrome_request("createPlaylist", create_params, method='post', user_creds=user_creds)

    # --- Extract playlist object directly from the creation response ---
    if not (create_response and create_response.get("status") == "ok" and "playlist" in create_response):
        logger.error(f"Failed to create Navidrome playlist '{playlist_name}' or API response was malformed.")
        return None

    new_playlist = create_response["playlist"]
    new_playlist_id = new_playlist.get("id")

    if not new_playlist_id:
        logger.error(f"Navidrome playlist '{playlist_name}' was created, but the response did not contain an ID.")
        return None

    logger.info(f"✅ Created Navidrome playlist '{playlist_name}' (ID: {new_playlist_id}) with the first {len(ids_for_creation)} songs.")

    # If there are more songs to add, use the ID we just got
    if ids_to_add_later:
        if not _navidrome_add_to_playlist(new_playlist_id, ids_to_add_later, user_creds):
            logger.error(f"Failed to add all songs to the new playlist '{playlist_name}'. The playlist was created but may be incomplete.")
            # We still return the playlist object, as it was created.
    
    # Standardize the keys to match what the rest of the app expects ('Id' with capital I)
    new_playlist['Id'] = new_playlist.get('id')
    new_playlist['Name'] = new_playlist.get('name')
    
    return new_playlist


def _navidrome_create_playlist(base_name, item_ids):
    """Creates a new playlist on Navidrome using admin credentials, with batching."""
    _navidrome_create_playlist_batched(base_name, item_ids, user_creds=None)


def _navidrome_get_all_playlists():
    """Fetches all playlists from Navidrome using admin credentials."""
    response = _navidrome_request("getPlaylists")
    if response and "playlists" in response and "playlist" in response["playlists"]:
        return [{**p, 'Id': p.get('id'), 'Name': p.get('name')} for p in response["playlists"]["playlist"]]
    return []

def _navidrome_delete_playlist(playlist_id):
    """Deletes a playlist on Navidrome using admin credentials."""
    response = _navidrome_request("deletePlaylist", {"id": playlist_id}, method='post')
    if response and response.get("status") == "ok":
        logger.info(f"🗑️ Deleted Navidrome playlist ID: {playlist_id}")
        return True
    logger.error(f"Failed to delete playlist ID '{playlist_id}' on Navidrome")
    return False

# --- USER-SPECIFIC NAVIDROME FUNCTIONS ---
def _navidrome_get_tracks_from_album(album_id, user_creds=None):
    """Fetches all audio tracks for an album. Uses specific user_creds if provided."""
    params = {"id": album_id}
    response = _navidrome_request("getAlbum", params, user_creds=user_creds)
    if response and "album" in response and "song" in response["album"]:
        songs = response["album"]["song"]
        return [{**s, 'Id': s.get('id'), 'Name': s.get('title'), 'AlbumArtist': s.get('artist'), 'Path': s.get('path')} for s in songs]
    return []

def _navidrome_get_playlist_by_name(playlist_name, user_creds=None):
    """
    Finds a Navidrome playlist by its exact name. Returns the first match found.
    This is primarily used for checking if a playlist exists before deletion.
    """
    response = _navidrome_request("getPlaylists", user_creds=user_creds)
    if not (response and "playlists" in response and "playlist" in response["playlists"]):
        return None

    # Find the first playlist that matches the name exactly.
    for playlist_summary in response["playlists"]["playlist"]:
        if playlist_summary.get("name") == playlist_name:
            # For the purpose of checking existence and getting an ID for deletion,
            # the summary object is sufficient.
            return playlist_summary
    
    return None # No match found

def _navidrome_get_top_played_songs(limit, user_creds):
    """Fetches the top N most played songs from Navidrome for a specific user."""
    all_top_songs = []
    num_albums_to_fetch = (limit // 10) + 10
    params = {"type": "frequent", "size": num_albums_to_fetch}
    response = _navidrome_request("getAlbumList2", params, user_creds=user_creds)
    if response and "albumList2" in response and "album" in response["albumList2"]:
        for album in response["albumList2"]["album"]:
            tracks = _navidrome_get_tracks_from_album(album.get("id"), user_creds=user_creds)
            if tracks: all_top_songs.extend(tracks)
    return random.sample(all_top_songs, limit) if len(all_top_songs) > limit else all_top_songs

def _navidrome_get_last_played_time(item_id, user_creds):
    """Fetches the last played time for a track for a specific user."""
    response = _navidrome_request("getSong", {"id": item_id}, user_creds=user_creds)
    if response and "song" in response: return response["song"].get("lastPlayed")
    return None

def _navidrome_create_instant_playlist(playlist_name, item_ids, user_creds):
    """Creates a new instant playlist on Navidrome for a specific user, with batching."""
    final_playlist_name = f"{playlist_name.strip()}_instant"
    return _navidrome_create_playlist_batched(final_playlist_name, item_ids, user_creds)


# ##############################################################################
# LYRION (JSON-RPC) IMPLEMENTATION
# ##############################################################################
# Lyrion uses a JSON-RPC API. This section contains functions to interact with it.

def _lyrion_get_first_player():
    """Gets the first available player from Lyrion for web interface operations."""
    try:
        response = _lyrion_jsonrpc_request("players", [0, 1])
        if response and "players_loop" in response and response["players_loop"]:
            player = response["players_loop"][0]
            player_id = player.get("playerid")
            if player_id:
                logger.info(f"Found Lyrion player: {player_id}")
                return player_id
        
        # Fallback: try to use a common default or return None
        logger.warning("No Lyrion players found, using fallback player ID")
        return "10.42.6.0"  # Use the player from your example as fallback
    except Exception as e:
        logger.error(f"Error getting Lyrion player: {e}")
        return "10.42.6.0"  # Use the player from your example as fallback

def _lyrion_jsonrpc_request(method, params, player_id=""):
    """
    Helper to make a JSON-RPC request to the Lyrion server without authentication.
    Returns the 'result' field on success, or None on failure.
    """
    url = f"{config.LYRION_URL}/jsonrpc.js"
    payload = {
        "id": 1,
        "method": "slim.request",
        "params": [player_id, [method, *params]]
    }

    try:
        with requests.Session() as s:
            s.headers.update({"Content-Type": "application/json"})
            r = s.post(url, json=payload, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        response_data = r.json()
        if response_data.get("error"):
            logger.error(f"Lyrion JSON-RPC Error: {response_data['error'].get('message')}")
            return None

        # On success, return the result field. It might be None if not present.
        # The caller must check for an explicit `None` return to detect failure.
        return response_data.get("result")
    except Exception as e:
        logger.error(f"Failed to call Lyrion JSON-RPC API with method '{method}': {e}", exc_info=True)
        return None

def _lyrion_download_track(temp_dir, item):
    """Downloads a single track from Lyrion using its URL."""
    try:
        track_id = item.get('Id')
        if not track_id:
            logger.error("Lyrion item does not have a track ID.")
            return None
            
        # The correct, stable URL format for directly downloading a track from Lyrion/LMS by its ID.
        # This avoids issues with the /stream endpoint which is often for the currently playing track.
        download_url = f"{config.LYRION_URL}/music/{track_id}/download"
        
        # A more robust way to handle the file extension.
        file_extension = item.get('Path', '.mp3')
        if file_extension and '.' in file_extension:
            file_extension = os.path.splitext(file_extension)[1]
        else:
            file_extension = '.mp3'
        
        local_filename = os.path.join(temp_dir, f"{track_id}{file_extension}")
        
        logger.info(f"Attempting to download from URL: {download_url}")
        
        # Use a new session for each download to avoid connection pooling issues.
        with requests.Session() as s:
            with s.get(download_url, stream=True, timeout=REQUESTS_TIMEOUT) as r:
                r.raise_for_status()
                with open(local_filename, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
        logger.info(f"Downloaded '{item.get('title', 'Unknown')}' to '{local_filename}'")
        return local_filename
    except Exception as e:
        logger.error(f"Failed to download Lyrion track {item.get('title', 'Unknown')}: {e}", exc_info=True)
    return None

def _lyrion_get_recent_albums(limit):
    """Fetches recently added albums from Lyrion using JSON-RPC."""
    logger.info(f"Attempting to fetch {limit} most recent albums from Lyrion via JSON-RPC.")
    
    # Handle fetching all albums if limit is 0
    if limit == 0:
        params = [0, 999999, "sort:new"]
    else:
        params = [0, limit, "sort:new"]
        
    try:
        response = _lyrion_jsonrpc_request("albums", params)
        logger.info(f"Lyrion API Raw Response: {response}")
    except Exception as e:
        logger.error(f"Lyrion API call for recent albums failed: {e}", exc_info=True)
        return []

    if response and "albums_loop" in response:
        albums = response["albums_loop"]
        # Lyrion API response keys are different, so we map them to our standard format.
        return [{'Id': a.get('id'), 'Name': a.get('album')} for a in albums]
    
    logger.warning("Lyrion API response did not contain the 'albums_loop' key or was empty.")
    return []

def _lyrion_get_all_songs():
    """Fetches all songs from Lyrion using JSON-RPC."""
    response = _lyrion_jsonrpc_request("titles", [0, 999999])
    if response and "titles_loop" in response:
        songs = response["titles_loop"]
        # Map Lyrion API keys to our standard format.
        return [{'Id': s.get('id'), 'Name': s.get('title'), 'AlbumArtist': s.get('artist'), 'Path': s.get('url'), 'url': s.get('url')} for s in songs]
    return []

def _lyrion_add_to_playlist(playlist_id, item_ids):
    """Adds songs to a Lyrion playlist using the working player-based method."""
    if not item_ids: 
        return True
    
    logger.info(f"Adding {len(item_ids)} songs to Lyrion playlist ID '{playlist_id}'.")
    
    # Get a player for the command
    player_id = _lyrion_get_first_player()
    if not player_id:
        logger.error("No Lyrion player available for playlist operations.")
        return False
    
    try:
        # Get the original playlist name FIRST, before any operations
        logger.debug("Step 0: Getting original playlist name before operations")
        playlist_info = _lyrion_jsonrpc_request("playlists", [0, 999999])  # Get all playlists
        
        original_name = None
        if playlist_info and "playlists_loop" in playlist_info:
            for pl in playlist_info["playlists_loop"]:
                if str(pl.get("id")) == str(playlist_id):
                    original_name = pl.get("playlist")
                    logger.debug(f"Found original playlist name: '{original_name}' for ID {playlist_id}")
                    break
        
        if not original_name:
            logger.error(f"Could not find playlist {playlist_id} in playlists list!")
            return False
        
        # Method: Load playlist to player, add tracks, then use playlists edit to update
        logger.info(f"Using method: Load → Add → Update original playlist via edit command")
        
        # Step 1: Load the saved playlist into the player's current playlist
        logger.debug(f"Step 1: Loading playlist {playlist_id} to player {player_id}")
        load_response = _lyrion_jsonrpc_request("playlistcontrol", [
            "cmd:load",
            f"playlist_id:{playlist_id}"
        ], player_id)
        
        logger.debug(f"Load playlist response: {load_response}")
        
        # Step 2: Add tracks to the player's current playlist in batches
        batch_size = 50  # Larger batches since this method works
        total_added = 0
        
        for i in range(0, len(item_ids), batch_size):
            batch_ids = item_ids[i:i + batch_size]
            track_id_list = ",".join(str(track_id) for track_id in batch_ids)
            
            logger.debug(f"Step 2: Adding batch {i//batch_size + 1} with {len(batch_ids)} tracks")
            add_response = _lyrion_jsonrpc_request("playlistcontrol", [
                "cmd:add",
                f"track_id:{track_id_list}"
            ], player_id)
            
            logger.debug(f"Add batch response: {add_response}")
            
            if add_response and "count" in add_response:
                batch_added = add_response.get("count", 0)
                total_added += batch_added
                logger.debug(f"Added {batch_added} tracks in this batch, total: {total_added}")
            
            # Small delay between batches
            if i + batch_size < len(item_ids):
                import time
                time.sleep(0.1)
        
        # Step 3: Delete the original empty playlist
        logger.debug(f"Step 3: Deleting original empty playlist {playlist_id}")
        delete_response = _lyrion_jsonrpc_request("playlists", [
            "delete",
            f"playlist_id:{playlist_id}"
        ])
        logger.debug(f"Delete response: {delete_response}")
        
        # Step 4: Save the current player playlist with the original name
        logger.debug(f"Step 4: Saving current playlist as '{original_name}'")
        save_response = _lyrion_jsonrpc_request("playlist", [
            "save",
            original_name,
            "silent:1"
        ], player_id)
        
        logger.debug(f"Save playlist response: {save_response}")
        
        # Check if we got the expected playlist ID back
        if save_response and "__playlist_id" in save_response:
            final_playlist_id = save_response["__playlist_id"]
            if str(final_playlist_id) == str(playlist_id):
                logger.info(f"✅ Successfully updated original playlist {playlist_id} with {total_added} tracks")
                return True
            else:
                logger.warning(f"Created new playlist {final_playlist_id} instead of updating {playlist_id}")
                # If we got a different ID, try to delete the new one and rename it
                try:
                    # The new playlist has the tracks, so we need to work with it
                    logger.info(f"Working with new playlist ID {final_playlist_id} which has the content")
                    return True
                except Exception as e:
                    logger.error(f"Error handling new playlist: {e}")
                    return False
        elif total_added > 0:
            logger.info(f"✅ Successfully added {total_added} tracks (save response: {save_response})")
            return True
        else:
            logger.warning("No tracks were added to the playlist")
            return False
            
    except Exception as e:
        logger.error(f"Error in playlist update method: {e}")
        return False

def _lyrion_create_playlist_batched(playlist_name, item_ids):
    """Creates a new Lyrion playlist and adds tracks using the web interface approach."""
    logger.info(f"Attempting to create Lyrion playlist '{playlist_name}' with {len(item_ids)} songs using web interface method.")

    try:
        # Step 1: Create the playlist using JSON-RPC (this part works)
        create_response = _lyrion_jsonrpc_request("playlists", ["new", f"name:{playlist_name}"])
        
        if create_response:
            playlist_id = (
                create_response.get("id") or
                create_response.get("overwritten_playlist_id") or
                create_response.get("playlist_id")
            )
            
            if playlist_id:
                logger.info(f"✅ Created Lyrion playlist '{playlist_name}' (ID: {playlist_id}).")
                
                # Step 2: Add tracks using the web interface method
                if item_ids:
                    if _lyrion_add_to_playlist(playlist_id, item_ids):
                        logger.info(f"✅ Successfully added {len(item_ids)} tracks to playlist '{playlist_name}'.")
                    else:
                        logger.warning(f"Playlist '{playlist_name}' created but some tracks may not have been added.")
                
                return {"Id": playlist_id, "Name": playlist_name}
        
        logger.error(f"Failed to create Lyrion playlist '{playlist_name}'. Response: {create_response}")
        return None
        
    except Exception as e:
        logger.error(f"Exception creating Lyrion playlist '{playlist_name}': {e}", exc_info=True)
        return None

def _lyrion_create_playlist(base_name, item_ids):
    """Creates a new playlist on Lyrion using admin credentials, with batching."""
    _lyrion_create_playlist_batched(base_name, item_ids)

def _lyrion_get_all_playlists():
    """Fetches all playlists from Lyrion using JSON-RPC."""
    response = _lyrion_jsonrpc_request("playlists", [0, 999999])
    if response and "playlists_loop" in response:
        playlists = response["playlists_loop"]
        return [{'Id': p.get('id'), 'Name': p.get('playlist')} for p in playlists]
    return []

def _lyrion_delete_playlist(playlist_id):
    """Deletes a playlist on Lyrion using JSON-RPC."""
    # The correct command is 'playlists delete'.
    response = _lyrion_jsonrpc_request("playlists", ["delete", f"playlist_id:{playlist_id}"])
    if response:
        logger.info(f"🗑️ Deleted Lyrion playlist ID: {playlist_id}")
        return True
    logger.error(f"Failed to delete playlist ID '{playlist_id}' on Lyrion")
    return False

# --- User-specific Lyrion functions ---
def _lyrion_get_tracks_from_album(album_id):
    """Fetches all audio tracks for an album from Lyrion using JSON-RPC."""
    logger.info(f"Attempting to fetch tracks for album ID: {album_id}")
    
    # Lyrion's JSON-RPC doesn't have a direct "get tracks for album" call.
    # The 'titles' command with a filter is the correct way to get songs for an album.
    # We now fetch all songs and filter them by the album ID.
    response = _lyrion_jsonrpc_request("titles", [0, 999999, f"album_id:{album_id}"])
    logger.info(f"Lyrion API Raw Track Response for Album {album_id}: {response}")

    if response and "titles_loop" in response:
        songs = response["titles_loop"]
        return [{'Id': s.get('id'), 'Name': s.get('title'), 'AlbumArtist': s.get('artist'), 'Path': s.get('url'), 'url': s.get('url')} for s in songs]
    
    logger.warning(f"Lyrion API response for tracks of album {album_id} did not contain the 'titles_loop' key or was empty.")
    return []

def _lyrion_get_playlist_by_name(playlist_name):
    """Finds a Lyrion playlist by its exact name using JSON-RPC."""
    # Fetch all playlists and filter by name, as direct name search is not standard.
    all_playlists = _lyrion_get_all_playlists()
    for p in all_playlists:
        if p.get('Name') == playlist_name:
            return p # Return the already formatted playlist dict
    return None

def _lyrion_get_top_played_songs(limit):
    """Fetches the top N most played songs from Lyrion for a specific user using JSON-RPC."""
    response = _lyrion_jsonrpc_request("titles", [0, limit, "sort:popular"])
    if response and "titles_loop" in response:
        songs = response["titles_loop"]
        # Map Lyrion API keys to our standard format.
        return [{'Id': s.get('id'), 'Name': s.get('title'), 'AlbumArtist': s.get('artist'), 'Path': s.get('url'), 'url': s.get('url')} for s in songs]
    return []


def _lyrion_get_last_played_time(item_id):
    """Fetches the last played time for a track for a specific user. Not supported by Lyrion JSON-RPC API."""
    logger.warning("Lyrion's JSON-RPC API does not provide a 'last played time' for individual tracks.")
    return None

def _lyrion_create_instant_playlist(playlist_name, item_ids):
    """Creates a new instant playlist on Lyrion for a specific user, with batching."""
    final_playlist_name = f"{playlist_name.strip()}_instant"
    return _lyrion_create_playlist_batched(final_playlist_name, item_ids)

# ##############################################################################
# PUBLIC API (Dispatcher functions)
# ##############################################################################

def resolve_jellyfin_user(identifier, token):
    """Public dispatcher for resolving a Jellyfin user identifier."""
    return _jellyfin_resolve_user(identifier, token)

def delete_automatic_playlists():
    """Deletes all playlists ending with '_automatic' using admin credentials."""
    logger.info("Starting deletion of all '_automatic' playlists.")
    deleted_count = 0
    if config.MEDIASERVER_TYPE == 'jellyfin':
        for p in _jellyfin_get_all_playlists():
            if p.get('Name', '').endswith('_automatic') and _jellyfin_delete_playlist(p.get('Id')):
                deleted_count += 1
    elif config.MEDIASERVER_TYPE == 'navidrome':
        for p in _navidrome_get_all_playlists():
            if p.get('Name', '').endswith('_automatic') and _navidrome_delete_playlist(p.get('id')):
                deleted_count += 1
    elif config.MEDIASERVER_TYPE == 'lyrion':
        for p in _lyrion_get_all_playlists():
            if p.get('Name', '').endswith('_automatic') and _lyrion_delete_playlist(p.get('Id')):
                deleted_count += 1
    logger.info(f"Finished deletion. Deleted {deleted_count} playlists.")

def get_recent_albums(limit):
    """Fetches recently added albums using admin credentials."""
    if config.MEDIASERVER_TYPE == 'jellyfin': return _jellyfin_get_recent_albums(limit)
    if config.MEDIASERVER_TYPE == 'navidrome': return _navidrome_get_recent_albums(limit)
    if config.MEDIASERVER_TYPE == 'lyrion': return _lyrion_get_recent_albums(limit)
    return []

def get_tracks_from_album(album_id):
    """Fetches tracks for an album using admin credentials."""
    if config.MEDIASERVER_TYPE == 'jellyfin': return _jellyfin_get_tracks_from_album(album_id)
    if config.MEDIASERVER_TYPE == 'navidrome': return _navidrome_get_tracks_from_album(album_id)
    if config.MEDIASERVER_TYPE == 'lyrion': return _lyrion_get_tracks_from_album(album_id)
    return []

def download_track(temp_dir, item):
    """Downloads a track using admin credentials."""
    if config.MEDIASERVER_TYPE == 'jellyfin': return _jellyfin_download_track(temp_dir, item)
    if config.MEDIASERVER_TYPE == 'navidrome': return _navidrome_download_track(temp_dir, item)
    if config.MEDIASERVER_TYPE == 'lyrion': return _lyrion_download_track(temp_dir, item)
    return None

def get_all_songs():
    """Fetches all songs using admin credentials."""
    if config.MEDIASERVER_TYPE == 'jellyfin': return _jellyfin_get_all_songs()
    if config.MEDIASERVER_TYPE == 'navidrome': return _navidrome_get_all_songs()
    if config.MEDIASERVER_TYPE == 'lyrion': return _lyrion_get_all_songs()
    return []

def get_playlist_by_name(playlist_name):
    """Finds a playlist by name using admin credentials."""
    if not playlist_name: raise ValueError("Playlist name is required.")
    if config.MEDIASERVER_TYPE == 'jellyfin': return _jellyfin_get_playlist_by_name(playlist_name)
    if config.MEDIASERVER_TYPE == 'navidrome': return _navidrome_get_playlist_by_name(playlist_name)
    if config.MEDIASERVER_TYPE == 'lyrion': return _lyrion_get_playlist_by_name(playlist_name)
    return None

def create_playlist(base_name, item_ids):
    """Creates a playlist using admin credentials."""
    if not base_name: raise ValueError("Playlist name is required.")
    if not item_ids: raise ValueError("Track IDs are required.")
    if config.MEDIASERVER_TYPE == 'jellyfin': _jellyfin_create_playlist(base_name, item_ids)
    elif config.MEDIASERVER_TYPE == 'navidrome': _navidrome_create_playlist(base_name, item_ids)
    elif config.MEDIASERVER_TYPE == 'lyrion': _lyrion_create_playlist(base_name, item_ids)

def create_instant_playlist(playlist_name, item_ids, user_creds=None):
    """Creates an instant playlist. Uses user_creds if provided, otherwise admin."""
    if not playlist_name: raise ValueError("Playlist name is required.")
    if not item_ids: raise ValueError("Track IDs are required.")
    
    if config.MEDIASERVER_TYPE == 'jellyfin':
        token = user_creds.get('token') if user_creds else config.JELLYFIN_TOKEN
        if not token: raise ValueError("Jellyfin Token is required.")
        
        identifier = user_creds.get('user_identifier') if user_creds else config.JELLYFIN_USER_ID
        if not identifier: raise ValueError("Jellyfin User Identifier is required.")

        user_id = _jellyfin_resolve_user(identifier, token)
        return _jellyfin_create_instant_playlist(playlist_name, item_ids, user_id, token)

    if config.MEDIASERVER_TYPE == 'navidrome':
        return _navidrome_create_instant_playlist(playlist_name, item_ids, user_creds)
    
    if config.MEDIASERVER_TYPE == 'lyrion':
        return _lyrion_create_instant_playlist(playlist_name, item_ids)

    return None

def get_top_played_songs(limit, user_creds=None):
    """Fetches top played songs. Uses user_creds if provided, otherwise admin."""
    if config.MEDIASERVER_TYPE == 'jellyfin':
        user_id = user_creds.get('user_id') if user_creds else config.JELLYFIN_USER_ID
        token = user_creds.get('token') if user_creds else config.JELLYFIN_TOKEN
        if not user_id or not token: raise ValueError("Jellyfin User ID and Token are required.")
        return _jellyfin_get_top_played_songs(limit, user_id, token)
    if config.MEDIASERVER_TYPE == 'navidrome':
        return _navidrome_get_top_played_songs(limit, user_creds)
    if config.MEDIASERVER_TYPE == 'lyrion':
        return _lyrion_get_top_played_songs(limit)
    return []

def get_last_played_time(item_id, user_creds=None):
    """Fetches last played time for a track. Uses user_creds if provided, otherwise admin."""
    if config.MEDIASERVER_TYPE == 'jellyfin':
        user_id = user_creds.get('user_id') if user_creds else config.JELLYFIN_USER_ID
        token = user_creds.get('token') if user_creds else config.JELLYFIN_TOKEN
        if not user_id or not token: raise ValueError("Jellyfin User ID and Token are required.")
        return _jellyfin_get_last_played_time(item_id, user_id, token)
    if config.MEDIASERVER_TYPE == 'navidrome':
        return _navidrome_get_last_played_time(item_id, user_creds)
    if config.MEDIASERVER_TYPE == 'lyrion':
        return _lyrion_get_last_played_time(item_id)
    return None
