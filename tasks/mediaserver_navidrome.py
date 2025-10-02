# tasks/mediaserver_navidrome.py

import requests
import logging
import os
import random
import config

logger = logging.getLogger(__name__)

REQUESTS_TIMEOUT = 300
NAVIDROME_API_BATCH_SIZE = 40

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

def download_track(temp_dir, item):
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

def get_recent_albums(limit):
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

def get_all_songs():
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

def _add_to_playlist(playlist_id, item_ids, user_creds=None):
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

def _create_playlist_batched(playlist_name, item_ids, user_creds=None):
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
        if not _add_to_playlist(new_playlist_id, ids_to_add_later, user_creds):
            logger.error(f"Failed to add all songs to the new playlist '{playlist_name}'. The playlist was created but may be incomplete.")
            # We still return the playlist object, as it was created.
    
    # Standardize the keys to match what the rest of the app expects ('Id' with capital I)
    new_playlist['Id'] = new_playlist.get('id')
    new_playlist['Name'] = new_playlist.get('name')
    
    return new_playlist


def create_playlist(base_name, item_ids):
    """Creates a new playlist on Navidrome using admin credentials, with batching."""
    _create_playlist_batched(base_name, item_ids, user_creds=None)


def get_all_playlists():
    """Fetches all playlists from Navidrome using admin credentials."""
    response = _navidrome_request("getPlaylists")
    if response and "playlists" in response and "playlist" in response["playlists"]:
        return [{**p, 'Id': p.get('id'), 'Name': p.get('name')} for p in response["playlists"]["playlist"]]
    return []

def delete_playlist(playlist_id):
    """Deletes a playlist on Navidrome using admin credentials."""
    response = _navidrome_request("deletePlaylist", {"id": playlist_id}, method='post')
    if response and response.get("status") == "ok":
        logger.info(f"🗑️ Deleted Navidrome playlist ID: {playlist_id}")
        return True
    logger.error(f"Failed to delete playlist ID '{playlist_id}' on Navidrome")
    return False

# --- USER-SPECIFIC NAVIDROME FUNCTIONS ---
def get_tracks_from_album(album_id, user_creds=None):
    """Fetches all audio tracks for an album. Uses specific user_creds if provided."""
    params = {"id": album_id}
    response = _navidrome_request("getAlbum", params, user_creds=user_creds)
    if response and "album" in response and "song" in response["album"]:
        songs = response["album"]["song"]
        return [{**s, 'Id': s.get('id'), 'Name': s.get('title'), 'AlbumArtist': s.get('artist'), 'Path': s.get('path')} for s in songs]
    return []

def get_playlist_by_name(playlist_name, user_creds=None):
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

def get_top_played_songs(limit, user_creds):
    """Fetches the top N most played songs from Navidrome for a specific user."""
    all_top_songs = []
    num_albums_to_fetch = (limit // 10) + 10
    params = {"type": "frequent", "size": num_albums_to_fetch}
    response = _navidrome_request("getAlbumList2", params, user_creds=user_creds)
    if response and "albumList2" in response and "album" in response["albumList2"]:
        for album in response["albumList2"]["album"]:
            tracks = get_tracks_from_album(album.get("id"), user_creds=user_creds)
            if tracks: all_top_songs.extend(tracks)
    return random.sample(all_top_songs, limit) if len(all_top_songs) > limit else all_top_songs

def get_last_played_time(item_id, user_creds):
    """Fetches the last played time for a track for a specific user."""
    response = _navidrome_request("getSong", {"id": item_id}, user_creds=user_creds)
    if response and "song" in response: return response["song"].get("lastPlayed")
    return None

def create_instant_playlist(playlist_name, item_ids, user_creds):
    """Creates a new instant playlist on Navidrome for a specific user, with batching."""
    final_playlist_name = f"{playlist_name.strip()}_instant"
    return _create_playlist_batched(final_playlist_name, item_ids, user_creds)
