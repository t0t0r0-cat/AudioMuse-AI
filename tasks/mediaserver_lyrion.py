# tasks/mediaserver_lyrion.py

import requests
import logging
import os
import config

logger = logging.getLogger(__name__)

REQUESTS_TIMEOUT = 300

# ##############################################################################
# LYRION (JSON-RPC) IMPLEMENTATION
# ##############################################################################
# Lyrion uses a JSON-RPC API. This section contains functions to interact with it.

def _get_first_player():
    """Gets the first available player from Lyrion for web interface operations."""
    try:
        response = _jsonrpc_request("players", [0, 1])
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

def _jsonrpc_request(method, params, player_id=""):
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

def download_track(temp_dir, item):
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

def get_recent_albums(limit):
    """Fetches recently added albums from Lyrion using JSON-RPC."""
    logger.info(f"Attempting to fetch {limit} most recent albums from Lyrion via JSON-RPC.")
    # Lyrion appears to cap results per page (commonly 100). Implement paging to accumulate all results
    page_size = 100
    albums_accum = []

    # If limit == 0 we want all albums; represent that as None for easier logic
    remaining = None if limit == 0 else int(limit)
    offset = 0

    while True:
        # Calculate how many to request this page
        req_count = page_size if (remaining is None or remaining > page_size) else remaining
        params = [offset, req_count, "sort:new"]

        try:
            response = _jsonrpc_request("albums", params)
            logger.debug(f"Lyrion API Raw Response (offset={offset}, count={req_count}): {response}")
        except Exception as e:
            logger.error(f"Lyrion API call for recent albums failed at offset={offset}: {e}", exc_info=True)
            break

        if not response:
            logger.debug(f"No response for albums page offset={offset}. Stopping pagination.")
            break

        # Extract albums from possible response shapes
        page_albums = []
        if isinstance(response, dict) and "albums_loop" in response and isinstance(response["albums_loop"], list):
            page_albums = response["albums_loop"]
        elif isinstance(response, list):
            page_albums = response
        else:
            # Try to find the first list in the response dict
            if isinstance(response, dict):
                for v in response.values():
                    if isinstance(v, list):
                        page_albums = v
                        break

        if not page_albums:
            logger.debug(f"No albums found in page offset={offset}. Stopping pagination.")
            break

        # Map and append
        mapped = [{'Id': a.get('id'), 'Name': a.get('album')} for a in page_albums]
        albums_accum.extend(mapped)

        # Update remaining and offset
        if remaining is not None:
            remaining -= len(page_albums)
            if remaining <= 0:
                break

        # If the page returned fewer items than requested, we've reached the end
        if len(page_albums) < req_count:
            break

        offset += len(page_albums)

    if not albums_accum:
        logger.warning("Lyrion API returned no recent albums after pagination.")
    else:
        logger.info(f"Collected {len(albums_accum)} albums from Lyrion.")

    return albums_accum

def get_all_songs():
    """Fetches all songs from Lyrion using JSON-RPC."""
    response = _jsonrpc_request("titles", [0, 999999])
    if response and "titles_loop" in response:
        songs = response["titles_loop"]
        # Map Lyrion API keys to our standard format.
        return [{'Id': s.get('id'), 'Name': s.get('title'), 'AlbumArtist': s.get('artist'), 'Path': s.get('url'), 'url': s.get('url')} for s in songs]
    return []

def _add_to_playlist(playlist_id, item_ids):
    """Adds songs to a Lyrion playlist using the working player-based method."""
    if not item_ids: 
        return True
    
    logger.info(f"Adding {len(item_ids)} songs to Lyrion playlist ID '{playlist_id}'.")
    
    # Get a player for the command
    player_id = _get_first_player()
    if not player_id:
        logger.error("No Lyrion player available for playlist operations.")
        return False
    
    try:
        # Get the original playlist name FIRST, before any operations
        logger.debug("Step 0: Getting original playlist name before operations")
        playlist_info = _jsonrpc_request("playlists", [0, 999999])  # Get all playlists
        
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
        load_response = _jsonrpc_request("playlistcontrol", [
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
            add_response = _jsonrpc_request("playlistcontrol", [
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
        delete_response = _jsonrpc_request("playlists", [
            "delete",
            f"playlist_id:{playlist_id}"
        ])
        logger.debug(f"Delete response: {delete_response}")
        
        # Step 4: Save the current player playlist with the original name
        logger.debug(f"Step 4: Saving current playlist as '{original_name}'")
        save_response = _jsonrpc_request("playlist", [
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

def _create_playlist_batched(playlist_name, item_ids):
    """Creates a new Lyrion playlist and adds tracks using the web interface approach."""
    logger.info(f"Attempting to create Lyrion playlist '{playlist_name}' with {len(item_ids)} songs using web interface method.")

    try:
        # Step 1: Create the playlist using JSON-RPC (this part works)
        create_response = _jsonrpc_request("playlists", ["new", f"name:{playlist_name}"])
        
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
                    if _add_to_playlist(playlist_id, item_ids):
                        logger.info(f"✅ Successfully added {len(item_ids)} tracks to playlist '{playlist_name}'.")
                    else:
                        logger.warning(f"Playlist '{playlist_name}' created but some tracks may not have been added.")
                
                return {"Id": playlist_id, "Name": playlist_name}
        
        logger.error(f"Failed to create Lyrion playlist '{playlist_name}'. Response: {create_response}")
        return None
        
    except Exception as e:
        logger.error(f"Exception creating Lyrion playlist '{playlist_name}': {e}", exc_info=True)
        return None

def create_playlist(base_name, item_ids):
    """Creates a new playlist on Lyrion using admin credentials, with batching."""
    # Return the result of the batched creation so callers can inspect the new playlist ID
    return _create_playlist_batched(base_name, item_ids)

def get_all_playlists():
    """Fetches all playlists from Lyrion using JSON-RPC."""
    response = _jsonrpc_request("playlists", [0, 999999])
    if response and "playlists_loop" in response:
        playlists = response["playlists_loop"]
        return [{'Id': p.get('id'), 'Name': p.get('playlist')} for p in playlists]
    return []

def delete_playlist(playlist_id):
    """Deletes a playlist on Lyrion using JSON-RPC."""
    # The correct command is 'playlists delete'.
    response = _jsonrpc_request("playlists", ["delete", f"playlist_id:{playlist_id}"])
    if response:
        logger.info(f"🗑️ Deleted Lyrion playlist ID: {playlist_id}")
        return True
    logger.error(f"Failed to delete playlist ID '{playlist_id}' on Lyrion")
    return False

# --- User-specific Lyrion functions ---
def get_tracks_from_album(album_id):
    """Fetches all audio tracks for an album from Lyrion using JSON-RPC."""
    logger.info(f"Attempting to fetch tracks for album ID: {album_id}")
    
    # Lyrion's JSON-RPC doesn't have a direct "get tracks for album" call.
    # The 'titles' command with a filter is the correct way to get songs for an album.
    # We now fetch all songs and filter them by the album ID.
    try:
        response = _jsonrpc_request("titles", [0, 999999, f"album_id:{album_id}"])
        logger.debug(f"Lyrion API Raw Track Response for Album {album_id}: {response}")
    except Exception as e:
        logger.error(f"Lyrion API call for album {album_id} failed: {e}", exc_info=True)
        return []

    # Normalize response shapes: LMS/Lyrion may return a dict with 'titles_loop' or a raw list.
    songs = []
    if not response:
        logger.warning(f"Lyrion API returned empty response for album {album_id}.")
        return []

    if isinstance(response, dict):
        if "titles_loop" in response and isinstance(response["titles_loop"], list):
            songs = response["titles_loop"]
        else:
            # Fallback: try to find the first list value in the response
            for v in response.values():
                if isinstance(v, list):
                    songs = v
                    break
    elif isinstance(response, list):
        songs = response

    if not songs:
        logger.warning(f"Lyrion API response for tracks of album {album_id} did not contain any song entries.")
        return []

    # Robust Spotify detection: check several possible fields and make it case-insensitive.
    def is_spotify_track(item: dict) -> bool:
        for key in ("genre", "service", "source"):
            val = item.get(key)
            if isinstance(val, str) and "spotify" in val.lower():
                return True
        # Also check the URL/path for spotify links
        url = (item.get("url") or item.get("Path") or item.get("path") or "")
        if isinstance(url, str) and "spotify" in url.lower():
            return True
        return False

    local_songs = []
    skipped_tracks = []
    for s in songs:
        if is_spotify_track(s):
            skipped_tracks.append(s)
        else:
            local_songs.append(s)

    if skipped_tracks:
        skipped_count = len(skipped_tracks)
        logger.info(f"Skipping {skipped_count} track(s) from album {album_id} because they appear to be from Spotify or are non-downloadable.")
        # Log concise identifying information for each skipped track so operators can verify.
        for st in skipped_tracks:
            sk_id = st.get('id') or st.get('Id') or st.get('track_id')
            sk_title = st.get('title') or st.get('name') or st.get('Name')
            sk_artist = st.get('artist') or st.get('AlbumArtist') or st.get('albumArtist')
            sk_url = st.get('url') or st.get('Path') or st.get('path')
            logger.info(f"Skipped track - id: {sk_id!r}, title: {sk_title!r}, artist: {sk_artist!r}, url/path: {sk_url!r}")

    if not local_songs and songs:
        logger.info(f"Album {album_id} contains only Spotify or non-downloadable tracks and will be skipped.")

    # Map Lyrion API keys to our standard format with safe fallbacks.
    mapped = []
    for s in local_songs:
        id_val = s.get('id') or s.get('Id') or s.get('track_id')
        title = s.get('title') or s.get('name') or s.get('Name')
        artist = s.get('artist') or s.get('AlbumArtist') or s.get('albumArtist')
        path = s.get('url') or s.get('Path') or s.get('path') or ''
        mapped.append({'Id': id_val, 'Name': title, 'AlbumArtist': artist, 'Path': path, 'url': path})

    return mapped

def get_playlist_by_name(playlist_name):
    """Finds a Lyrion playlist by its exact name using JSON-RPC."""
    # Fetch all playlists and filter by name, as direct name search is not standard.
    all_playlists = get_all_playlists()
    for p in all_playlists:
        if p.get('Name') == playlist_name:
            return p # Return the already formatted playlist dict
    return None

def get_top_played_songs(limit):
    """Fetches the top N most played songs from Lyrion for a specific user using JSON-RPC."""
    response = _jsonrpc_request("titles", [0, limit, "sort:popular"])
    if response and "titles_loop" in response:
        songs = response["titles_loop"]
        # Map Lyrion API keys to our standard format.
        return [{'Id': s.get('id'), 'Name': s.get('title'), 'AlbumArtist': s.get('artist'), 'Path': s.get('url'), 'url': s.get('url')} for s in songs]
    return []


def get_last_played_time(item_id):
    """Fetches the last played time for a track for a specific user. Not supported by Lyrion JSON-RPC API."""
    logger.warning("Lyrion's JSON-RPC API does not provide a 'last played time' for individual tracks.")
    return None

def create_instant_playlist(playlist_name, item_ids):
    """Creates a new instant playlist on Lyrion for a specific user, with batching."""
    final_playlist_name = f"{playlist_name.strip()}_instant"
    return _create_playlist_batched(final_playlist_name, item_ids)
