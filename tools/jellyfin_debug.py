"""
Simple diagnostic script to exercise Jellyfin API helper functions in this repo.
Run from the repo root with your environment set (JELLYFIN_URL, JELLYFIN_TOKEN, JELLYFIN_USER_ID).

Examples (PowerShell):
    # Use your environment or export them inline
    $env:JELLYFIN_URL = 'http://your_jellyfin:8096'; $env:JELLYFIN_TOKEN = 'xxxx'; $env:JELLYFIN_USER_ID = 'xxxxxxxx'
    python .\tools\jellyfin_debug.py

The script prints counts and sample item fields to help pinpoint why media may not be found.
"""

import logging
import json
import traceback

import config
from tasks import mediaserver_jellyfin as mj

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
logger = logging.getLogger('jellyfin_debug')

def print_conf():
    logger.info("Using JELLYFIN_URL=%s, JELLYFIN_USER_ID=%s (token hidden), JELLYFIN_RECURSIVE=%s", config.JELLYFIN_URL, config.JELLYFIN_USER_ID, config.JELLYFIN_RECURSIVE)
    # Abort early if URL looks like placeholder to avoid accidental requests
    lower_url = (config.JELLYFIN_URL or '').lower()
    if 'your_jellyfin' in lower_url or 'your_jellyfin_url' in lower_url or 'http://your_jellyfin' in lower_url:
        logger.error("JELLYFIN_URL looks like a placeholder/default (%s). Aborting. Set environment variable JELLYFIN_URL to your real server and re-run.", config.JELLYFIN_URL)
        raise SystemExit(2)

def safe_print_items(items, limit=5):
    if not items:
        print("  (no items)")
        return
    print(f"  Total items returned: {len(items)} (showing up to {limit})")
    for i, it in enumerate(items[:limit], 1):
        try:
            # Print a compact representation of the item keys we commonly expect
            keys = ['Id', 'Name', 'AlbumArtist', 'Album', 'ParentId', 'IndexNumber', 'Path', 'Type', 'ServerId', 'Container', 'ProviderIds']
            display = {k: it.get(k) for k in keys if k in it}
            # MediaSources may be nested and large; show whether present
            if 'MediaSources' in it:
                display['MediaSources_count'] = len(it.get('MediaSources') or [])
            print(f"  {i}. {json.dumps(display, default=str)}")
        except Exception:
            print(f"  {i}. (failed to print item: {traceback.format_exc()})")


def run():
    print_conf()

    # 1) Verify we can list users
    print('\n1) Listing users via _jellyfin_get_users()')
    try:
        users = mj._jellyfin_get_users(config.JELLYFIN_TOKEN)
        if users is None:
            print('  -> _jellyfin_get_users returned None (check token / connectivity)')
        else:
            print(f'  -> Found {len(users)} users. Sample:')
            safe_print_items(users, limit=5)
    except Exception as e:
        print('  -> Exception calling _jellyfin_get_users()')
        traceback.print_exc()

    # 2) Recent albums
    print('\n2) get_recent_albums(limit=10)')
    try:
        albums = mj.get_recent_albums(10)
        safe_print_items(albums, limit=10)
    except Exception:
        print('  -> Exception calling get_recent_albums()')
        traceback.print_exc()

    # 3) get_all_songs (careful, may be large). We'll only display count and first items
    print('\n3) get_all_songs() [will show up to 10 items]')
    try:
        songs = mj.get_all_songs()
        if songs is None:
            print('  -> None returned')
        else:
            safe_print_items(songs, limit=10)
    except Exception:
        print('  -> Exception calling get_all_songs()')
        traceback.print_exc()

    # 4) If recent albums exist, fetch tracks from the first album
    try:
        if albums:
            first_album = albums[0]
            print(f"\n4) get_tracks_from_album for first recent album: Id={first_album.get('Id')}, Name={first_album.get('Name')}")
            try:
                tracks = mj.get_tracks_from_album(first_album.get('Id'))
                safe_print_items(tracks, limit=20)
            except Exception:
                print('  -> Exception calling get_tracks_from_album()')
                traceback.print_exc()
        else:
            print('\n4) No recent albums to test get_tracks_from_album()')
    except NameError:
        print('\n4) Skipping get_tracks_from_album (albums undefined)')

    # 5) Attempt to download first audio item (dry-run): only show the download URL and HEAD
    print('\n5) Attempt HEAD on the first audio item to check availability (no download)')
    try:
        songs_list = mj.get_all_songs()
        first_audio = None
        if songs_list:
            # pick the first audio item
            for s in songs_list:
                if s.get('Id'):
                    first_audio = s
                    break
        if not first_audio:
            print('  -> No audio items found to attempt HEAD on.')
        else:
            track_id = first_audio['Id']
            download_url = f"{config.JELLYFIN_URL}/Items/{track_id}/Download"
            print(f"  -> Download URL: {download_url}")
            try:
                # Use the wrapper in streaming mode to get response object
                resp = mj._jellyfin_get(download_url, headers=config.HEADERS, stream=True)
                print(f"  -> HEAD/GET status: {resp.status_code}")
                # Don't stream content; just close
                resp.close()
            except Exception:
                print('  -> Exception issuing HEAD/GET to download URL')
                traceback.print_exc()
    except Exception:
        print('  -> Exception preparing HEAD test')
        traceback.print_exc()


if __name__ == '__main__':
    try:
        run()
    except Exception:
        logger.exception('Unhandled exception in jellyfin_debug')
        raise
