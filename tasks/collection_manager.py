import time
import uuid
import logging
import traceback
import numpy as np
import hashlib
from rq import get_current_job, Retry
import requests # Import requests to catch specific exceptions

# Import project modules
from .pocketbase import PocketBaseClient
from .mediaserver import get_recent_albums, get_tracks_from_album

logger = logging.getLogger(__name__)

def sync_album_batch_task(parent_task_id, album_batch, pocketbase_url, pocketbase_token):
    """
    RQ subtask to synchronize a BATCH of albums with Pocketbase.
    This de-duplicates songs, fetches all records for the relevant artists, then processes locally.
    """
    from app import (app, redis_conn, save_task_status, get_tracks_by_ids, save_track_embedding, save_track_analysis,
                     TASK_STATUS_SUCCESS, TASK_STATUS_FAILURE)

    current_job = get_current_job()
    task_id = current_job.id if current_job else str(uuid.uuid4())
    
    batch_album_names = ", ".join([a.get('Name', 'N/A') for a in album_batch])
    batch_album_ids = sorted([a.get('Id') for a in album_batch])
    
    lock_id = hashlib.sha1(str(batch_album_ids).encode()).hexdigest()
    batch_lock_key = f"lock:album_batch_sync:{lock_id}"
    is_batch_lock_acquired = redis_conn.set(batch_lock_key, task_id, nx=True, ex=1800)

    with app.app_context():
        if not is_batch_lock_acquired:
            logger.warning(f"Album batch starting with '{album_batch[0].get('Name', 'N/A')}' is already being processed. Skipping.")
            return

        try:
            pb_client = PocketBaseClient(base_url=pocketbase_url, token=pocketbase_token)
            
            unique_songs = {}
            for album in album_batch:
                tracks = get_tracks_from_album(album.get('Id'))
                for track in tracks:
                    artist = track.get('AlbumArtist', 'Unknown Artist')
                    title = track.get('Name')
                    unique_songs[(artist.strip().lower(), title.strip().lower())] = track

            if not unique_songs:
                return

            unique_artists = list(set(t.get('AlbumArtist', 'Unknown Artist') for t in unique_songs.values()))
            
            remote_embeddings = pb_client.get_records_by_artists(unique_artists, collection='embedding')
            remote_scores = pb_client.get_records_by_artists(unique_artists, collection='score')
            
            remote_embedding_map = {(r['artist'].strip().lower(), r['title'].strip().lower()): r for r in remote_embeddings}
            remote_score_map = {(r['artist'].strip().lower(), r['title'].strip().lower()): r for r in remote_scores}

            all_track_ids = [t['Id'] for t in unique_songs.values()]
            local_tracks_data = get_tracks_by_ids(all_track_ids)
            local_tracks_map = {t['item_id']: t for t in local_tracks_data}

            embeddings_to_upload = []
            scores_to_upload = []
            song_locks_acquired = [] # Keep track of locks to release them

            for remote_key, track in unique_songs.items():
                track_id = track['Id']
                artist = track.get('AlbumArtist', 'Unknown Artist')
                title = track.get('Name')
                
                local_track_data = local_tracks_map.get(track_id)
                is_local = local_track_data and local_track_data.get('embedding_vector') is not None and local_track_data['embedding_vector'].size > 0
                is_remote = remote_key in remote_embedding_map

                if is_local and not is_remote:
                    # --- NEW: Implement Song-Level Lock ---
                    song_lock_key = f"lock:song_upload:{artist.strip().lower()}:{title.strip().lower()}"
                    if redis_conn.set(song_lock_key, task_id, nx=True, ex=300): # 5-minute lock
                        song_locks_acquired.append(song_lock_key) # Track lock for release
                        embeddings_to_upload.append({
                            "artist": artist, "title": title,
                            "embedding": local_track_data['embedding_vector'].tolist()
                        })
                        scores_to_upload.append({
                            "title": title, "artist": artist,
                            "tempo": local_track_data.get('tempo'), "key": local_track_data.get('key'),
                            "scale": local_track_data.get('scale'), "mood_vector": local_track_data.get('mood_vector'),
                            "energy": local_track_data.get('energy'), "other_features": local_track_data.get('other_features')
                        })
                    else:
                        logger.warning(f"Song '{title}' by '{artist}' is locked by another worker. Skipping upload.")

                elif not is_local and is_remote:
                    remote_embedding_record = remote_embedding_map.get(remote_key)
                    remote_score_record = remote_score_map.get(remote_key)

                    if remote_embedding_record and remote_score_record:
                        embedding_vector = np.array(remote_embedding_record.get('embedding', [])).astype(np.float32)
                        save_track_embedding(track_id, embedding_vector)
                        
                        moods_str = remote_score_record.get('mood_vector', '')
                        moods = {p.split(':')[0]: float(p.split(':')[1]) for p in moods_str.split(',') if ':' in p} if moods_str else {}
                        
                        save_track_analysis(
                            item_id=track_id, title=title, author=artist,
                            tempo=remote_score_record.get('tempo'), key=remote_score_record.get('key'),
                            scale=remote_score_record.get('scale'), moods=moods,
                            energy=remote_score_record.get('energy'),
                            other_features=remote_score_record.get('other_features')
                        )
            
            # The original try/except blocks are kept as a final safeguard
            try:
                if embeddings_to_upload:
                    logger.info(f"Uploading {len(embeddings_to_upload)} new embedding records...")
                    pb_client.create_records_batch(embeddings_to_upload, collection='embedding')
            except requests.exceptions.HTTPError as e:
                if e.response and e.response.status_code == 400:
                    logger.warning(f"Embedding batch upload failed, likely due to duplicates (this is safe). Details: {e.response.text}")
                else: raise e

            try:
                if scores_to_upload:
                    logger.info(f"Uploading {len(scores_to_upload)} new score records...")
                    pb_client.create_records_batch(scores_to_upload, collection='score')
            except requests.exceptions.HTTPError as e:
                if e.response and e.response.status_code == 400:
                    logger.warning(f"Score batch upload failed, likely due to duplicates (this is safe). Details: {e.response.text}")
                else: raise e

        except Exception as e:
            logger.error(f"Error in sync_album_batch_task for batch '{batch_album_names}': {e}", exc_info=True)
            save_task_status(task_id, "album_batch_sync", TASK_STATUS_FAILURE, parent_task_id=parent_task_id, sub_type_identifier=lock_id, details={"error": str(e)})
            raise
        finally:
            # --- NEW: Release all acquired locks ---
            if song_locks_acquired:
                redis_conn.delete(*song_locks_acquired)
            if is_batch_lock_acquired:
                redis_conn.delete(batch_lock_key)

# --- Main task ---
def sync_collections_task(url, email, password, num_albums):
    from app import (app, save_task_status, rq_queue_default,
                     TASK_STATUS_STARTED, TASK_STATUS_PROGRESS,
                     TASK_STATUS_SUCCESS, TASK_STATUS_FAILURE)

    current_job = get_current_job()
    current_task_id = current_job.id if current_job else str(uuid.uuid4())
    log_prefix = f"[SyncTask-{current_task_id}]"

    with app.app_context():
        def log_and_update(message, progress, status=TASK_STATUS_PROGRESS, **kwargs):
            logger.info(f"{log_prefix} {message}")
            details = {"log": [f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"], **kwargs}
            save_task_status(current_task_id, "main_sync", status, progress=progress, details=details)

        try:
            log_and_update("Starting collection synchronization...", 0, status=TASK_STATUS_STARTED)
            
            pb_client = PocketBaseClient(base_url=url, email=email, password=password)
            pb_client.authenticate()
            pocketbase_token = pb_client.token

            log_and_update("Authentication successful. Fetching recent albums...", 5)
            albums = get_recent_albums(num_albums)
            total_albums = len(albums)

            if not albums:
                log_and_update("No recent albums found to sync.", 100, status=TASK_STATUS_SUCCESS)
                return

            ALBUM_BATCH_SIZE = 20
            album_chunks = [albums[i:i + ALBUM_BATCH_SIZE] for i in range(0, len(albums), ALBUM_BATCH_SIZE)]
            total_chunks = len(album_chunks)
            launched_jobs = []
            
            log_and_update(f"Found {total_albums} albums. Preparing to queue {total_chunks} batches...", 10)
            albums_queued_so_far = 0
            for idx, album_batch in enumerate(album_chunks):
                albums_queued_so_far += len(album_batch)
                logger.info(f"{log_prefix} Queueing sync for albums {albums_queued_so_far}/{total_albums} (batch {idx + 1}/{total_chunks})")
                
                sub_job = rq_queue_default.enqueue(
                    'tasks.collection_manager.sync_album_batch_task',
                    args=(current_task_id, album_batch, url, pocketbase_token),
                    job_id=str(uuid.uuid4()),
                    retry=Retry(max=2),
                    job_timeout='1h'
                )
                launched_jobs.append(sub_job)
                time.sleep(1)

            total_launched = len(launched_jobs)
            log_and_update(f"All {total_launched} batches queued. Monitoring for completion...", 10)

            while True:
                finished_jobs = [j for j in launched_jobs if j.is_finished or j.is_failed or j.is_canceled]
                
                if total_launched > 0:
                    progress = 10 + int(85 * (len(finished_jobs) / total_launched))
                else:
                    progress = 100

                status_message = f"Monitoring... {len(finished_jobs)}/{total_launched} batches completed."
                log_and_update(status_message, progress)

                if len(finished_jobs) == total_launched:
                    break
                
                time.sleep(10)

            failed_jobs = [j for j in launched_jobs if j.is_failed]
            if failed_jobs:
                raise Exception(f"{len(failed_jobs)}/{total_launched} batch sync tasks failed.")

            log_and_update("All batches completed successfully.", 100, status=TASK_STATUS_SUCCESS)

        except Exception as e:
            logger.error(f"{log_prefix} An unexpected error occurred: {e}", exc_info=True)
            log_and_update(f"Error: {e}", 100, status=TASK_STATUS_FAILURE)
            raise

