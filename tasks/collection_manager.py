import time
import uuid
import logging
import traceback
import numpy as np
import hashlib
from rq import get_current_job, Retry

# Import project modules
from .pocketbase import PocketBaseClient
from .mediaserver import get_recent_albums, get_tracks_from_album

logger = logging.getLogger(__name__)

# --- MODIFIED: Batch processing task now chunks its PocketBase calls ---
def sync_album_batch_task(parent_task_id, album_batch, pocketbase_url, pocketbase_token):
    """
    RQ subtask to synchronize a BATCH of albums with Pocketbase.
    This de-duplicates songs across albums and queries PocketBase in chunks to avoid URL length limits.
    """
    from app import (app, redis_conn, save_task_status, get_tracks_by_ids, save_track_embedding, save_track_analysis,
                     TASK_STATUS_SUCCESS, TASK_STATUS_FAILURE)

    current_job = get_current_job()
    task_id = current_job.id if current_job else str(uuid.uuid4())
    
    batch_album_names = ", ".join([a.get('Name', 'N/A') for a in album_batch])
    batch_album_ids = sorted([a.get('Id') for a in album_batch])
    
    lock_id = hashlib.sha1(str(batch_album_ids).encode()).hexdigest()
    lock_key = f"lock:album_batch_sync:{lock_id}"
    is_lock_acquired = redis_conn.set(lock_key, task_id, nx=True, ex=1800)

    with app.app_context():
        if not is_lock_acquired:
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
                logger.info(f"Batch with albums '{batch_album_names}' contains no tracks. Task complete.")
                return

            all_track_ids = [t['Id'] for t in unique_songs.values()]
            local_tracks_data = get_tracks_by_ids(all_track_ids)
            local_tracks_map = {t['item_id']: t for t in local_tracks_data}

            # --- NEW: Query PocketBase in smaller chunks to avoid long URLs ---
            POCKETBASE_QUERY_CHUNK_SIZE = 20
            remote_records = []
            songs_to_check = [{'artist': t.get('AlbumArtist', 'Unknown Artist'), 'title': t.get('Name')} for t in unique_songs.values()]
            
            for i in range(0, len(songs_to_check), POCKETBASE_QUERY_CHUNK_SIZE):
                chunk = songs_to_check[i:i + POCKETBASE_QUERY_CHUNK_SIZE]
                logger.info(f"Checking existence of {len(chunk)} songs on PocketBase...")
                remote_records.extend(pb_client.get_records_by_songs(chunk, collection='embedding')) # Check embedding collection
                time.sleep(0.2) # Small delay between chunk queries

            remote_tracks_map = {(r['artist'].strip().lower(), r['title'].strip().lower()): r for r in remote_records}

            embeddings_to_upload = []
            scores_to_upload = []

            for remote_key, track in unique_songs.items():
                track_id = track['Id']
                artist = track.get('AlbumArtist', 'Unknown Artist')
                title = track.get('Name')
                log_msg_prefix = f"'{title}' - '{artist}'"
                
                local_track_data = local_tracks_map.get(track_id)
                is_local = local_track_data and local_track_data.get('embedding_vector') is not None and local_track_data['embedding_vector'].size > 0
                is_remote = remote_key in remote_tracks_map

                if is_local and not is_remote:
                    logger.info(f"{log_msg_prefix}: Adding to PocketBase upload queue.")
                    embeddings_to_upload.append({
                        "artist": artist, "title": title,
                        "embedding": local_track_data['embedding_vector'].tolist()
                    })
                    scores_to_upload.append({
                        "title": title, "artist": artist,
                        "tempo": local_track_data.get('tempo'),
                        "key": local_track_data.get('key'),
                        "scale": local_track_data.get('scale'),
                        "mood_vector": local_track_data.get('mood_vector'),
                        "energy": local_track_data.get('energy'),
                        "other_features": local_track_data.get('other_features')
                    })
                elif not is_local and is_remote:
                    logger.info(f"{log_msg_prefix}: Not present locally, downloading from PocketBase.")
                    remote_embedding_record = remote_tracks_map.get(remote_key)
                    remote_score_record = pb_client.get_single_record_by_artist_title(artist, title, collection='score')

                    if remote_embedding_record and remote_score_record:
                        embedding_vector = np.array(remote_embedding_record.get('embedding', [])).astype(np.float32)
                        save_track_embedding(track_id, embedding_vector)
                        
                        moods = {pair.split(':')[0]: float(pair.split(':')[1]) for pair in remote_score_record.get('mood_vector', '').split(',') if ':' in pair}
                        save_track_analysis(
                            item_id=track_id, title=title, author=artist,
                            tempo=remote_score_record.get('tempo'), key=remote_score_record.get('key'),
                            scale=remote_score_record.get('scale'), moods=moods,
                            energy=remote_score_record.get('energy'),
                            other_features=remote_score_record.get('other_features')
                        )

            if embeddings_to_upload:
                logger.info(f"Uploading {len(embeddings_to_upload)} new unique embeddings for batch starting with '{album_batch[0].get('Name')}'...")
                pb_client.create_records_batch(embeddings_to_upload, collection='embedding')
            
            if scores_to_upload:
                logger.info(f"Uploading {len(scores_to_upload)} new unique scores for batch starting with '{album_batch[0].get('Name')}'...")
                pb_client.create_records_batch(scores_to_upload, collection='score')

        except Exception as e:
            logger.error(f"Error in sync_album_batch_task for batch '{batch_album_names}': {e}", exc_info=True)
            save_task_status(task_id, "album_batch_sync", TASK_STATUS_FAILURE, parent_task_id=parent_task_id, sub_type_identifier=lock_id, details={"error": str(e)})
            raise
        finally:
            if is_lock_acquired:
                redis_conn.delete(lock_key)

# --- Main task (remains the same) ---
def sync_collections_task(url, email, password, num_albums):
    """
    Main RQ task to synchronize local embeddings with a PocketBase collection.
    It now groups albums into batches to be processed by sub-tasks.
    """
    from app import (app, redis_conn, save_task_status, rq_queue_default, get_task_info_from_db,
                     TASK_STATUS_STARTED, TASK_STATUS_PROGRESS,
                     TASK_STATUS_SUCCESS, TASK_STATUS_FAILURE, TASK_STATUS_REVOKED)

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

            if not albums:
                log_and_update("No recent albums found to sync.", 100, status=TASK_STATUS_SUCCESS)
                return

            ALBUM_BATCH_SIZE = 20
            album_chunks = [albums[i:i + ALBUM_BATCH_SIZE] for i in range(0, len(albums), ALBUM_BATCH_SIZE)]
            total_chunks = len(album_chunks)
            launched_jobs = []

            for idx, album_batch in enumerate(album_chunks):
                progress = 10 + int(85 * (idx + 1) / total_chunks)
                log_and_update(f"Queueing sync for album batch {idx + 1}/{total_chunks}", progress)
                
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
            log_and_update(f"All {total_launched} batches queued. Monitoring...", 95)

            while True:
                finished_jobs = [j for j in launched_jobs if j.is_finished or j.is_failed or j.is_canceled]
                finished_count = len(finished_jobs)
                if finished_count == total_launched:
                    break
                time.sleep(10)

            failed_jobs = [j for j in launched_jobs if j.is_failed]
            if failed_jobs:
                error_summary = f"{len(failed_jobs)}/{total_launched} batch sync tasks failed."
                raise Exception(error_summary)

            log_and_update("All batches completed successfully.", 100, status=TASK_STATUS_SUCCESS)

        except Exception as e:
            logger.error(f"{log_prefix} An unexpected error occurred: {e}", exc_info=True)
            log_and_update(f"Error: {e}", 100, status=TASK_STATUS_FAILURE)
            raise

