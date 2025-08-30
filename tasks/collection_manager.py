import time
import uuid
import logging
import traceback
import numpy as np
import hashlib
from rq import get_current_job, Retry
from rq.job import Job
from rq.exceptions import NoSuchJobError
import requests # Import requests to catch specific exceptions

# Import project modules
from .pocketbase import PocketBaseClient
from .mediaserver import get_recent_albums, get_tracks_from_album

logger = logging.getLogger(__name__)

def sync_album_batch_task(parent_task_id, album_batch, pocketbase_url, pocketbase_token, main_task_log_prefix="[MainTask-Unknown]"):
    """
    RQ subtask to synchronize a BATCH of albums with Pocketbase.
    This de-duplicates songs, fetches all records for the relevant artists, then processes locally.
    """
    from app import (app, redis_conn, save_task_status, get_tracks_by_ids, save_track_embedding, save_track_analysis,
                     get_task_info_from_db, TASK_STATUS_STARTED, TASK_STATUS_PROGRESS,
                     TASK_STATUS_SUCCESS, TASK_STATUS_FAILURE, TASK_STATUS_REVOKED)

    current_job = get_current_job()
    task_id = current_job.id if current_job else str(uuid.uuid4())
    
    batch_album_names = ", ".join([a.get('Name', 'N/A') for a in album_batch])
    batch_name_short = album_batch[0].get('Name', 'UnknownAlbum') if album_batch else 'EmptyBatch'
    log_prefix = f"{main_task_log_prefix} -> [SubTask-{task_id[:8]}-{batch_name_short}]"
    
    batch_album_ids = sorted([a.get('Id') for a in album_batch if a.get('Id')])
    lock_id = hashlib.sha1(str(batch_album_ids).encode()).hexdigest()
    batch_lock_key = f"lock:album_batch_sync:{lock_id}"
    is_batch_lock_acquired = redis_conn.set(batch_lock_key, task_id, nx=True, ex=1800)

    with app.app_context():
        # --- Cancellation Check ---
        # Check if the parent task has been revoked before starting any work.
        parent_info = get_task_info_from_db(parent_task_id)
        if parent_info and parent_info.get('status') == TASK_STATUS_REVOKED:
            logger.warning(f"{log_prefix} Parent task was revoked. Cancelling this sub-task.")
            save_task_status(task_id, "album_batch_sync", TASK_STATUS_REVOKED, parent_task_id=parent_task_id, sub_type_identifier=lock_id, details={"message": "Cancelled due to parent task revocation."})
            return

        # --- Initial Status Update ---
        initial_details = {"batch_name": batch_name_short, "log": [f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Sub-task started for batch: {batch_album_names}."]}
        save_task_status(task_id, "album_batch_sync", TASK_STATUS_STARTED, parent_task_id=parent_task_id, sub_type_identifier=lock_id, progress=0, details=initial_details)

        if not is_batch_lock_acquired:
            logger.warning(f"{log_prefix} Album batch is already being processed by another worker. Skipping.")
            save_task_status(task_id, "album_batch_sync", TASK_STATUS_SUCCESS, parent_task_id=parent_task_id, sub_type_identifier=lock_id, progress=100, details={"message": "Skipped, batch already locked."})
            return

        try:
            pb_client = PocketBaseClient(base_url=pocketbase_url, token=pocketbase_token, log_prefix=log_prefix)
            
            unique_songs = {}
            for album in album_batch:
                tracks = get_tracks_from_album(album.get('Id'))
                for track in tracks:
                    artist = track.get('AlbumArtist', 'Unknown Artist')
                    title = track.get('Name')
                    unique_songs[(artist.strip().lower(), title.strip().lower())] = track

            if not unique_songs:
                logger.info(f"{log_prefix} No unique songs found in this batch. Task successful.")
                save_task_status(task_id, "album_batch_sync", TASK_STATUS_SUCCESS, parent_task_id=parent_task_id, sub_type_identifier=lock_id, progress=100, details={"message": "No songs to process."})
                return

            unique_artists = list(set(t.get('AlbumArtist', 'Unknown Artist') for t in unique_songs.values()))
            
            logger.info(f"{log_prefix} Fetching remote records for {len(unique_artists)} artists.")
            remote_embeddings = pb_client.get_records_by_artists(unique_artists, collection='embedding')
            remote_scores = pb_client.get_records_by_artists(unique_artists, collection='score')
            
            remote_embedding_map = {(r['artist'].strip().lower(), r['title'].strip().lower()): r for r in remote_embeddings}
            remote_score_map = {(r['artist'].strip().lower(), r['title'].strip().lower()): r for r in remote_scores}

            all_track_ids = [t['Id'] for t in unique_songs.values()]
            local_tracks_data = get_tracks_by_ids(all_track_ids)
            local_tracks_map = {t['item_id']: t for t in local_tracks_data}

            embeddings_to_upload = []
            scores_to_upload = []
            song_locks_acquired = [] 

            for remote_key, track in unique_songs.items():
                track_id = track['Id']
                artist = track.get('AlbumArtist', 'Unknown Artist')
                title = track.get('Name')
                
                local_track_data = local_tracks_map.get(track_id)
                is_local = local_track_data and local_track_data.get('embedding_vector') is not None and local_track_data['embedding_vector'].size > 0
                is_remote = remote_key in remote_embedding_map

                if is_local and not is_remote:
                    song_lock_key = f"lock:song_upload:{artist.strip().lower()}:{title.strip().lower()}"
                    if redis_conn.set(song_lock_key, task_id, nx=True, ex=300):
                        song_locks_acquired.append(song_lock_key)
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
                        logger.warning(f"{log_prefix} Song '{title}' by '{artist}' is locked by another worker. Skipping upload.")

                elif not is_local and is_remote:
                    remote_embedding_record = remote_embedding_map.get(remote_key)
                    remote_score_record = remote_score_map.get(remote_key)

                    if remote_embedding_record and remote_score_record:
                        try:
                            # --- FIX: Correctly deserialize the embedding JSON string from PocketBase ---
                            embedding_json = remote_embedding_record.get('embedding', '[]')
                            embedding_list = json.loads(embedding_json) if embedding_json else []
                            embedding_vector = np.array(embedding_list).astype(np.float32)
                            
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
                        except (json.JSONDecodeError, TypeError) as e:
                            logger.warning(f"{log_prefix} Could not parse or process remote record for '{title}'. Skipping sync-down. Error: {e}")
            
            try:
                if embeddings_to_upload:
                    logger.info(f"{log_prefix} Uploading {len(embeddings_to_upload)} new embedding records...")
                    pb_client.create_records_batch(embeddings_to_upload, collection='embedding')
            except requests.exceptions.HTTPError as e:
                # A 400 error can be due to malformed data or duplicates. Log as a warning but don't crash the whole task.
                logger.warning(f"{log_prefix} A non-critical error occurred during embedding batch upload (e.g., duplicates, bad data). Details: {e.response.text if e.response else str(e)}")
                # We don't re-raise, allowing the task to continue to the scores upload.

            try:
                if scores_to_upload:
                    logger.info(f"{log_prefix} Uploading {len(scores_to_upload)} new score records...")
                    pb_client.create_records_batch(scores_to_upload, collection='score')
            except requests.exceptions.HTTPError as e:
                logger.warning(f"{log_prefix} A non-critical error occurred during score batch upload (e.g., duplicates, bad data). Details: {e.response.text if e.response else str(e)}")
                # We don't re-raise here either.

            save_task_status(task_id, "album_batch_sync", TASK_STATUS_SUCCESS, parent_task_id=parent_task_id, sub_type_identifier=lock_id, progress=100, details={"message": "Batch processed successfully."})

        except Exception as e:
            logger.error(f"{log_prefix} Error in sync_album_batch_task: {e}", exc_info=True)
            save_task_status(task_id, "album_batch_sync", TASK_STATUS_FAILURE, parent_task_id=parent_task_id, sub_type_identifier=lock_id, details={"error": str(e), "traceback": traceback.format_exc()})
            raise
        finally:
            if song_locks_acquired:
                logger.info(f"{log_prefix} Releasing {len(song_locks_acquired)} song locks.")
                redis_conn.delete(*song_locks_acquired)
            if is_batch_lock_acquired:
                logger.info(f"{log_prefix} Releasing batch lock: {batch_lock_key}")
                redis_conn.delete(batch_lock_key)

# --- Main task ---
def sync_collections_task(url, email, password, num_albums):
    from app import (app, redis_conn, save_task_status, get_task_info_from_db, rq_queue_default,
                     get_child_tasks_from_db, TASK_STATUS_STARTED, TASK_STATUS_PROGRESS,
                     TASK_STATUS_SUCCESS, TASK_STATUS_FAILURE, TASK_STATUS_REVOKED)

    current_job = get_current_job()
    current_task_id = current_job.id if current_job else str(uuid.uuid4())
    log_prefix = f"[MainSyncTask-{current_task_id}]"

    with app.app_context():
        def log_and_update(message, progress, status=TASK_STATUS_PROGRESS, **kwargs):
            logger.info(f"{log_prefix} {message}")
            # Ensure details are always a dictionary
            db_details = kwargs.get('details', {})
            # Append new log message to existing log list in details
            log_entry = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
            if 'log' not in db_details or not isinstance(db_details['log'], list):
                db_details['log'] = []
            db_details['log'].append(log_entry)
            
            save_task_status(current_task_id, "main_collection_sync", status, progress=progress, details=db_details)

        try:
            log_and_update("Starting collection synchronization...", 0, status=TASK_STATUS_STARTED)
            
            pb_client = PocketBaseClient(base_url=url, email=email, password=password, log_prefix=log_prefix)

            # --- Specific handling for authentication ---
            try:
                pb_client.authenticate()
                pocketbase_token = pb_client.token
            except ConnectionError as auth_error:
                # Check if it's a 400 error, which implies bad credentials
                if '400' in str(auth_error):
                    user_friendly_error = "Authentication failed. Please check the PocketBase URL, email, and password."
                    log_and_update(user_friendly_error, 100, status=TASK_STATUS_FAILURE, details={"error": user_friendly_error, "original_error": str(auth_error)})
                else: # Other connection error (timeout, DNS, etc.)
                    user_friendly_error = f"Could not connect to PocketBase server. Please verify the URL and network connectivity."
                    log_and_update(user_friendly_error, 100, status=TASK_STATUS_FAILURE, details={"error": user_friendly_error, "original_error": str(auth_error)})
                # Re-raise the exception to ensure the RQ job is marked as failed.
                raise

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
                # --- MODIFIED: Provide granular progress updates during the enqueuing loop ---
                enqueuing_progress = 10 + int(5 * ((idx + 1) / total_chunks))
                log_and_update(f"Queueing batch {idx + 1}/{total_chunks}...", enqueuing_progress)

                albums_queued_so_far += len(album_batch)

                sub_job = rq_queue_default.enqueue(
                    'tasks.collection_manager.sync_album_batch_task',
                    args=(current_task_id, album_batch, url, pocketbase_token, log_prefix),
                    job_id=str(uuid.uuid4()),
                    retry=Retry(max=2),
                    job_timeout='1h'
                )
                launched_jobs.append(sub_job)

            total_launched = len(launched_jobs)
            log_and_update(f"All {total_launched} batches queued. Monitoring for completion...", 15)

            while True:
                # --- Cancellation Check ---
                # Check if this main task has been revoked by an API call.
                main_task_info = get_task_info_from_db(current_task_id)
                if main_task_info and main_task_info.get('status') == TASK_STATUS_REVOKED:
                    log_and_update("Main task revoked. Stopping monitoring.", 99)
                    # The recursive cancel from the API will handle sub-jobs,
                    # so we can just exit here.
                    return

                # Fetch jobs from RQ to get their latest status
                finished_job_ids = set()
                try:
                    fetched_jobs = Job.fetch_many([j.id for j in launched_jobs], connection=redis_conn)
                    for job in fetched_jobs:
                        if job and (job.is_finished or job.is_failed or job.is_canceled):
                            finished_job_ids.add(job.id)
                except (NoSuchJobError, ConnectionError) as e:
                    logger.warning(f"{log_prefix} Could not fetch all job statuses from Redis: {e}. Will retry.")


                if total_launched > 0:
                    progress = 15 + int(80 * (len(finished_job_ids) / total_launched))
                else:
                    progress = 100

                status_message = f"Monitoring... {len(finished_job_ids)}/{total_launched} batches completed."
                log_and_update(status_message, progress)

                if len(finished_job_ids) == total_launched:
                    break
                
                time.sleep(10)

            # Final check for failed jobs based on final DB state of sub-tasks
            failed_sub_tasks = [
                task for task in get_child_tasks_from_db(current_task_id) 
                if task.get('status') == TASK_STATUS_FAILURE
            ]
            if failed_sub_tasks:
                raise Exception(f"{len(failed_sub_tasks)}/{total_launched} batch sync tasks failed.")

            log_and_update("All batches completed successfully.", 100, status=TASK_STATUS_SUCCESS)

        except Exception as e:
            logger.error(f"{log_prefix} An unexpected error occurred in main sync task: {e}", exc_info=True)
            log_and_update(f"Error: {e}", 100, status=TASK_STATUS_FAILURE, details={"error": str(e), "traceback": traceback.format_exc()})
            raise
