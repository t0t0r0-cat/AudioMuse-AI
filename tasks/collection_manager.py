import time
import uuid
import logging
import traceback
import numpy as np
import hashlib
import json
from rq import get_current_job, Retry
from rq.job import Job
from rq.exceptions import NoSuchJobError
import requests # Import requests to catch specific exceptions

# Import project modules
from .pocketbase import PocketBaseClient
from .mediaserver import get_recent_albums, get_tracks_from_album

logger = logging.getLogger(__name__)

# Lua script for an atomic check-and-delete. This is the safest way to release a lock.
# It ensures that we only delete the lock if we are still the owner (the value matches our task_id).
LUA_SAFE_RELEASE_LOCK_SCRIPT = """
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("del", KEYS[1])
else
    return 0
end
"""

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

            batch_requests_to_upload = []

            for remote_key, track in unique_songs.items():
                track_id = track['Id']
                artist = track.get('AlbumArtist', 'Unknown Artist')
                title = track.get('Name')
                
                local_track_data = local_tracks_map.get(track_id)
                is_present_locally = local_track_data and local_track_data.get('embedding_vector') is not None and local_track_data['embedding_vector'].size > 0
                is_present_remotely = remote_key in remote_embedding_map

                if is_present_locally and not is_present_remotely:
                    embedding_body = {
                        "artist": artist, "title": title,
                        "embedding": json.dumps(local_track_data['embedding_vector'].tolist())
                    }
                    batch_requests_to_upload.append({
                        "method": "POST",
                        "url": "/api/collections/embedding/records",
                        "body": embedding_body
                    })

                    score_body = {
                        "title": title, "artist": artist,
                        "tempo": local_track_data.get('tempo'), "key": local_track_data.get('key'),
                        "scale": local_track_data.get('scale'), "mood_vector": local_track_data.get('mood_vector'),
                        "energy": local_track_data.get('energy'), "other_features": local_track_data.get('other_features')
                    }
                    batch_requests_to_upload.append({
                        "method": "POST",
                        "url": "/api/collections/score/records",
                        "body": score_body
                    })
                    logger.info(f"{log_prefix} Queued for atomic upload: '{title}' by '{artist}'.")

                elif not is_present_locally and is_present_remotely:
                    remote_embedding_record = remote_embedding_map.get(remote_key)
                    remote_score_record = remote_score_map.get(remote_key)

                    if remote_embedding_record and remote_score_record:
                        try:
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
                            logger.info(f"{log_prefix} Synced down from remote: '{title}' by '{artist}'.")
                        except (json.JSONDecodeError, TypeError) as e:
                            logger.warning(f"{log_prefix} Could not parse/process remote record for '{title}'. Skipping sync-down. Error: {e}")
                
                else:
                    logger.debug(f"{log_prefix} No action needed for '{title}' by '{artist}' (State: Local={is_present_locally}, Remote={is_present_remotely}).")

            if batch_requests_to_upload:
                # MODIFIED: Chunk the batch requests to avoid server-side timeouts or payload size limits.
                # A chunk size of 50 means 25 songs (1 embedding + 1 score per song).
                REQUEST_CHUNK_SIZE = 50 
                request_chunks = [batch_requests_to_upload[i:i + REQUEST_CHUNK_SIZE] for i in range(0, len(batch_requests_to_upload), REQUEST_CHUNK_SIZE)]
                
                logger.info(f"{log_prefix} Total requests to upload: {len(batch_requests_to_upload)}. Split into {len(request_chunks)} chunks.")

                for i, chunk in enumerate(request_chunks):
                    try:
                        num_songs_in_chunk = len(chunk) // 2
                        logger.info(f"{log_prefix} Atomically uploading chunk {i+1}/{len(request_chunks)} with {num_songs_in_chunk} songs...")
                        pb_client.submit_batch_request(chunk)
                    except requests.exceptions.HTTPError as e:
                        logger.warning(f"{log_prefix} Atomic batch upload for chunk {i+1} failed and was rolled back. Details: {e.response.text if e.response else str(e)}")
                        # For robustness, we log the warning and continue with other chunks.

            save_task_status(task_id, "album_batch_sync", TASK_STATUS_SUCCESS, parent_task_id=parent_task_id, sub_type_identifier=lock_id, progress=100, details={"message": "Batch processed successfully."})

        except Exception as e:
            logger.error(f"{log_prefix} Error in sync_album_batch_task: {e}", exc_info=True)
            failure_details = {
                "error": str(e), 
                "traceback": traceback.format_exc(),
                "batch_name": batch_name_short 
            }
            save_task_status(task_id, "album_batch_sync", TASK_STATUS_FAILURE, parent_task_id=parent_task_id, sub_type_identifier=lock_id, details=failure_details)
            raise
        finally:
            if is_batch_lock_acquired:
                try:
                    released = redis_conn.eval(LUA_SAFE_RELEASE_LOCK_SCRIPT, 1, batch_lock_key, task_id)
                    if released:
                        logger.info(f"{log_prefix} Safely released batch lock: {batch_lock_key}")
                    else:
                        logger.warning(f"{log_prefix} Did not release batch lock {batch_lock_key} as I am no longer the owner (it may have expired).")
                except Exception as e:
                    logger.error(f"{log_prefix} Failed to execute Lua script to release batch lock {batch_lock_key}: {e}")

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
            current_task_info = get_task_info_from_db(current_task_id)
            db_details = {}
            if current_task_info and current_task_info.get('details'):
                try:
                    db_details = json.loads(current_task_info['details'])
                except (json.JSONDecodeError, TypeError):
                    db_details = {"raw_details": current_task_info['details']}

            log_entry = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
            if 'log' not in db_details or not isinstance(db_details.get('log'), list):
                db_details['log'] = []
            db_details['log'].append(log_entry)
            
            if 'details' in kwargs:
                db_details.update(kwargs['details'])

            save_task_status(current_task_id, "main_collection_sync", status, progress=progress, details=db_details)

        try:
            log_and_update("Starting collection synchronization...", 0, status=TASK_STATUS_STARTED)
            
            pb_client = PocketBaseClient(base_url=url, email=email, password=password, log_prefix=log_prefix)

            try:
                pb_client.authenticate()
                pocketbase_token = pb_client.token
            except ConnectionError as auth_error:
                if '400' in str(auth_error):
                    user_friendly_error = "Authentication failed. Please check the PocketBase URL, email, and password."
                    log_and_update(user_friendly_error, 100, status=TASK_STATUS_FAILURE, details={"error": user_friendly_error, "original_error": str(auth_error)})
                else:
                    user_friendly_error = f"Could not connect to PocketBase server. Please verify the URL and network connectivity."
                    log_and_update(user_friendly_error, 100, status=TASK_STATUS_FAILURE, details={"error": user_friendly_error, "original_error": str(auth_error)})
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

            completed_sub_tasks_count = 0
            while completed_sub_tasks_count < total_launched:
                main_task_info = get_task_info_from_db(current_task_id)
                if main_task_info and main_task_info.get('status') == TASK_STATUS_REVOKED:
                    log_and_update("Main task revoked. Stopping monitoring.", 99)
                    return

                all_child_tasks = get_child_tasks_from_db(current_task_id)
                finished_tasks = [
                    t for t in all_child_tasks 
                    if t.get('status') in [TASK_STATUS_SUCCESS, TASK_STATUS_FAILURE, TASK_STATUS_REVOKED]
                ]
                completed_sub_tasks_count = len(finished_tasks)

                if total_launched > 0:
                    progress = 15 + int(80 * (completed_sub_tasks_count / total_launched))
                else:
                    progress = 100

                status_message = f"Monitoring... {completed_sub_tasks_count}/{total_launched} batches completed."
                log_and_update(status_message, progress)
                
                if completed_sub_tasks_count >= total_launched:
                    break
                
                time.sleep(5)

            log_and_update("All batches have completed. Performing final check...", 96)
            all_child_tasks = get_child_tasks_from_db(current_task_id)
            
            tasks_by_batch = {}
            for task in all_child_tasks:
                batch_id = task.get('sub_type_identifier')
                if batch_id:
                    tasks_by_batch.setdefault(batch_id, []).append(task)

            unprocessed_batches_info = []
            for batch_id, tasks in tasks_by_batch.items():
                is_processed = False
                final_failure_details = None

                for t in tasks:
                    details_dict = {}
                    details_str = t.get('details')
                    if details_str and isinstance(details_str, str):
                        try:
                            details_dict = json.loads(details_str)
                        except json.JSONDecodeError:
                            logger.warning(f"Could not parse details JSON for sub-task {t.get('task_id')}")

                    if (t.get('status') == TASK_STATUS_SUCCESS and 
                            "Skipped" not in details_dict.get('message', '')):
                        is_processed = True
                        break 
                    
                    if t.get('status') == TASK_STATUS_FAILURE:
                        final_failure_details = {
                            "task_id": t.get('task_id'),
                            "error": details_dict.get('error', 'Unknown error'),
                            "batch_name": details_dict.get('batch_name', 'Unknown Batch')
                        }
                
                if not is_processed:
                    unprocessed_batches_info.append(final_failure_details or 
                        {"batch_id": batch_id, "error": "Batch was skipped or status was inconclusive."})

            if unprocessed_batches_info:
                error_summary = f"{len(unprocessed_batches_info)}/{total_launched} batches failed."
                detailed_errors = [
                    f"  - Batch '{info.get('batch_name', 'N/A')}' failed in task {info.get('task_id', 'N/A')} with error: {info.get('error', 'N/A').splitlines()[0]}" 
                    for info in unprocessed_batches_info
                ]
                full_error_message = f"{error_summary}\n\nReasons:\n" + "\n".join(detailed_errors)
                
                log_and_update(error_summary, 100, status=TASK_STATUS_FAILURE, details={"error": full_error_message, "unprocessed_batches": unprocessed_batches_info})
                raise Exception(full_error_message)

            log_and_update("All batches completed successfully.", 100, status=TASK_STATUS_SUCCESS)

        except Exception as e:
            logger.error(f"{log_prefix} An unexpected error occurred in main sync task: {e}", exc_info=True)
            task_info = get_task_info_from_db(current_task_id)
            if not task_info or task_info.get('status') != TASK_STATUS_FAILURE:
                 log_and_update(f"Error: {e}", 100, status=TASK_STATUS_FAILURE, details={"error": str(e), "traceback": traceback.format_exc()})
            raise

