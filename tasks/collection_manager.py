import time
import uuid
import logging
import traceback
import numpy as np
from rq import get_current_job, Retry

# Import project modules
from .pocketbase import PocketBaseClient
from .mediaserver import get_recent_albums, get_tracks_from_album

logger = logging.getLogger(__name__)

def sync_album_task(parent_task_id, album, pocketbase_url, pocketbase_token):
    """
    RQ subtask to synchronize a single album with Pocketbase based on the specified logic.
    Includes a Redis lock to prevent concurrent processing of the same album.
    """
    # --- Local imports to prevent circular dependency ---
    from app import (app, redis_conn, save_task_status, get_tracks_by_ids, save_track_embedding, save_track_analysis,
                     TASK_STATUS_STARTED, TASK_STATUS_PROGRESS,
                     TASK_STATUS_SUCCESS, TASK_STATUS_FAILURE, TASK_STATUS_REVOKED)

    current_job = get_current_job()
    task_id = current_job.id if current_job else str(uuid.uuid4())
    album_name = album.get('Name', 'Unknown Album')
    album_id = album.get('Id')

    # --- Redis Locking ---
    lock_key = f"lock:album_sync:{album_id}"
    is_lock_acquired = redis_conn.set(lock_key, task_id, nx=True, ex=600)

    with app.app_context():
        if not is_lock_acquired:
            logger.warning(f"Album '{album_name}' ({album_id}) is already being processed by another worker. Skipping.")
            return

        try:
            pb_client = PocketBaseClient(base_url=pocketbase_url, token=pocketbase_token)
            tracks = get_tracks_from_album(album_id)

            if not tracks:
                save_task_status(task_id, "album_sync", TASK_STATUS_SUCCESS, parent_task_id=parent_task_id, sub_type_identifier=album_id, progress=100, details={"message": "Album has no tracks to sync."})
                return

            track_ids = [t['Id'] for t in tracks]
            local_tracks_data = get_tracks_by_ids(track_ids)
            local_tracks_map = {t['item_id']: t for t in local_tracks_data}

            remote_records = pb_client.get_records_by_artist_and_titles(
                album.get('AlbumArtist', 'Unknown Artist'),
                [t['Name'] for t in tracks]
            )
            remote_tracks_map = {(r['artist'].strip().lower(), r['title'].strip().lower()): r for r in remote_records}

            records_to_upload = []
            tracks_downloaded = 0
            
            for track in tracks:
                track_id = track['Id']
                artist = track.get('AlbumArtist', 'Unknown Artist')
                title = track.get('Name')
                log_msg_prefix = f"'{title}' - '{artist}'"
                
                local_track_data = local_tracks_map.get(track_id)
                is_local = local_track_data and local_track_data.get('embedding_vector') is not None and local_track_data['embedding_vector'].size > 0

                remote_key = (artist.strip().lower(), title.strip().lower())
                is_remote = remote_key in remote_tracks_map

                if is_local and is_remote:
                    logger.info(f"{log_msg_prefix}: Present Locally and on pocketbase")
                elif is_local and not is_remote:
                    logger.info(f"{log_msg_prefix}: Adding to PocketBase upload queue.")
                    records_to_upload.append({
                        "artist": artist,
                        "title": title,
                        "embedding": local_track_data['embedding_vector'].tolist()
                    })
                elif not is_local and is_remote:
                    logger.info(f"{log_msg_prefix}: Not present locally, downloading from pocketbase")
                    remote_record = remote_tracks_map[remote_key]
                    embedding_list = remote_record.get('embedding')
                    if embedding_list and isinstance(embedding_list, list):
                        embedding_vector = np.array(embedding_list, dtype=np.float32)
                        save_track_analysis(item_id=track_id, title=title, author=artist, tempo=None, key=None, scale=None, moods={}, energy=None, other_features=None)
                        save_track_embedding(track_id, embedding_vector)
                        tracks_downloaded += 1
                    else:
                        logger.warning(f"{log_msg_prefix}: Found on PocketBase, but embedding data is missing or invalid.")
                elif not is_local and not is_remote:
                    logger.info(f"{log_msg_prefix}: Not present locally or on pocketbase")

            if records_to_upload:
                logger.info(f"Uploading {len(records_to_upload)} new records for album '{album_name}' via batch API.")
                success = pb_client.create_records_batch(records_to_upload)
                if not success:
                    logger.warning(f"One or more records failed to upload for album '{album_name}'. This is often due to duplicates and is not a fatal error. See previous logs for details.")

            details = {"uploaded_count": len(records_to_upload), "downloaded_count": tracks_downloaded, "message": "Sync complete."}
            save_task_status(task_id, "album_sync", TASK_STATUS_SUCCESS, parent_task_id=parent_task_id, sub_type_identifier=album_id, progress=100, details=details)
        
        except Exception as e:
            logger.error(f"Error in sync_album_task for album '{album_name}': {e}", exc_info=True)
            save_task_status(task_id, "album_sync", TASK_STATUS_FAILURE, parent_task_id=parent_task_id, sub_type_identifier=album_id, progress=100, details={"error": str(e)})
            raise
        finally:
            if is_lock_acquired:
                redis_conn.delete(lock_key)
                logger.info(f"Released lock for album '{album_name}'.")

def sync_collections_task(url, email, password, num_albums):
    """
    Main RQ task to synchronize local embeddings with a PocketBase collection.
    """
    from app import (app, redis_conn, save_task_status, rq_queue_default, get_task_info_from_db,
                     TASK_STATUS_STARTED, TASK_STATUS_PROGRESS,
                     TASK_STATUS_SUCCESS, TASK_STATUS_FAILURE, TASK_STATUS_REVOKED)

    current_job = get_current_job()
    current_task_id = current_job.id if current_job else str(uuid.uuid4())
    log_prefix = f"[SyncTask-{current_task_id}]"

    processed_albums_key = f"processed_albums:{current_task_id}"

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

            total_albums = len(albums)
            launched_jobs = []
            for idx, album in enumerate(albums):
                album_id = album.get('Id')
                if redis_conn.sismember(processed_albums_key, album_id):
                    logger.info(f"{log_prefix} Album '{album.get('Name')}' already queued in this run. Skipping.")
                    continue

                progress = 10 + int(85 * (idx + 1) / total_albums)
                log_and_update(f"Queueing sync for album: {album.get('Name')} ({idx + 1}/{total_albums})", progress)
                
                sub_job = rq_queue_default.enqueue(
                    'tasks.collection_manager.sync_album_task',
                    args=(current_task_id, album, url, pocketbase_token),
                    job_id=str(uuid.uuid4()),
                    retry=Retry(max=2)
                )
                launched_jobs.append(sub_job)
                redis_conn.sadd(processed_albums_key, album_id)
                # --- MODIFIED: Increased staggering delay ---
                time.sleep(0.5)
            
            redis_conn.expire(processed_albums_key, 3600)

            total_launched = len(launched_jobs)
            if total_launched == 0:
                log_and_update("No new albums to process in this run.", 100, status=TASK_STATUS_SUCCESS)
                return

            log_and_update(f"All {total_launched} unique albums queued. Monitoring sub-tasks...", 95)

            while True:
                if get_task_info_from_db(current_task_id).get('status') == 'REVOKED':
                    log_and_update("Main sync task revoked. Stopping.", 100, status=TASK_STATUS_REVOKED)
                    return

                finished_jobs = [j for j in launched_jobs if j.is_finished or j.is_failed or j.is_canceled]
                finished_count = len(finished_jobs)
                
                monitoring_progress = 95 + int(5 * (finished_count / total_launched))
                log_and_update(f"Waiting for sub-tasks to complete... ({finished_count}/{total_launched})", monitoring_progress)

                if finished_count == total_launched:
                    break
                time.sleep(5) 

            failed_jobs = [j for j in launched_jobs if j.is_failed]
            if failed_jobs:
                error_summary = f"{len(failed_jobs)}/{total_launched} album sync tasks failed."
                logger.error(f"{log_prefix} {error_summary} - Failed IDs: {[j.id for j in failed_jobs]}")
                raise Exception(error_summary)

            log_and_update("All sub-tasks have completed successfully.", 100, status=TASK_STATUS_SUCCESS)

        except (ConnectionError, ValueError) as e:
            logger.error(f"{log_prefix} Pre-flight check or authentication failed: {e}", exc_info=True)
            log_and_update(f"Error: {e}", 100, status=TASK_STATUS_FAILURE, final_summary_details={"error": str(e), "traceback": traceback.format_exc()})
            raise
        except Exception as e:
            logger.error(f"{log_prefix} An unexpected error occurred or sub-tasks failed: {e}", exc_info=True)
            log_and_update(f"An unexpected error occurred or sub-tasks failed: {e}", 100, status=TASK_STATUS_FAILURE, final_summary_details={"error": str(e), "traceback": traceback.format_exc()})
            raise

