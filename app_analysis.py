# app_analysis.py
from flask import Blueprint, jsonify, request
import uuid
import logging

# Import configuration from the main config.py
from config import NUM_RECENT_ALBUMS, TOP_N_MOODS

# RQ import
from rq import Retry

logger = logging.getLogger(__name__)

# Create a Blueprint for analysis-related routes
analysis_bp = Blueprint('analysis_bp', __name__)

@analysis_bp.route('/cleaning', methods=['GET'])
def cleaning_page():
    """
    Serves the HTML page for the Database Cleaning feature.
    ---
    tags:
      - UI
    responses:
      200:
        description: HTML content of the cleaning page.
        content:
          text/html:
            schema:
              type: string
    """
    from flask import render_template
    return render_template('cleaning.html')

@analysis_bp.route('/api/analysis/start', methods=['POST'])
def start_analysis_endpoint():
    """
    Start the music analysis process for recent albums.
    This endpoint enqueues a main analysis task.
    Note: Starting a new analysis task will archive previously successful tasks by setting their status to REVOKED.
    ---
    tags:
      - Analysis
    requestBody:
      description: Configuration for the analysis task.
      required: false
      content:
        application/json:
          schema:
            type: object
            properties:
              num_recent_albums:
                type: integer
                description: Number of recent albums to process.
                default: "Configured NUM_RECENT_ALBUMS"
              top_n_moods:
                type: integer
                description: Number of top moods to extract per track.
                default: "Configured TOP_N_MOODS"
    responses:
      202:
        description: Analysis task successfully enqueued.
        content:
          application/json:
            schema:
              type: object
              properties:
                task_id:
                  type: string
                  description: The ID of the enqueued main analysis task.
                task_type:
                  type: string
                  description: Type of the task (e.g., main_analysis).
                  example: main_analysis
                status:
                  type: string
                  description: The initial status of the job in the queue (e.g., queued).
      400:
        description: Invalid input.
      500:
        description: Server error during task enqueue.
    """
    # Local import to prevent circular dependency at startup
    from app import clean_up_previous_main_tasks, save_task_status, TASK_STATUS_PENDING, rq_queue_high

    data = request.json or {}
    # MODIFIED: Removed jellyfin_url, jellyfin_user_id, and jellyfin_token as they are no longer passed to the task.
    # The task now gets these details from the central config.
    num_recent_albums = int(data.get('num_recent_albums', NUM_RECENT_ALBUMS))
    top_n_moods = int(data.get('top_n_moods', TOP_N_MOODS))

    job_id = str(uuid.uuid4())

    # Clean up details of previously successful or stale tasks before starting a new one
    clean_up_previous_main_tasks()
    save_task_status(job_id, "main_analysis", TASK_STATUS_PENDING, details={"message": "Task enqueued."})

    # Enqueue task using a string path to its function.
    # MODIFIED: The arguments passed to the task are updated to match the new function signature.
    job = rq_queue_high.enqueue(
        'tasks.analysis.run_analysis_task',
        args=(num_recent_albums, top_n_moods),
        job_id=job_id,
        description="Main Music Analysis",
        retry=Retry(max=3),
        job_timeout=-1 # No timeout
    )
    return jsonify({"task_id": job.id, "task_type": "main_analysis", "status": job.get_status()}), 202

@analysis_bp.route('/api/cleaning/identify', methods=['POST'])
def start_cleaning_identification_endpoint():
    """
    Identify orphaned albums that exist in the database but not on the media server.
    This endpoint enqueues a cleaning identification task.
    ---
    tags:
      - Cleaning
    responses:
      202:
        description: Cleaning identification task successfully enqueued.
        content:
          application/json:
            schema:
              type: object
              properties:
                task_id:
                  type: string
                  description: The ID of the enqueued cleaning identification task.
                task_type:
                  type: string
                  description: Type of the task (cleaning_identify).
                  example: cleaning_identify
                status:
                  type: string
                  description: The initial status of the job in the queue (e.g., queued).
      500:
        description: Server error during task enqueue.
    """
    # Local import to prevent circular dependency at startup
    from app import clean_up_previous_main_tasks, save_task_status, TASK_STATUS_PENDING, rq_queue_high

    # Clean up any previous cleaning tasks
    clean_up_previous_main_tasks()

    job_id = str(uuid.uuid4())
    save_task_status(job_id, "cleaning_identify", TASK_STATUS_PENDING, details={"message": "Cleaning identification task enqueued."})

    # Enqueue cleaning identification task
    job = rq_queue_high.enqueue(
        'tasks.cleaning.identify_orphaned_albums_task',
        job_id=job_id,
        description="Orphaned Albums Identification",
        retry=Retry(max=2),
        job_timeout=-1 # No timeout
    )
    return jsonify({"task_id": job.id, "task_type": "cleaning_identify", "status": job.get_status()}), 202

@analysis_bp.route('/api/cleaning/delete', methods=['POST'])
def delete_orphaned_albums_endpoint():
    """
    Delete confirmed orphaned albums from the database.
    This is a synchronous operation that requires explicit confirmation.
    ---
    tags:
      - Cleaning
    requestBody:
      description: List of track IDs to delete from the database.
      required: true
      content:
        application/json:
          schema:
            type: object
            properties:
              orphaned_track_ids:
                type: array
                items:
                  type: string
                description: List of track IDs to delete from database.
                example: ["track_id_1", "track_id_2"]
              confirm:
                type: boolean
                description: Explicit confirmation flag (must be true).
                example: true
    responses:
      200:
        description: Deletion operation completed.
        content:
          application/json:
            schema:
              type: object
              properties:
                status:
                  type: string
                  description: Operation status (SUCCESS or FAILURE).
                message:
                  type: string
                  description: Summary message.
                deleted_count:
                  type: integer
                  description: Number of tracks successfully deleted.
                failed_deletions:
                  type: array
                  description: List of failed deletion attempts.
                total_requested:
                  type: integer
                  description: Total number of tracks requested for deletion.
      400:
        description: Invalid input or missing confirmation.
      500:
        description: Server error during deletion.
    """
    # Local import to prevent circular dependency
    from tasks.cleaning import delete_orphaned_albums_sync

    data = request.json or {}
    orphaned_track_ids = data.get('orphaned_track_ids', [])
    confirm = data.get('confirm', False)

    # Validation
    if not isinstance(orphaned_track_ids, list):
        return jsonify({"error": "orphaned_track_ids must be a list"}), 400
    
    if not confirm:
        return jsonify({"error": "Explicit confirmation required. Set 'confirm': true"}), 400
    
    if not orphaned_track_ids:
        return jsonify({"error": "No track IDs provided for deletion"}), 400

    try:
        # Call the synchronous deletion function
        result = delete_orphaned_albums_sync(orphaned_track_ids)
        status_code = 200 if result["status"] == "SUCCESS" else 500
        return jsonify(result), status_code
    except Exception as e:
        logger.error(f"Error during orphaned album deletion: {e}", exc_info=True)
        return jsonify({
            "status": "FAILURE",
            "message": f"Server error during deletion: {str(e)}",
            "deleted_count": 0
        }), 500
