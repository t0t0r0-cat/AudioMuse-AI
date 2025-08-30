# app_collection.py
from flask import Blueprint, jsonify, request, render_template
import uuid
import logging

from rq import Retry

logger = logging.getLogger(__name__)

collection_bp = Blueprint('collection_bp', __name__)

@collection_bp.route('/collection')
def collection_page():
    """Serves the HTML page for the Collection Sync feature."""
    return render_template('collection.html')

@collection_bp.route('/api/collection/start', methods=['POST'])
def start_collection_sync():
    """
    Starts the process of synchronizing local song data with a remote PocketBase collection.
    This enqueues the main parent task for the synchronization.
    """
    # Local import to avoid circular dependency
    from app import save_task_status, TASK_STATUS_PENDING, rq_queue_high

    data = request.json
    if not data or not all(k in data for k in ['url', 'email', 'password', 'num_albums']):
        return jsonify({"message": "Missing required parameters: url, email, password, num_albums"}), 400

    pocketbase_url = data['url']
    user_email = data['email']
    password = data['password']
    num_last_albums = int(data['num_albums'])
    
    job_id = str(uuid.uuid4())
    
    # Save the initial "PENDING" state for the main task
    save_task_status(
        job_id,
        "main_collection_sync",
        TASK_STATUS_PENDING,
        details={"message": "Synchronization task has been enqueued."}
    )

    # Enqueue the main parent task to the high priority queue
    job = rq_queue_high.enqueue(
        'tasks.collection_manager.sync_collections_task',
        args=(pocketbase_url, user_email, password, num_last_albums),
        job_id=job_id,
        description="Main Collection Synchronization",
        retry=Retry(max=2),
        job_timeout='2h' # Set a reasonable timeout for the parent task
    )

    return jsonify({
        "task_id": job.id,
        "task_type": "main_collection_sync",
        "status": job.get_status()
    }), 202
