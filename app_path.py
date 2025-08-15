# app_path.py
from flask import Blueprint, jsonify, request, render_template
import logging
from tasks.path_manager import find_path_between_songs
from config import PATH_DEFAULT_LENGTH

logger = logging.getLogger(__name__)

# Create a Blueprint for the path finding routes
path_bp = Blueprint('path_bp', __name__, template_folder='../templates')

@path_bp.route('/path', methods=['GET'])
def path_page():
    """
    Serves the frontend page for finding a path between songs.
    """
    return render_template('path.html')

@path_bp.route('/api/find_path', methods=['GET'])
def find_path_endpoint():
    """
    Finds a path of similar songs between a start and end track.
    """
    start_song_id = request.args.get('start_song_id')
    end_song_id = request.args.get('end_song_id')
    # Use the default from config if max_steps is not provided in the request
    max_steps = request.args.get('max_steps', PATH_DEFAULT_LENGTH, type=int)

    if not start_song_id or not end_song_id:
        return jsonify({"error": "Both a start and end song must be provided."}), 400

    if start_song_id == end_song_id:
        return jsonify({"error": "Start and end songs cannot be the same."}), 400

    try:
        path, total_distance = find_path_between_songs(start_song_id, end_song_id, max_steps)

        if not path:
            return jsonify({"error": f"No path found between the selected songs within {max_steps} steps."}), 404

        return jsonify({
            "path": path,
            "total_distance": total_distance
        })

    except Exception as e:
        logger.error(f"Error finding path between {start_song_id} and {end_song_id}: {e}", exc_info=True)
        return jsonify({"error": "An unexpected error occurred while finding the path."}), 500
