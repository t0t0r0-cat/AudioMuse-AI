# tasks/path_manager.py
import logging
from .voyager_manager import get_vector_by_id, find_nearest_neighbors_by_vector
import numpy as np
from config import PATH_CENTROID_PERCENTAGE, PATH_NEIGHBOR_PERCENTAGE

logger = logging.getLogger(__name__)


def get_euclidean_distance(v1, v2):
    """Calculates the Euclidean distance between two vectors."""
    if v1 is not None and v2 is not None:
        return np.linalg.norm(v1 - v2)
    return float('inf')


def _create_path_from_ids(path_ids):
    """Helper to fetch song details for a list of IDs and format the final path."""
    from app import get_score_data_by_ids
    if not path_ids:
        return []
    
    seen = set()
    unique_path_ids = [x for x in path_ids if not (x in seen or seen.add(x))]

    path_details = get_score_data_by_ids(unique_path_ids)
    details_map = {d['item_id']: d for d in path_details}
    
    ordered_path_details = [details_map[song_id] for song_id in unique_path_ids if song_id in details_map]
    return ordered_path_details


def find_path_between_songs(start_item_id, end_item_id, max_steps=100):
    """
    Finds a path between two songs using a configurable centroid-based approach.
    This method ensures path quality by selecting the best unique song for each centroid in order.
    """
    logger.info(f"Starting configurable centroid path generation from {start_item_id} to {end_item_id} with exactly {max_steps} steps.")

    start_vector = get_vector_by_id(start_item_id)
    end_vector = get_vector_by_id(end_item_id)

    if start_vector is None or end_vector is None:
        logger.error("Could not retrieve vectors for start or end song.")
        return None, 0.0

    if max_steps < 2:
        return _create_path_from_ids([start_item_id, end_item_id]), 0.0

    # --- Configurable Centroid and Neighbor Calculation ---
    num_intermediate_songs_needed = max_steps - 2
    if num_intermediate_songs_needed <= 0:
        return _create_path_from_ids([start_item_id, end_item_id]), 0.0

    num_centroids = max(1, int(num_intermediate_songs_needed * (PATH_CENTROID_PERCENTAGE / 100.0)))
    num_neighbors_per_centroid = max(1, int(max_steps * (PATH_NEIGHBOR_PERCENTAGE / 100.0)))

    logger.info(f"Using {num_centroids} centroids, searching for {num_neighbors_per_centroid} neighbors each.")

    # --- Step 1: Find neighbors for each centroid ---
    centroid_vectors = np.linspace(start_vector, end_vector, num=num_centroids + 2)[1:-1]
    centroid_neighbors = []
    songs_to_exclude = {start_item_id, end_item_id}

    for i, vector in enumerate(centroid_vectors):
        try:
            neighbors = find_nearest_neighbors_by_vector(vector, n=num_neighbors_per_centroid)
            # For each centroid, store its list of valid neighbors
            valid_neighbors = [n for n in neighbors if n['item_id'] not in songs_to_exclude]
            centroid_neighbors.append(valid_neighbors)
        except Exception as e:
            logger.error(f"Error finding neighbors for centroid {i}: {e}")
            centroid_neighbors.append([]) # Append empty list on error to maintain index alignment

    # --- Step 2: Assign the best unique song to each centroid IN ORDER ---
    intermediate_path = []
    used_song_ids = set()

    # Iterate through each centroid step from start to finish
    for i in range(num_centroids):
        # Ensure we have a neighbor list for this centroid
        if i >= len(centroid_neighbors):
            logger.warning(f"Missing neighbor data for centroid {i}, cannot fill this step.")
            continue

        best_choice_for_this_centroid = None
        # Find the best (closest) unique song from this specific centroid's neighbor list
        for neighbor in centroid_neighbors[i]:
            if neighbor['item_id'] not in used_song_ids:
                best_choice_for_this_centroid = neighbor['item_id']
                break  # Found the best unique song for this step
        
        if best_choice_for_this_centroid:
            intermediate_path.append(best_choice_for_this_centroid)
            used_song_ids.add(best_choice_for_this_centroid)
        else:
            # If no unique song is found for this specific centroid, we leave the slot empty.
            # This is better than picking a random, out-of-place song.
            logger.warning(f"Could not find a unique neighbor for centroid {i}. The path may be shorter than requested, but will maintain quality.")

    # --- Step 3: Build the Final Path ---
    # The number of songs will be correct up to the point where unique neighbors ran out.
    final_path_ids = [start_item_id] + intermediate_path + [end_item_id]

    return _create_path_from_ids(final_path_ids), 0.0
