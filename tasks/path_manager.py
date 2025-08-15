# tasks/path_manager.py
import logging
import numpy as np
import random
import psycopg2 # Import psycopg2 to catch specific errors

# Imports from the project
from .voyager_manager import get_vector_by_id, find_nearest_neighbors_by_vector, find_nearest_neighbors_by_id
from app import get_db, get_score_data_by_ids
from config import PATH_AVG_JUMP_SAMPLE_SIZE, PATH_CANDIDATES_PER_STEP

logger = logging.getLogger(__name__)


def get_euclidean_distance(v1, v2):
    """Calculates the Euclidean distance between two vectors."""
    if v1 is not None and v2 is not None:
        return np.linalg.norm(v1 - v2)
    return float('inf')


def _create_path_from_ids(path_ids):
    """Helper to fetch song details for a list of IDs and format the final path."""
    if not path_ids:
        return []
    
    seen = set()
    unique_path_ids = [x for x in path_ids if not (x in seen or seen.add(x))]

    path_details = get_score_data_by_ids(unique_path_ids)
    details_map = {d['item_id']: d for d in path_details}
    
    ordered_path_details = [details_map[song_id] for song_id in unique_path_ids if song_id in details_map]
    return ordered_path_details


def _calculate_local_average_jump_distance(start_item_id, end_item_id, sample_size=PATH_AVG_JUMP_SAMPLE_SIZE):
    """
    Calculates the average Euclidean distance based on the direct neighbors of the
    start and end songs, without querying the broader database.
    """
    logger.info(f"Calculating local average jump distance based on neighbors of {start_item_id} and {end_item_id}.")
    
    distances = []
    
    # Process neighbors for both the start and end points
    for item_id in [start_item_id, end_item_id]:
        try:
            # Find a sample of neighbors using Voyager
            neighbors = find_nearest_neighbors_by_id(item_id, n=sample_size)
            if not neighbors:
                continue

            # The distance for these neighbors is already provided by the Voyager query result
            for neighbor in neighbors:
                distances.append(neighbor['distance'])

        except Exception as e:
            logger.warning(f"Could not process neighbors for song {item_id} during local jump calculation: {e}")

    if not distances:
        logger.error("No valid neighbor distances could be calculated from start/end songs.")
        return 0.1 # Return a sensible fallback default

    avg_dist = np.mean(distances)
    logger.info(f"Calculated local average jump distance: {avg_dist:.4f} from {len(distances)} neighbors.")
    return avg_dist


def find_path_between_songs(start_item_id, end_item_id, Lreq=100):
    """
    Finds an adaptive path between two songs using an approach based on embedding space density.
    The number of steps is adapted to the distance between songs to create a more natural path.
    """
    logger.info(f"Starting adaptive path generation from {start_item_id} to {end_item_id} with requested length {Lreq}.")

    # --- Step 1: Get vectors and calculate local jump distance ---
    start_vector = get_vector_by_id(start_item_id)
    end_vector = get_vector_by_id(end_item_id)

    if start_vector is None or end_vector is None:
        logger.error("Could not retrieve vectors for start or end song.")
        return None, 0.0

    # Calculate the average jump distance based on the local neighborhood of the start and end songs.
    # This is done for every request and is NOT cached globally.
    delta_avg = _calculate_local_average_jump_distance(start_item_id, end_item_id)
    
    if delta_avg is None or delta_avg == 0:
        logger.error("Average jump distance (delta_avg) is not available or zero, cannot generate path.")
        return None, 0.0

    # --- Step 2: Prepare the path request ---
    D = get_euclidean_distance(start_vector, end_vector)
    # Estimate max realistic steps. +1 to ensure we have at least start and end.
    Lmax = int(np.floor(D / delta_avg)) + 1
    
    # Set backbone length (Lcore) by taking the minimum of user request and realistic max.
    # Ensure it's at least 2 (for start and end songs).
    Lcore = min(Lreq, Lmax)
    Lcore = max(2, Lreq) if Lreq < Lmax else Lcore
    Lcore = max(2, Lcore)


    logger.info(f"Direct distance (D): {D:.4f}, Local avg jump (δ_avg): {delta_avg:.4f}")
    logger.info(f"Requested steps (L_req): {Lreq}, Max realistic (L_max): {Lmax}, Using core steps (L_core): {Lcore}")
    
    if Lcore < 2:
        return _create_path_from_ids([start_item_id, end_item_id]), D

    # --- Step 3: Generate backbone centroids ---
    # These are the ideal, evenly-spaced points in the embedding space.
    backbone_centroids = np.linspace(start_vector, end_vector, num=Lcore)

    # --- Step 4: Map centroids to real songs (Backbone Path) ---
    path_ids = [start_item_id]
    used_song_ids = {start_item_id, end_item_id} # Exclude start and end from intermediate steps

    # Iterate through the intermediate centroids to find the songs in between.
    for i in range(1, Lcore - 1):
        centroid_vec = backbone_centroids[i]
        
        try:
            # Find candidate songs near the ideal centroid position
            candidates = find_nearest_neighbors_by_vector(centroid_vec, n=PATH_CANDIDATES_PER_STEP)
        except Exception as e:
            logger.error(f"Error finding neighbors for centroid {i}: {e}")
            continue

        best_song_id = None
        min_dist_to_centroid = float('inf')

        # Find the best *available* candidate (closest to the ideal point)
        for candidate in candidates:
            candidate_id = candidate['item_id']
            if candidate_id not in used_song_ids:
                candidate_vector = get_vector_by_id(candidate_id)
                if candidate_vector is not None:
                    dist = get_euclidean_distance(candidate_vector, centroid_vec)
                    if dist < min_dist_to_centroid:
                        min_dist_to_centroid = dist
                        best_song_id = candidate_id
        
        if best_song_id:
            path_ids.append(best_song_id)
            used_song_ids.add(best_song_id)
        else:
            logger.warning(f"Could not find a unique song for step {i+1}. The path may be shorter than {Lcore}.")

    path_ids.append(end_item_id)

    # --- Step 5: Fill extra steps if Lreq > current path length ---
    if Lreq > len(path_ids):
        logger.info(f"Requested path length ({Lreq}) is longer than the natural path ({len(path_ids)}). Filling in extra songs.")
        
        final_path_ids = []
        songs_to_add = Lreq - len(path_ids)
        num_segments = len(path_ids) - 1

        if num_segments > 0:
            base_songs_per_segment = songs_to_add // num_segments
            remainder = songs_to_add % num_segments
        else:
            base_songs_per_segment = 0
            remainder = 0

        # Build the final path by inserting songs into the segments
        for i in range(num_segments):
            start_segment_id = path_ids[i]
            end_segment_id = path_ids[i+1]
            
            final_path_ids.append(start_segment_id)

            num_to_insert = base_songs_per_segment + (1 if i < remainder else 0)

            if num_to_insert > 0:
                start_segment_vec = get_vector_by_id(start_segment_id)
                end_segment_vec = get_vector_by_id(end_segment_id)

                if start_segment_vec is not None and end_segment_vec is not None:
                    intermediate_centroids = np.linspace(start_segment_vec, end_segment_vec, num=num_to_insert + 2)[1:-1]
                    
                    for centroid_vec in intermediate_centroids:
                        candidates = find_nearest_neighbors_by_vector(centroid_vec, n=PATH_CANDIDATES_PER_STEP * 2)
                        
                        best_song_id = None
                        min_dist_to_centroid = float('inf')
                        
                        for candidate in candidates:
                            candidate_id = candidate['item_id']
                            if candidate_id not in used_song_ids:
                                candidate_vector = get_vector_by_id(candidate_id)
                                if candidate_vector is not None:
                                    dist = get_euclidean_distance(candidate_vector, centroid_vec)
                                    if dist < min_dist_to_centroid:
                                        min_dist_to_centroid = dist
                                        best_song_id = candidate_id
                        
                        if best_song_id:
                            final_path_ids.append(best_song_id)
                            used_song_ids.add(best_song_id)
                        else:
                            logger.warning(f"Could not find a unique filler song for segment {i}. Path may be shorter than requested.")
        
        final_path_ids.append(path_ids[-1])
        path_ids = final_path_ids

    # --- Step 6: Output path and calculate total distance ---
    final_path_details = _create_path_from_ids(path_ids)
    
    # Calculate the total "jump distance" of the final path by summing segment distances
    total_path_distance = 0.0
    if len(final_path_details) > 1:
        path_vectors = [get_vector_by_id(song['item_id']) for song in final_path_details]
        for i in range(len(path_vectors) - 1):
            v1 = path_vectors[i]
            v2 = path_vectors[i+1]
            if v1 is not None and v2 is not None:
                total_path_distance += get_euclidean_distance(v1, v2)

    return final_path_details, total_path_distance
