# tasks/path_manager.py
import heapq
import logging
from .voyager_manager import find_nearest_neighbors_by_id, get_vector_by_id
from app import get_score_data_by_ids
import numpy as np

logger = logging.getLogger(__name__)

def heuristic(v1, v2):
    """
    Calculate Euclidean distance between two vectors as the heuristic.
    """
    return np.linalg.norm(v1 - v2)

def find_path_between_songs(start_item_id, end_item_id, max_steps=10):
    """
    Finds a path of similar songs from a start to an end song using A* search.
    """
    logger.info(f"Starting A* path search from {start_item_id} to {end_item_id} with max {max_steps} steps.")

    start_vector = get_vector_by_id(start_item_id)
    end_vector = get_vector_by_id(end_item_id)

    if start_vector is None or end_vector is None:
        logger.error("Could not retrieve vectors for start or end song.")
        return None, 0

    # The priority queue will store tuples of (priority, cost, path_list)
    # priority = cost_so_far + heuristic
    open_set = [(0 + heuristic(start_vector, end_vector), 0, [start_item_id])]
    closed_set = set()

    while open_set:
        _, cost_so_far, current_path = heapq.heappop(open_set)
        current_song_id = current_path[-1]

        if current_song_id == end_item_id:
            logger.info(f"Path found with {len(current_path)} songs and total distance {cost_so_far}.")
            path_details = get_score_data_by_ids(current_path)
            details_map = {d['item_id']: d for d in path_details}
            # Ensure the path is returned in the correct order
            ordered_path_details = [details_map[song_id] for song_id in current_path]
            return ordered_path_details, cost_so_far

        if len(current_path) >= max_steps:
            continue

        if current_song_id in closed_set:
            continue
        
        closed_set.add(current_song_id)

        # Get neighbors for the current song
        # We get a few neighbors to explore paths.
        neighbors = find_nearest_neighbors_by_id(current_song_id, n=5, eliminate_duplicates=True)
        if not neighbors:
            continue
            
        current_vector = get_vector_by_id(current_song_id)
        if current_vector is None: continue


        for neighbor in neighbors:
            neighbor_id = neighbor['item_id']
            distance = neighbor['distance']

            if neighbor_id in closed_set or neighbor_id in current_path:
                continue

            neighbor_vector = get_vector_by_id(neighbor_id)
            if neighbor_vector is None: continue

            new_cost = cost_so_far + distance
            new_path = current_path + [neighbor_id]
            priority = new_cost + heuristic(neighbor_vector, end_vector)
            
            heapq.heappush(open_set, (priority, new_cost, new_path))

    logger.warning(f"A* search completed without finding a path within {max_steps} steps.")
    return None, 0
