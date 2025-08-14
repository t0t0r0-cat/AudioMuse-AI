# tasks/path_manager.py
import heapq
import logging
from collections import Counter
from .voyager_manager import find_nearest_neighbors_by_id, get_vector_by_id
import numpy as np
from config import (
    PATH_A_STAR_ITERATION_LIMIT,
    PATH_MAX_SONGS_PER_ARTIST,
    PATH_ARTIST_PENALTY,
    PATH_NEIGHBOR_COUNT
)

logger = logging.getLogger(__name__)


def heuristic(v1, v2):
    """
    Calculate Euclidean distance between two vectors as the heuristic.
    """
    if v1 is None or v2 is None:
        return float('inf')
    return np.linalg.norm(v1 - v2)

def _run_greedy_search_to_target(start_id, target_id, max_len, excluded_artists=None):
    """
    Performs a simple greedy search from a start node towards a target node.
    It will stop when it reaches the target or the max_len.
    """
    from app import get_score_data_by_ids # Local import to prevent circular dependency
    if excluded_artists is None:
        excluded_artists = set()

    path = [start_id]
    current_id = start_id
    target_vector = get_vector_by_id(target_id)
    if target_vector is None:
        logger.error(f"Cannot perform greedy search: target vector for {target_id} not found.")
        return path

    all_details = get_score_data_by_ids([start_id])
    artist_counts = Counter(d['author'] for d in all_details)

    for _ in range(max_len - 1):
        if current_id == target_id:
            break

        neighbors = find_nearest_neighbors_by_id(current_id, n=PATH_NEIGHBOR_COUNT, eliminate_duplicates=True)
        if not neighbors:
            logger.warning(f"Greedy search stuck at {current_id}, no new neighbors found.")
            break

        best_next_id = None
        min_h = float('inf')

        neighbor_details = get_score_data_by_ids([n['item_id'] for n in neighbors])
        details_map = {d['item_id']: d for d in neighbor_details}

        for neighbor in neighbors:
            neighbor_id = neighbor['item_id']
            neighbor_artist = details_map.get(neighbor_id, {}).get('author')

            if neighbor_id in path or (neighbor_artist and neighbor_artist in excluded_artists):
                continue
            
            # Penalize artists that are already frequent in the greedy path itself
            if neighbor_artist and artist_counts.get(neighbor_artist, 0) >= PATH_MAX_SONGS_PER_ARTIST:
                continue

            neighbor_vector = get_vector_by_id(neighbor_id)
            h = heuristic(neighbor_vector, target_vector)
            
            if h < min_h:
                min_h = h
                best_next_id = neighbor_id
        
        if best_next_id:
            path.append(best_next_id)
            current_id = best_next_id
            # Update artist counts for the newly added song
            best_next_artist = details_map.get(best_next_id, {}).get('author')
            if best_next_artist:
                artist_counts[best_next_artist] += 1
        else:
            logger.warning("Greedy search could not find a new, unvisited node to advance to.")
            break
    
    return path

def _create_path_from_ids(path_ids):
    """Helper to fetch details and format the final path."""
    from app import get_score_data_by_ids # Local import to prevent circular dependency
    if not path_ids:
        return []
    path_details = get_score_data_by_ids(path_ids)
    details_map = {d['item_id']: d for d in path_details}
    ordered_path_details = [details_map[song_id] for song_id in path_ids if song_id in details_map]
    return ordered_path_details

def find_path_between_songs(start_item_id, end_item_id, max_steps=10):
    """
    Finds a path of similar songs using a persistent A* search with artist diversity penalty.
    If a complete path is not found within the iteration limit, it constructs one by searching 
    from both ends and connecting the two partial paths.
    """
    from app import get_score_data_by_ids # Local import to prevent circular dependency
    logger.info(f"Starting A* path search from {start_item_id} to {end_item_id} with max {max_steps} steps.")

    start_vector = get_vector_by_id(start_item_id)
    end_vector = get_vector_by_id(end_item_id)

    if start_vector is None or end_vector is None:
        logger.error("Could not retrieve vectors for start or end song.")
        return None, 0

    open_set = [(heuristic(start_vector, end_vector), 0, [start_item_id])]
    closed_set = set()
    all_track_details = {} 
    
    closest_path_so_far = [start_item_id]
    min_heuristic_so_far = heuristic(start_vector, end_vector)
    
    iterations = 0
    while open_set and iterations < PATH_A_STAR_ITERATION_LIMIT:
        iterations += 1
        _, cost_so_far, current_path = heapq.heappop(open_set)
        current_song_id = current_path[-1]

        if len(current_path) >= max_steps:
            continue
            
        current_vector = get_vector_by_id(current_song_id)
        current_heuristic = heuristic(current_vector, end_vector)
        if current_heuristic < min_heuristic_so_far:
            min_heuristic_so_far = current_heuristic
            closest_path_so_far = current_path

        if current_song_id == end_item_id:
            logger.info(f"A* path found after {iterations} iterations with {len(current_path)} songs.")
            return _create_path_from_ids(current_path), cost_so_far

        if current_song_id in closed_set:
            continue
        closed_set.add(current_song_id)

        neighbors = find_nearest_neighbors_by_id(current_song_id, n=PATH_NEIGHBOR_COUNT, eliminate_duplicates=True)
        if not neighbors:
            continue
            
        neighbor_ids = [n['item_id'] for n in neighbors]
        new_details = get_score_data_by_ids(neighbor_ids)
        for detail in new_details: all_track_details[detail['item_id']] = detail
        
        path_details = get_score_data_by_ids(current_path)
        for detail in path_details: all_track_details[detail['item_id']] = detail
            
        artist_counts = Counter(all_track_details[song_id]['author'] for song_id in current_path if song_id in all_track_details)

        for neighbor in neighbors:
            neighbor_id = neighbor['item_id']
            if neighbor_id in closed_set or neighbor_id in current_path:
                continue

            neighbor_vector = get_vector_by_id(neighbor_id)
            if neighbor_vector is None: continue

            distance = neighbor['distance']
            new_cost = cost_so_far + distance

            neighbor_artist = all_track_details.get(neighbor_id, {}).get('author')
            if neighbor_artist and artist_counts.get(neighbor_artist, 0) >= PATH_MAX_SONGS_PER_ARTIST:
                new_cost += PATH_ARTIST_PENALTY

            new_path = current_path + [neighbor_id]
            priority = new_cost + heuristic(neighbor_vector, end_vector)
            heapq.heappush(open_set, (priority, new_cost, new_path))

    # --- Fallback: Bidirectional Greedy Search ---
    logger.warning(f"A* did not find a complete path within {PATH_A_STAR_ITERATION_LIMIT} iterations. Initiating fallback.")
    
    path_a_to_b = closest_path_so_far
    
    path_a_artists = {d['author'] for d in get_score_data_by_ids(path_a_to_b)}
    
    connection_target = path_a_to_b[-1]
    remaining_steps = max_steps - len(path_a_to_b)

    if remaining_steps > 1:
        logger.info(f"Starting reverse greedy search from '{end_item_id}' towards '{connection_target}' for {remaining_steps} steps.")
        path_c_to_b = _run_greedy_search_to_target(end_item_id, connection_target, remaining_steps, excluded_artists=path_a_artists)
        path_c_to_b.reverse()
        final_path_ids = path_a_to_b + path_c_to_b[1:]
    else:
        logger.info("Not enough remaining steps for reverse search. Appending end song.")
        final_path_ids = path_a_to_b + [end_item_id]
    
    # Ensure the final path is clean and respects the length limit
    unique_path = []
    for item_id in final_path_ids:
        if item_id not in unique_path:
            unique_path.append(item_id)
            
    if len(unique_path) > max_steps:
        unique_path = unique_path[:max_steps - 1] + [end_item_id]

    return _create_path_from_ids(unique_path), 0
