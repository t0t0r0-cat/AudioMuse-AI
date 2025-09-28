# app_universe.py - Song Universe Map Blueprint

from flask import Blueprint, render_template, jsonify, request
from psycopg2.extras import DictCursor
import random
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Create blueprint
universe_bp = Blueprint('universe', __name__)

@universe_bp.route('/universe')
def universe():
    """
    Serve the Song Universe Map page.
    ---
    tags:
      - UI
    responses:
      200:
        description: HTML content of the Song Universe Map page.
        content:
          text/html:
            schema:
              type: string
    """
    return render_template('universe.html')

@universe_bp.route('/alternative_universe')
def alternative_universe():
    """
    Serve the Alternative Song Universe Map page (Graph View).
    ---
    tags:
      - UI
    responses:
      200:
        description: HTML content of the Alternative Song Universe Map page.
        content:
          text/html:
            schema:
              type: string
    """
    return render_template('alternative_universe.html')

@universe_bp.route('/music_map')
def music_map():
    """
    Serve the Music Map Universe page (t-SNE positioned songs).
    ---
    tags:
      - UI
    responses:
      200:
        description: HTML content of the Music Map Universe page.
        content:
          text/html:
            schema:
              type: string
    """
    return render_template('music_map.html')

@universe_bp.route('/api/universe/genre_songs', methods=['GET'])
def get_genre_songs_endpoint():
    """
    Get sample songs for a specific genre from mood_vector.
    ---
    tags:
      - Universe
    parameters:
      - name: genre
        in: query
        required: true
        description: The genre to search for.
        schema:
          type: string
      - name: limit
        in: query
        description: Maximum number of songs to return.
        schema:
          type: integer
          default: 15
    responses:
      200:
        description: A list of songs in the specified genre.
        content:
          application/json:
            schema:
              type: array
              items:
                type: object
                properties:
                  item_id:
                    type: string
                  title:
                    type: string
                  author:
                    type: string
                  mood_vector:
                    type: string
    """
    from app import get_db
    
    genre = request.args.get('genre', '')
    limit = request.args.get('limit', 15, type=int)
    
    if not genre:
        return jsonify({"error": "Genre parameter is required"}), 400
    
    try:
        db = get_db()
        cur = db.cursor(cursor_factory=DictCursor)
        
        # Search for songs that have this genre in their mood_vector
        # mood_vector format is like "rock:0.8,pop:0.3,jazz:0.1"
        query = """
            SELECT item_id, title, author, mood_vector
            FROM score 
            WHERE mood_vector LIKE %s
            ORDER BY random()
            LIMIT %s
        """
        
        search_pattern = f"%{genre}:%"
        cur.execute(query, (search_pattern, limit))
        results = cur.fetchall()
        cur.close()
        
        return jsonify([dict(row) for row in results])
    
    except Exception as e:
        logger.error(f"Error fetching genre songs for {genre}: {e}")
        return jsonify({"error": "Internal server error"}), 500

@universe_bp.route('/api/universe/graph_data', methods=['GET'])
def get_graph_data_endpoint():
    """
    Get graph data for visualizing the voyager index as a connected network.
    Uses the in-memory voyager index directly.
    """
    from tasks.voyager_manager import voyager_index, id_map, find_nearest_neighbors_by_id
    from app import get_db
    
    limit = request.args.get('limit', 1000, type=int)
    offset = request.args.get('offset', 0, type=int)
    
    try:
        # Check if voyager index is loaded
        if voyager_index is None or id_map is None:
            logger.warning("Voyager index not loaded in memory")
            return jsonify({"error": "Voyager index not available"}), 503
        
        logger.info(f"Voyager index loaded with {len(id_map)} items")
        
        # Get database connection for song metadata
        db = get_db()
        cur = db.cursor(cursor_factory=DictCursor)
        
        # Get a sample of songs from the voyager index
        total_songs = len(id_map)
        available_item_ids = list(id_map.values())
        
        # Apply pagination
        start_idx = min(offset, len(available_item_ids))
        end_idx = min(offset + limit, len(available_item_ids))
        sampled_item_ids = available_item_ids[start_idx:end_idx]
        
        if not sampled_item_ids:
            return jsonify({"nodes": [], "edges": []})
            
        logger.info(f"Processing {len(sampled_item_ids)} songs for graph")
        
        # Get metadata for these songs
        placeholders = ','.join(['%s'] * len(sampled_item_ids))
        metadata_query = f"""
            SELECT s.item_id, s.title, s.author, s.mood_vector
            FROM score s
            WHERE s.item_id IN ({placeholders})
              AND s.title IS NOT NULL 
              AND s.author IS NOT NULL
        """
        
        cur.execute(metadata_query, sampled_item_ids)
        metadata_results = cur.fetchall()
        cur.close()
        
        # Create nodes with metadata
        nodes = []
        valid_item_ids = set()
        
        genre_colors = {
            'rock': '#ff6b6b', 'pop': '#4ecdc4', 'jazz': '#ffe66d', 
            'classical': '#a8e6cf', 'electronic': '#ff8b94', 'hip-hop': '#ffaaa5',
            'country': '#88d8b0', 'r&b': '#ffd93d', 'blues': '#6c5ce7',
            'folk': '#fdcb6e', 'metal': '#636e72', 'alternative': '#00b894'
        }
        
        for row in metadata_results:
            try:
                # Extract primary genre from mood_vector
                genre = 'unknown'
                genre_color = '#ffeb3b'  # default yellow
                
                if row['mood_vector']:
                    genres = []
                    for part in row['mood_vector'].split(','):
                        if ':' in part:
                            g, score = part.split(':', 1)
                            try:
                                genres.append((g.strip(), float(score)))
                            except:
                                continue
                    
                    if genres:
                        genres.sort(key=lambda x: x[1], reverse=True)
                        genre = genres[0][0]
                        genre_color = genre_colors.get(genre, '#ffeb3b')
                
                node = {
                    'item_id': str(row['item_id']),
                    'title': str(row['title']),
                    'author': str(row['author']),
                    'genre': genre,
                    'genre_color': genre_color,
                    'connections': 0  # Will be calculated from edges
                }
                nodes.append(node)
                valid_item_ids.add(str(row['item_id']))
                
            except Exception as e:
                logger.warning(f"Error processing node {row.get('item_id', 'unknown')}: {e}")
                continue
        
        if not nodes:
            return jsonify({"nodes": [], "edges": []})
        
        logger.info(f"Created {len(nodes)} valid nodes")
        
        # Generate edges using voyager index
        edges = []
        connection_counts = {}
        
        # For each node, find its nearest neighbors
        max_connections_per_node = min(10, len(nodes) // 10)  # Limit connections
        
        for i, node in enumerate(nodes[:min(100, len(nodes))]):  # Process max 100 nodes for edges
            try:
                item_id = node['item_id']
                
                # Get neighbors from voyager
                neighbors = find_nearest_neighbors_by_id(item_id, n=max_connections_per_node)
                
                if neighbors:
                    for neighbor in neighbors:
                        neighbor_id = str(neighbor['item_id'])
                        distance = float(neighbor['distance'])
                        
                        # Only add edge if both nodes are in our current set
                        if neighbor_id in valid_item_ids and neighbor_id != item_id:
                            # Avoid duplicate edges
                            edge_key = tuple(sorted([item_id, neighbor_id]))
                            
                            edge = {
                                'source': item_id,
                                'target': neighbor_id,
                                'distance': distance
                            }
                            edges.append(edge)
                            
                            # Count connections
                            connection_counts[item_id] = connection_counts.get(item_id, 0) + 1
                            connection_counts[neighbor_id] = connection_counts.get(neighbor_id, 0) + 1
                            
            except Exception as e:
                logger.warning(f"Error processing edges for node {node['item_id']}: {e}")
                continue
        
        # Update connection counts in nodes
        for node in nodes:
            node['connections'] = connection_counts.get(node['item_id'], 0)
        
        logger.info(f"Generated {len(edges)} edges between {len(nodes)} nodes")
        
        return jsonify({
            "nodes": nodes,
            "edges": edges
        })
        
    except Exception as e:
        logger.error(f"Error fetching graph data: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@universe_bp.route('/api/universe/graph_data_centered', methods=['GET'])
def get_centered_graph_data_endpoint():
    """
    Get graph data centered on a specific song, showing the song and its neighbors.
    Uses the in-memory voyager index directly.
    """
    from tasks.voyager_manager import voyager_index, id_map, find_nearest_neighbors_by_id
    from app import get_db
    
    item_id = request.args.get('item_id', '')
    limit = request.args.get('limit', 100, type=int)
    
    if not item_id:
        return jsonify({"error": "item_id parameter is required"}), 400
    
    try:
        # Check if voyager index is loaded
        if voyager_index is None or id_map is None:
            logger.warning("Voyager index not loaded in memory")
            return jsonify({"error": "Voyager index not available"}), 503
        
        logger.info(f"Creating centered graph for song: {item_id}")
        
        # Get database connection for song metadata
        db = get_db()
        cur = db.cursor(cursor_factory=DictCursor)
        
        # First, get the center song and its neighbors from voyager
        try:
            neighbors = find_nearest_neighbors_by_id(item_id, n=limit-1)  # -1 for the center song itself
            logger.info(f"Found {len(neighbors)} neighbors for center song {item_id}")
        except Exception as e:
            logger.error(f"Error finding neighbors: {e}")
            neighbors = []
        
        # Collect all item IDs (center + neighbors)
        all_item_ids = [item_id]
        neighbor_distances = {item_id: 0.0}  # Center song has distance 0
        
        for neighbor in neighbors:
            neighbor_id = str(neighbor['item_id'])
            all_item_ids.append(neighbor_id)
            neighbor_distances[neighbor_id] = float(neighbor['distance'])
        
        if not all_item_ids:
            return jsonify({"nodes": [], "edges": []})
            
        logger.info(f"Getting metadata for {len(all_item_ids)} songs")
        
        # Get metadata for all songs
        placeholders = ','.join(['%s'] * len(all_item_ids))
        metadata_query = f"""
            SELECT s.item_id, s.title, s.author, s.mood_vector
            FROM score s
            WHERE s.item_id IN ({placeholders})
        """
        
        cur.execute(metadata_query, all_item_ids)
        metadata_results = cur.fetchall()
        cur.close()
        
        # Create nodes with metadata
        nodes = []
        valid_item_ids = set()
        
        genre_colors = {
            'rock': '#ff6b6b', 'pop': '#4ecdc4', 'jazz': '#ffe66d', 
            'classical': '#a8e6cf', 'electronic': '#ff8b94', 'hip-hop': '#ffaaa5',
            'country': '#88d8b0', 'r&b': '#ffd93d', 'blues': '#6c5ce7',
            'folk': '#fdcb6e', 'metal': '#636e72', 'alternative': '#00b894'
        }
        
        center_song = None
        
        for row in metadata_results:
            try:
                song_id = str(row['item_id'])
                
                # Extract primary genre from mood_vector
                genre = 'unknown'
                genre_color = '#ffeb3b'  # default yellow
                
                if row['mood_vector']:
                    genres = []
                    for part in row['mood_vector'].split(','):
                        if ':' in part:
                            g, score = part.split(':', 1)
                            try:
                                genres.append((g.strip(), float(score)))
                            except:
                                continue
                    
                    if genres:
                        genres.sort(key=lambda x: x[1], reverse=True)
                        genre = genres[0][0]
                        genre_color = genre_colors.get(genre, '#ffeb3b')
                
                # Special color for center song
                if song_id == item_id:
                    genre_color = '#ff4444'  # Red for center song
                
                node = {
                    'item_id': song_id,
                    'title': str(row['title']) if row['title'] else 'Unknown',
                    'author': str(row['author']) if row['author'] else 'Unknown',
                    'genre': genre,
                    'genre_color': genre_color,
                    'connections': 0,  # Will be calculated from edges
                    'distance_from_center': neighbor_distances.get(song_id, 0.0),
                    'is_center': song_id == item_id
                }
                nodes.append(node)
                valid_item_ids.add(song_id)
                
                if song_id == item_id:
                    center_song = node
                
            except Exception as e:
                logger.warning(f"Error processing node {row.get('item_id', 'unknown')}: {e}")
                continue
        
        if not nodes:
            return jsonify({"error": "No metadata found for the songs"}), 404
        
        if not center_song:
            logger.warning(f"Center song {item_id} not found in results")
        
        logger.info(f"Created {len(nodes)} nodes with center song")
        
        # Create edges - connect center to all neighbors, and neighbors to each other
        edges = []
        
        # Connect center song to all its neighbors
        for node in nodes:
            if node['item_id'] != item_id and not node['is_center']:
                edges.append({
                    'source': item_id,
                    'target': node['item_id'],
                    'distance': node['distance_from_center']
                })
        
        # Find connections between ALL songs (not just neighbors)
        # This creates a more interconnected network
        processed_pairs = set()
        
        for i, node1 in enumerate(nodes):
            try:
                # Find neighbors of this song
                if i < 30:  # Process first 30 songs to avoid too many API calls
                    node_neighbors = find_nearest_neighbors_by_id(node1['item_id'], n=min(15, len(nodes)))
                    for neighbor in node_neighbors:
                        target_id = str(neighbor['item_id'])
                        if target_id in valid_item_ids and target_id != node1['item_id']:
                            # Create a unique pair identifier
                            pair = tuple(sorted([node1['item_id'], target_id]))
                            if pair not in processed_pairs:
                                processed_pairs.add(pair)
                                edges.append({
                                    'source': node1['item_id'],
                                    'target': target_id,
                                    'distance': float(neighbor['distance'])
                                })
            except Exception as e:
                logger.warning(f"Error finding neighbors for {node1['item_id']}: {e}")
                continue
        
        # Remove duplicate edges and sort by distance
        seen_edges = set()
        unique_edges = []
        for edge in sorted(edges, key=lambda x: x['distance']):
            edge_key = tuple(sorted([edge['source'], edge['target']]))
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                unique_edges.append(edge)
        
        # Limit total edges to prevent overcrowding (keep the closest connections)
        edges = unique_edges[:min(200, len(unique_edges))]
        
        # Update connection counts
        connection_counts = {}
        for edge in edges:
            connection_counts[edge['source']] = connection_counts.get(edge['source'], 0) + 1
            connection_counts[edge['target']] = connection_counts.get(edge['target'], 0) + 1
        
        for node in nodes:
            node['connections'] = connection_counts.get(node['item_id'], 0)
        
        logger.info(f"Generated {len(edges)} edges for centered graph")
        
        return jsonify({
            "nodes": nodes,
            "edges": edges,
            "center_song": center_song
        })
        
    except Exception as e:
        logger.error(f"Error fetching centered graph data: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@universe_bp.route('/api/universe/music_map_data', methods=['GET'])
def get_music_map_data_endpoint():
    """
    Get all songs positioned using t-SNE dimensionality reduction of their embeddings.
    This creates a 2D map where similar songs are positioned close to each other.
    """
    from app import get_db
    import numpy as np  # Import numpy at the beginning of the function
    from config import STRATIFIED_GENRES  # Import stratified genres from config

    limit = request.args.get('limit', 100, type=int) 
    offset = request.args.get('offset', 0, type=int)
    
    try:
        # Get database connection
        db = get_db()
        cur = db.cursor(cursor_factory=DictCursor)
        
        # FIRST: Get the Voyager index order to query songs in the correct sequence
        try:
            from tasks.voyager_manager import voyager_index, id_map
            
            if voyager_index is not None and id_map is not None:
                logger.info(f"Using Voyager index with {len(id_map)} items for ordered query")
                
                # Get songs in Voyager index order (not random!)
                # Sort by voyager index position to get sequential similarity
                voyager_item_ids = []
                for voyager_idx in sorted(id_map.keys()):  # Sort by index position
                    voyager_item_ids.append(id_map[voyager_idx])
                
                # Apply limit to the ordered sequence
                limited_item_ids = voyager_item_ids[offset:offset + limit]
                
                if not limited_item_ids:
                    return jsonify({"songs": [], "total": 0})
                
                # Query songs in the exact Voyager index order
                placeholders = ','.join(['%s'] * len(limited_item_ids))
                query = f"""
                    SELECT s.item_id, s.title, s.author, s.mood_vector, e.embedding
                    FROM score s
                    LEFT JOIN embedding e ON s.item_id = e.item_id
                    WHERE s.item_id IN ({placeholders})
                      AND s.title IS NOT NULL 
                      AND s.author IS NOT NULL
                """
                
                cur.execute(query, limited_item_ids)
                results = cur.fetchall()
                
                # Create a mapping for ordering results by Voyager index
                result_map = {str(row['item_id']): row for row in results}
                
                # Reorder results to match the exact Voyager sequence
                ordered_results = []
                for item_id in limited_item_ids:
                    if str(item_id) in result_map:
                        ordered_results.append(result_map[str(item_id)])
                
                results = ordered_results
                logger.info(f"Loaded {len(results)} songs in exact Voyager index order")
                
            else:
                # Fallback to random if Voyager not available
                logger.warning("Voyager index not available, using random query")
                query = """
                    SELECT s.item_id, s.title, s.author, s.mood_vector, e.embedding
                    FROM score s
                    LEFT JOIN embedding e ON s.item_id = e.item_id
                    WHERE s.title IS NOT NULL 
                      AND s.author IS NOT NULL
                    ORDER BY random()
                    LIMIT %s OFFSET %s
                """
                cur.execute(query, (limit, offset))
                results = cur.fetchall()
                
        except Exception as e:
            logger.error(f"Error accessing Voyager index: {e}, using random query")
            query = """
                SELECT s.item_id, s.title, s.author, s.mood_vector, e.embedding
                FROM score s
                LEFT JOIN embedding e ON s.item_id = e.item_id
                WHERE s.title IS NOT NULL 
                  AND s.author IS NOT NULL
                ORDER BY random()
                LIMIT %s OFFSET %s
            """
            cur.execute(query, (limit, offset))
            results = cur.fetchall()
        
        cur.close()
        
        if not results:
            return jsonify({"songs": [], "total": 0})
        
        logger.info(f"Loaded {len(results)} songs for music map")
        
        # Extract embeddings and metadata
        embeddings = []
        songs = []
        
        # Create very different colors for all stratified genres
        stratified_colors = [
            '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF',  # Primary colors
            '#800000', '#008000', '#000080', '#808000', '#800080', '#008080',  # Dark colors
            '#FF8000', '#8000FF', '#FF0080', '#80FF00', '#0080FF', '#FF8080',  # Mixed colors
            '#80FF80', '#8080FF', '#FFFF80', '#FF80FF', '#80FFFF', '#C0C0C0',  # Light colors
            '#FFA500', '#DC143C', '#32CD32', '#B22222', '#DAA520', '#4682B4'   # Named colors
        ]
        
        # Map stratified genres to colors
        genre_colors = {}
        for i, genre in enumerate(STRATIFIED_GENRES):
            if i < len(stratified_colors):
                genre_colors[genre] = stratified_colors[i]
            else:
                # Fallback for additional genres (generate more colors)
                genre_colors[genre] = f"#{hex(hash(genre) % 16777215)[2:].zfill(6).upper()}"
        
        # Add fallback color for unknown genres
        genre_colors['unknown'] = '#A9A9A9'  # Dark Gray
        
        for row in results:
            try:
                # Add embedding if available
                if row['embedding'] is not None:
                    embedding = np.frombuffer(row['embedding'], dtype=np.float32)
                    embeddings.append(embedding)
                
                # Extract primary genre
                genre = 'unknown'
                genre_color = '#ffeb3b'
                
                if row['mood_vector']:
                    genres = []
                    for part in row['mood_vector'].split(','):
                        if ':' in part:
                            g, score = part.split(':', 1)
                            try:
                                genres.append((g.strip(), float(score)))
                            except:
                                continue
                    
                    if genres:
                        genres.sort(key=lambda x: x[1], reverse=True)
                        genre = genres[0][0]
                        genre_color = genre_colors.get(genre, '#ffeb3b')
                
                song = {
                    'item_id': str(row['item_id']),
                    'title': str(row['title']),
                    'author': str(row['author']),
                    'genre': genre,
                    'genre_color': genre_color,
                    'has_embedding': row['embedding'] is not None
                }
                songs.append(song)
                
            except Exception as e:
                logger.warning(f"Error processing song {row.get('item_id')}: {e}")
                continue
        
        # Use actual similarity traversal instead of arbitrary Voyager index order!
        logger.info(f"Creating similarity-based spiral traversal for {len(songs)} songs...")
        
        # Import voyager for similarity searches
        try:
            from tasks.voyager_manager import voyager_index, id_map, find_nearest_neighbors_by_id
            
            if voyager_index is not None and id_map is not None and songs:
                logger.info("Creating spiral based on ACTUAL similarity traversal")
                
                # Start from a central/popular song (first one we have)
                start_song_id = songs[0]['item_id']
                
                # Create similarity-based traversal order
                ordered_songs = []
                used_song_ids = set()
                current_song_id = start_song_id
                
                # Add the starting song
                start_song = next((s for s in songs if s['item_id'] == current_song_id), None)
                if start_song:
                    ordered_songs.append(start_song)
                    used_song_ids.add(current_song_id)
                
                # Create a lookup for faster song finding
                song_lookup = {s['item_id']: s for s in songs}
                
                # Build the similarity chain: each next song is the most similar unused song
                for step in range(len(songs) - 1):
                    try:
                        # Find neighbors of current song
                        neighbors = find_nearest_neighbors_by_id(current_song_id, n=50)  # Get more options
                        
                        # Find the first unused neighbor that we have in our song set
                        next_song = None
                        for neighbor in neighbors:
                            neighbor_id = str(neighbor['item_id'])
                            if neighbor_id not in used_song_ids and neighbor_id in song_lookup:
                                next_song = song_lookup[neighbor_id]
                                current_song_id = neighbor_id
                                break
                        
                        if next_song:
                            ordered_songs.append(next_song)
                            used_song_ids.add(current_song_id)
                        else:
                            # If no neighbors found, pick the next unused song
                            for song in songs:
                                if song['item_id'] not in used_song_ids:
                                    ordered_songs.append(song)
                                    used_song_ids.add(song['item_id'])
                                    current_song_id = song['item_id']
                                    break
                            else:
                                break  # No more songs
                                
                    except Exception as e:
                        logger.warning(f"Error finding neighbors for {current_song_id}: {e}")
                        # Fallback: add remaining unused songs
                        for song in songs:
                            if song['item_id'] not in used_song_ids:
                                ordered_songs.append(song)
                                used_song_ids.add(song['item_id'])
                                current_song_id = song['item_id']
                                break
                        else:
                            break
                
                # Update songs list to the similarity-ordered sequence
                songs = ordered_songs
                logger.info(f"Created similarity traversal chain with {len(songs)} songs")
                
            else:
                logger.warning("Voyager index not available, keeping original order")
        
        except Exception as e:
            logger.error(f"Error creating similarity traversal: {e}, keeping original order")
        
        # Now position songs in spiral where consecutive positions = consecutive similarity
        positions_2d = []
        
        # Position songs in spiral pattern where each step is the next most similar song
        for i, song in enumerate(songs):
            # Create tight spiral where position i+1 is the most similar to position i
            angle = i * 0.15  # Small angle increment for tight spiral
            radius = np.sqrt(i) * 10  # Gradual radius growth
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            positions_2d.append([x, y])
        
        positions_2d = np.array(positions_2d)
        logger.info(f"Created similarity-based spiral where position N+1 is most similar to position N")
        
        # Add positions to songs - simplified since we're using ordered Voyager positioning
        for i, song in enumerate(songs):
            if i < len(positions_2d):
                song['x'] = float(positions_2d[i, 0])
                song['y'] = float(positions_2d[i, 1])
            else:
                # Fallback for any extra songs
                song['x'] = float(random.uniform(-500, 500))
                song['y'] = float(random.uniform(-500, 500))
        
        logger.info(f"Created organized music map with {len(songs)} songs positioned by Voyager index order")
        
        return jsonify({
            "songs": songs,
            "total": len(songs),
            "bounds": {
                "min_x": float(np.min(positions_2d[:, 0])),
                "max_x": float(np.max(positions_2d[:, 0])),
                "min_y": float(np.min(positions_2d[:, 1])),
                "max_y": float(np.max(positions_2d[:, 1]))
            }
        })
        
    except Exception as e:
        logger.error(f"Error creating music map: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500