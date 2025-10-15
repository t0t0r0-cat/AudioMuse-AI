#AudioMuse-AI/config.py
import os

# --- Media Server Type ---
MEDIASERVER_TYPE = os.environ.get("MEDIASERVER_TYPE", "jellyfin").lower() # Possible values: jellyfin, navidrome, lyrion, mpd

# --- Jellyfin and DB Constants (Read from Environment Variables first) ---

# JELLYFIN_USER_ID and JELLYFIN_TOKEN come from a Kubernetes Secret
JELLYFIN_URL = os.environ.get("JELLYFIN_URL", "http://your_jellyfin_url:8096") # Replace with your default URL
JELLYFIN_USER_ID = os.environ.get("JELLYFIN_USER_ID", "your_default_user_id")  # Replace with a suitable default or handle missing case
JELLYFIN_TOKEN = os.environ.get("JELLYFIN_TOKEN", "your_default_token")  # Replace with a suitable default or handle missing case
TEMP_DIR = "/app/temp_audio"  # Always use /app/temp_audio
HEADERS = {"X-Emby-Token": JELLYFIN_TOKEN}
# Whether Jellyfin item queries should include nested items. Can be overridden via environment.
# Set to 'False' if your Jellyfin setup requires non-recursive queries.
JELLYFIN_RECURSIVE = os.environ.get("JELLYFIN_RECURSIVE", "True").lower() == "true"

# --- Navidrome (Subsonic API) Constants ---
# These are used only if MEDIASERVER_TYPE is "navidrome".
NAVIDROME_URL = os.environ.get("NAVIDROME_URL", "http://your_navidrome_url:4533")
NAVIDROME_USER = os.environ.get("NAVIDROME_USER", "your_navidrome_user")
NAVIDROME_PASSWORD = os.environ.get("NAVIDROME_PASSWORD", "your_navidrome_password") # Use the password directly

# --- Lyrion (LMS) Constants ---
# These are used only if MEDIASERVER_TYPE is "lyrion".
LYRION_URL = os.environ.get("LYRION_URL", "http://your_lyrion_url:9000")

# --- MPD (Music Player Daemon) Constants ---
# These are used only if MEDIASERVER_TYPE is "mpd".
MPD_HOST = os.environ.get("MPD_HOST", "localhost")
MPD_PORT = int(os.environ.get("MPD_PORT", "6600"))
MPD_PASSWORD = os.environ.get("MPD_PASSWORD", "")  # Optional password, leave empty if none
MPD_MUSIC_DIRECTORY = os.environ.get("MPD_MUSIC_DIRECTORY", "/var/lib/mpd/music")  # Path to MPD's music directory for file access


# --- General Constants (Read from Environment Variables where applicable) ---
APP_VERSION = "v0.7.0-beta"
MAX_DISTANCE = 0.5
MAX_SONGS_PER_CLUSTER = 0
MAX_SONGS_PER_ARTIST = int(os.getenv("MAX_SONGS_PER_ARTIST", "3")) # Max songs per artist in similarity results and clustering
# New: Default behavior for eliminating duplicates in similarity search. If param not passed to API, this is the default.
SIMILARITY_ELIMINATE_DUPLICATES_DEFAULT = os.environ.get("SIMILARITY_ELIMINATE_DUPLICATES_DEFAULT", "True").lower() == 'true'
NUM_RECENT_ALBUMS = int(os.getenv("NUM_RECENT_ALBUMS", "0")) # Convert to int
TOP_N_PLAYLISTS = int(os.environ.get("TOP_N_PLAYLISTS", "8")) # *** NEW: Default for Top N diverse playlists ***
MIN_PLAYLIST_SIZE_FOR_TOP_N = int(os.environ.get("MIN_PLAYLIST_SIZE_FOR_TOP_N", "20")) # Min songs for a playlist to be considered in the first pass of Top-N selection.

# --- Algorithm Choose Constants (Read from Environment Variables) ---
CLUSTER_ALGORITHM = os.environ.get("CLUSTER_ALGORITHM", "kmeans") # accepted dbscan, kmeans, gmm, or spectral
AI_MODEL_PROVIDER = os.environ.get("AI_MODEL_PROVIDER", "NONE").upper() # Accepted: OLLAMA, GEMINI, MISTRAL, NONE
ENABLE_CLUSTERING_EMBEDDINGS = os.environ.get("ENABLE_CLUSTERING_EMBEDDINGS", "True").lower() == "true"

# --- DBSCAN Only Constants (Ranges for Evolutionary Approach) ---
# Default ranges for DBSCAN parameters
DBSCAN_EPS_MIN = float(os.getenv("DBSCAN_EPS_MIN", "0.1"))
DBSCAN_EPS_MAX = float(os.getenv("DBSCAN_EPS_MAX", "0.5"))
DBSCAN_MIN_SAMPLES_MIN = int(os.getenv("DBSCAN_MIN_SAMPLES_MIN", "5"))
DBSCAN_MIN_SAMPLES_MAX = int(os.getenv("DBSCAN_MIN_SAMPLES_MAX", "20"))


# --- KMEANS Only Constants (Ranges for Evolutionary Approach) ---
# Default ranges for KMeans parameters
NUM_CLUSTERS_MIN = int(os.getenv("NUM_CLUSTERS_MIN", "40"))
NUM_CLUSTERS_MAX = int(os.getenv("NUM_CLUSTERS_MAX", "100"))
# New for MiniBatchKMeans
USE_MINIBATCH_KMEANS = os.environ.get("USE_MINIBATCH_KMEANS", "False").lower() == "true" # Enable MiniBatchKMeans
MINIBATCH_KMEANS_PROCESSING_BATCH_SIZE = int(os.getenv("MINIBATCH_KMEANS_PROCESSING_BATCH_SIZE", "1000")) # Internal batch size for MiniBatchKMeans partial_fit

# --- GMM Only Constants (Ranges for Evolutionary Approach) ---
# Default ranges for GMM parameters
GMM_N_COMPONENTS_MIN = int(os.getenv("GMM_N_COMPONENTS_MIN", "40"))
GMM_N_COMPONENTS_MAX = int(os.getenv("GMM_N_COMPONENTS_MAX", "100"))
GMM_COVARIANCE_TYPE = os.environ.get("GMM_COVARIANCE_TYPE", "full") # 'full', 'tied', 'diag', 'spherical'

# --- SpectralClustering Only Constants (Ranges for Evolutionary Approach) ---
SPECTRAL_N_CLUSTERS_MIN = int(os.getenv("SPECTRAL_N_CLUSTERS_MIN", "40"))
SPECTRAL_N_CLUSTERS_MAX = int(os.getenv("SPECTRAL_N_CLUSTERS_MAX", "100"))
SPECTRAL_N_NEIGHBORS = int(os.getenv("SPECTRAL_N_NEIGHBORS", "20"))

# --- PCA Constants (Ranges for Evolutionary Approach) ---
# Default ranges for PCA components
PCA_COMPONENTS_MIN = int(os.getenv("PCA_COMPONENTS_MIN", "0")) # 0 to disable PCA
PCA_COMPONENTS_MAX = int(os.getenv("PCA_COMPONENTS_MAX", "199")) # Max components for PCA 8 for score vectore, 199 for embeding

# --- Clustering Runs for Diversity (New Constant) ---
CLUSTERING_RUNS = int(os.environ.get("CLUSTERING_RUNS", "5000")) # Default to 100 runs for evolutionary search
MAX_QUEUED_ANALYSIS_JOBS = int(os.environ.get("MAX_QUEUED_ANALYSIS_JOBS", "100")) # Max album analysis jobs to keep in RQ queue

# --- Batching Constants for Clustering Runs ---
ITERATIONS_PER_BATCH_JOB = int(os.environ.get("ITERATIONS_PER_BATCH_JOB", "20")) # Number of clustering iterations per RQ batch job
MAX_CONCURRENT_BATCH_JOBS = int(os.environ.get("MAX_CONCURRENT_BATCH_JOBS", "10")) # Max number of batch jobs to run concurrently
DB_FETCH_CHUNK_SIZE = int(os.environ.get("DB_FETCH_CHUNK_SIZE", "1000")) # Chunk size for fetching full track data from DB in batch jobs

# --- Batching Constants for Analysis ---
REBUILD_INDEX_BATCH_SIZE = int(os.environ.get("REBUILD_INDEX_BATCH_SIZE", "10")) # Rebuild Voyager index after this many albums are analyzed.
AUDIO_LOAD_TIMEOUT = int(os.getenv("AUDIO_LOAD_TIMEOUT", "600")) # Timeout in seconds for loading a single audio file.

# --- Guided Evolutionary Clustering Constants ---
TOP_N_ELITES = int(os.environ.get("CLUSTERING_TOP_N_ELITES", "10")) # Number of best solutions to keep as elites
EXPLOITATION_START_FRACTION = float(os.environ.get("CLUSTERING_EXPLOITATION_START_FRACTION", "0.2")) # Fraction of runs before starting to use elites (e.g., 0.2 means after 20% of runs)
EXPLOITATION_PROBABILITY_CONFIG = float(os.environ.get("CLUSTERING_EXPLOITATION_PROBABILITY", "0.7")) # Probability of mutating an elite vs. random generation, once exploitation starts
MUTATION_INT_ABS_DELTA = int(os.environ.get("CLUSTERING_MUTATION_INT_ABS_DELTA", "3")) # Max absolute change for integer parameter mutation
MUTATION_FLOAT_ABS_DELTA = float(os.environ.get("CLUSTERING_MUTATION_FLOAT_ABS_DELTA", "0.05")) # Max absolute change for float parameter mutation (e.g., for DBSCAN eps)
MUTATION_KMEANS_COORD_FRACTION = float(os.environ.get("CLUSTERING_MUTATION_KMEANS_COORD_FRACTION", "0.05")) # Fractional change for KMeans centroid coordinates based on data range

# --- Scoring Weights for Enhanced Diversity Score ---
SCORE_WEIGHT_DIVERSITY = float(os.environ.get("SCORE_WEIGHT_DIVERSITY", "2.0")) # Weight for the base diversity (inter-playlist mood diversity)
SCORE_WEIGHT_PURITY = float(os.environ.get("SCORE_WEIGHT_PURITY", "1.0"))    # Weight for playlist purity (intra-playlist mood consistency)
SCORE_WEIGHT_OTHER_FEATURE_DIVERSITY = float(os.environ.get("SCORE_WEIGHT_OTHER_FEATURE_DIVERSITY", "0.0")) # New: Weight for inter-playlist other feature diversity
SCORE_WEIGHT_OTHER_FEATURE_PURITY = float(os.environ.get("SCORE_WEIGHT_OTHER_FEATURE_PURITY", "0.0"))       # New: Weight for intra-playlist other feature consistency
# --- Weights for Internal Validation Metrics ---
SCORE_WEIGHT_SILHOUETTE = float(os.environ.get("SCORE_WEIGHT_SILHOUETTE", "0.0")) # ex 0.6 - Weight for Silhouette Score - This metric measures how similar an object is to its own cluster compared to other clusters.
SCORE_WEIGHT_DAVIES_BOULDIN = float(os.environ.get("SCORE_WEIGHT_DAVIES_BOULDIN", "0.0")) # Set to 0 to effectively disable - This index quantifies the average similarity between each cluster and its most similar one
SCORE_WEIGHT_CALINSKI_HARABASZ = float(os.environ.get("SCORE_WEIGHT_CALINSKI_HARABASZ", "0.0")) # Set to 0 to effectively disable - This metric focuses on the ratio of between-cluster dispersion to within-cluster dispersion
TOP_K_MOODS_FOR_PURITY_CALCULATION = int(os.environ.get("TOP_K_MOODS_FOR_PURITY_CALCULATION", "3")) # Number of centroid's top moods to consider for purity

# --- Statistics for Raw Score Scaling (Mood Diversity and Purity) ---
# These are based on observed typical ranges for the raw scores.
# The 'sd' (standard deviation) is stored as requested but not used in the current LN + MinMax scaling.
# Constants for Log-Transformed and Standardized Mood Diversity
LN_MOOD_DIVERSITY_STATS = {
    "min": float(os.environ.get("LN_MOOD_DIVERSITY_MIN", "-0.1863")),
    "max": float(os.environ.get("LN_MOOD_DIVERSITY_MAX", "1.5518")),
    "mean": float(os.environ.get("LN_MOOD_DIVERSITY_MEAN", "0.9995")),
    "sd": float(os.environ.get("LN_MOOD_DIVERSITY_SD", "0.3541"))
}

# Constants for Log-Transformed and Standardized Mood Diversity WHEN EMBEDDINGS ARE USED
LN_MOOD_DIVERSITY_EMBEDING_STATS = { # Corrected spelling to "EMBEDING"
    "min": float(os.environ.get("LN_MOOD_DIVERSITY_EMBEDDING_MIN", "-0.174")),
    "max": float(os.environ.get("LN_MOOD_DIVERSITY_EMBEDDING_MAX", "0.570")),
    "mean": float(os.environ.get("LN_MOOD_DIVERSITY_EMBEDDING_MEAN", "-0.101")),
    "sd": float(os.environ.get("LN_MOOD_DIVERSITY_EMBEDDING_SD", "0.245")) # Kept env var name consistent for now
}

# Constants for Log-Transformed and Standardized Mood Purity
LN_MOOD_PURITY_STATS = {
    "min": float(os.environ.get("LN_MOOD_PURITY_MIN", "0.6981")),
    "max": float(os.environ.get("LN_MOOD_PURITY_MAX", "7.2848")),
    "mean": float(os.environ.get("LN_MOOD_PURITY_MEAN", "5.8679")),
    "sd": float(os.environ.get("LN_MOOD_PURITY_SD", "1.1557"))
}

# Constants for Log-Transformed and Standardized Mood Purity WHEN EMBEDDINGS ARE USED
LN_MOOD_PURITY_EMBEDING_STATS = { # Note: User provided "EMBEDING" spelling
    "min": float(os.environ.get("LN_MOOD_PURITY_EMBEDDING_MIN", "-0.494")),
    "max": float(os.environ.get("LN_MOOD_PURITY_EMBEDDING_MAX", "2.583")),
    "mean": float(os.environ.get("LN_MOOD_PURITY_EMBEDDING_MEAN", "0.673")),
    "sd": float(os.environ.get("LN_MOOD_PURITY_EMBEDDING_SD", "1.063"))
}

# --- Statistics for Log-Transformed and Standardized "Other Features" Scores ---
# IMPORTANT: Replace these placeholder values with actual statistics derived from your data.
# These are used for Z-score standardization of the "other features" diversity and purity.
LN_OTHER_FEATURES_DIVERSITY_STATS = {
    "min": float(os.environ.get("LN_OTHER_FEAT_DIV_MIN", "-0.19")), # Placeholder
    "max": float(os.environ.get("LN_OTHER_FEAT_DIV_MAX", "2.06")), # Placeholder
    "mean": float(os.environ.get("LN_OTHER_FEAT_DIV_MEAN", "1.5")), # Placeholder
    "sd": float(os.environ.get("LN_OTHER_FEAT_DIV_SD", "0.46"))      # Placeholder
}

LN_OTHER_FEATURES_PURITY_STATS = {
    "min": float(os.environ.get("LN_OTHER_FEAT_PUR_MIN", "8.67")),   # Updated value
    "max": float(os.environ.get("LN_OTHER_FEAT_PUR_MAX", "8.95")),   # Updated value
    "mean": float(os.environ.get("LN_OTHER_FEAT_PUR_MEAN", "8.84")),  # Updated value
    "sd": float(os.environ.get("LN_OTHER_FEAT_PUR_SD", "0.07"))     # Updated value
}

# Threshold for considering an "other feature" predominant in a playlist for purity calculation
OTHER_FEATURE_PREDOMINANCE_THRESHOLD_FOR_PURITY = float(os.environ.get("OTHER_FEATURE_PREDOMINANCE_THRESHOLD_FOR_PURITY", "0.3"))

# --- AI Playlist Naming ---
# USE_AI_PLAYLIST_NAMING is replaced by AI_MODEL_PROVIDER
OLLAMA_SERVER_URL = os.environ.get("OLLAMA_SERVER_URL", "http://192.168.3.15:11434/api/generate") # URL for your Ollama instance
OLLAMA_MODEL_NAME = os.environ.get("OLLAMA_MODEL_NAME", "mistral:7b") # Ollama model to use

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR-GEMINI-API-KEY-HERE") # Default API key
GEMINI_MODEL_NAME = os.environ.get("GEMINI_MODEL_NAME", "gemini-2.5-pro") # Default Gemini model gemini-2.5-pro, alternative gemini-1.5-flash-latest

MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "YOUR-GEMINI-API-KEY-HERE")
MISTRAL_MODEL_NAME = os.environ.get("MISTRAL_MODEL_NAME", "ministral-3b-latest")
REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')

# Construct DATABASE_URL from individual components for better security in K8s
POSTGRES_USER = os.environ.get("POSTGRES_USER", "audiomuse")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "audiomusepassword")
POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "postgres-service.playlist") # Default for K8s
POSTGRES_PORT = os.environ.get("POSTGRES_PORT", "5432")
POSTGRES_DB = os.environ.get("POSTGRES_DB", "audiomusedb")

DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

# --- AI User for Chat SQL Execution ---
AI_CHAT_DB_USER_NAME = os.environ.get("AI_CHAT_DB_USER_NAME", "ai_user")
AI_CHAT_DB_USER_PASSWORD = os.environ.get("AI_CHAT_DB_USER_PASSWORD", "ChangeThisSecurePassword123!") # IMPORTANT: Change this default and use environment variables

# --- Classifier Constant ---
MOOD_LABELS = [
    'rock', 'pop', 'alternative', 'indie', 'electronic', 'female vocalists', 'dance', '00s', 'alternative rock', 'jazz',
    'beautiful', 'metal', 'chillout', 'male vocalists', 'classic rock', 'soul', 'indie rock', 'Mellow', 'electronica', '80s',
    'folk', '90s', 'chill', 'instrumental', 'punk', 'oldies', 'blues', 'hard rock', 'ambient', 'acoustic', 'experimental',
    'female vocalist', 'guitar', 'Hip-Hop', '70s', 'party', 'country', 'easy listening', 'sexy', 'catchy', 'funk', 'electro',
    'heavy metal', 'Progressive rock', '60s', 'rnb', 'indie pop', 'sad', 'House', 'happy'
]

TOP_N_MOODS = 5
TOP_N_OTHER_FEATURES = int(os.environ.get("TOP_N_OTHER_FEATURES", "2")) # Number of top "other features" to consider for clustering vector
EMBEDDING_MODEL_PATH = "/app/model/msd-musicnn-1.onnx"
PREDICTION_MODEL_PATH = "/app/model/msd-msd-musicnn-1.onnx"
EMBEDDING_DIMENSION = 200

# --- Voyager Index Constants ---
INDEX_NAME = os.environ.get("VOYAGER_INDEX_NAME", "music_library") # The primary key for our index in the DB
VOYAGER_METRIC = os.environ.get("VOYAGER_METRIC", "angular") # Options: 'angular' (Cosine), 'euclidean', 'dot' (InnerProduct)
VOYAGER_EF_CONSTRUCTION = int(os.environ.get("VOYAGER_EF_CONSTRUCTION", "1024"))
VOYAGER_M = int(os.environ.get("VOYAGER_M", "64"))
VOYAGER_QUERY_EF = int(os.environ.get("VOYAGER_QUERY_EF", "1024"))

# --- Pathfinding Constants ---
# The distance metric to use for pathfinding. Options: 'angular', 'euclidean'.
PATH_DISTANCE_METRIC = os.environ.get("PATH_DISTANCE_METRIC", "angular").lower()
# Default number of songs in the path if not specified in the API request.
PATH_DEFAULT_LENGTH = int(os.environ.get("PATH_DEFAULT_LENGTH", "25"))
# Number of random songs to sample for calculating the average jump distance.
PATH_AVG_JUMP_SAMPLE_SIZE = int(os.environ.get("PATH_AVG_JUMP_SAMPLE_SIZE", "200"))
# Number of candidate songs to retrieve from Voyager for each step in the path.
PATH_CANDIDATES_PER_STEP = int(os.environ.get("PATH_CANDIDATES_PER_STEP", "25"))
# Multiplier for the core number of steps (Lcore) to generate more backbone centroids.
PATH_LCORE_MULTIPLIER = int(os.environ.get("PATH_LCORE_MULTIPLIER", "3"))


# --- Other Essentia Model Paths ---
# Paths for models used in predict_other_models (VGGish-based)
DANCEABILITY_MODEL_PATH = os.environ.get("DANCEABILITY_MODEL_PATH", "/app/model/danceability-msd-musicnn-1.onnx") # Example, adjust if different
AGGRESSIVE_MODEL_PATH = os.environ.get("AGGRESSIVE_MODEL_PATH", "/app/model/mood_aggressive-msd-musicnn-1.onnx")
HAPPY_MODEL_PATH = os.environ.get("HAPPY_MODEL_PATH", "/app/model/mood_happy-msd-musicnn-1.onnx")
PARTY_MODEL_PATH = os.environ.get("PARTY_MODEL_PATH", "/app/model/mood_party-msd-musicnn-1.onnx")
RELAXED_MODEL_PATH = os.environ.get("RELAXED_MODEL_PATH", "/app/model/mood_relaxed-msd-musicnn-1.onnx")
SAD_MODEL_PATH = os.environ.get("SAD_MODEL_PATH", "/app/model/mood_sad-msd-musicnn-1.onnx")

# --- Energy Normalization Range ---
ENERGY_MIN = float(os.getenv("ENERGY_MIN", "0.01"))
ENERGY_MAX = float(os.getenv("ENERGY_MAX", "0.15"))

# --- Tempo Normalization Range (BPM) ---
TEMPO_MIN_BPM = float(os.getenv("TEMPO_MIN_BPM", "40.0"))
TEMPO_MAX_BPM = float(os.getenv("TEMPO_MAX_BPM", "200.0"))
OTHER_FEATURE_LABELS = ['danceable', 'aggressive', 'happy', 'party', 'relaxed', 'sad']

# --- Sonic Fingerprint Constants ---
SONIC_FINGERPRINT_TOP_N_SONGS = int(os.environ.get("SONIC_FINGERPRINT_TOP_N_SONGS", "20"))
SONIC_FINGERPRINT_NEIGHBORS = int(os.environ.get("SONIC_FINGERPRINT_NEIGHBORS", "100"))

# --- Database Cleaning Safety ---
CLEANING_SAFETY_LIMIT = int(os.environ.get("CLEANING_SAFETY_LIMIT", "100"))  # Max orphaned albums to delete in one run

# --- Stratified Sampling Constants (New) ---
# Genres for which to enforce equal representation during stratified sampling
STRATIFIED_GENRES = [
    'rock', 'pop', 'alternative', 'indie', 'electronic', 'jazz', 'metal', 'classic rock', 'soul',
    'indie rock', 'electronica', 'folk', 'punk', 'blues', 'hard rock', 'ambient', 'acoustic',
    'experimental', 'Hip-Hop', 'country', 'funk', 'electro', 'heavy metal', 'Progressive rock',
    'rnb', 'indie pop', 'House'
]

# Minimum number of songs to target per genre for stratified sampling.
# This will be dynamically adjusted based on actual available songs.
MIN_SONGS_PER_GENRE_FOR_STRATIFICATION = int(os.getenv("MIN_SONGS_PER_GENRE_FOR_STRATIFICATION", "100"))

# Percentile to use for determining the target number of songs per genre in stratified sampling.
# E.g., 75 means the target will be based on the 75th percentile of song counts among stratified genres.
STRATIFIED_SAMPLING_TARGET_PERCENTILE = int(os.getenv("STRATIFIED_SAMPLING_TARGET_PERCENTILE", "50"))

# Percentage of songs to change in the stratified sample between clustering runs (0.0 to 1.0)
SAMPLING_PERCENTAGE_CHANGE_PER_RUN = float(os.getenv("SAMPLING_PERCENTAGE_CHANGE_PER_RUN", "0.2"))


# --- NEW: Duplicate Detection by Distance ---
# Threshold for considering songs as duplicates based on their distance in the vector space.
# This helps catch identical songs with slightly different metadata (e.g., from different albums).
DUPLICATE_DISTANCE_THRESHOLD_COSINE = float(os.getenv("DUPLICATE_DISTANCE_THRESHOLD_COSINE", "0.01"))
DUPLICATE_DISTANCE_THRESHOLD_EUCLIDEAN = float(os.getenv("DUPLICATE_DISTANCE_THRESHOLD_EUCLIDEAN", "0.15"))
DUPLICATE_DISTANCE_CHECK_LOOKBACK = int(os.getenv("DUPLICATE_DISTANCE_CHECK_LOOKBACK", "1"))
