PORT = 8003
WHISPER_MODEL = "base"
CHUNK_DURATION_MS = 4000
SAMPLE_RATE = 16000
OUTPUT_DIR = "transcriptions"
MIN_AUDIO_LENGTH_SEC = 6

# Device Configuration
DEVICE = "auto"  # "auto", "cpu", "cuda"
COMPUTE_TYPE = "auto"  # "auto", "int8", "float16", "int8_float16"

# Upload Configuration
MAX_UPLOAD_SIZE_MB = 50  # Maximum file size in MB (0 = unlimited)

# LLM Configuration
LLM_PROVIDER = "ollama"  # "ollama" or "openai"
LLM_MODEL = "llama3.2"
LLM_BASE_URL = "http://localhost:11434"  # For Ollama
OPENAI_API_KEY = ""  # Set if using OpenAI
OPENAI_MODEL = "gpt-4"  # Model to use if provider is "openai"
