"""
Project settings. Edit these if you need to.

OLLAMA_MODEL: the local AI model used for reading notes and writing SQL.
  - "llama3.2:3b"  -> smaller and faster, works on most laptops (8 GB RAM)  [default]
  - "llama3.1:8b"  -> more accurate, but needs 16 GB RAM and is slower
  Whichever you choose, download it first with:  ollama pull <model name>
"""
import os

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")

# File locations
DATA_DIR = "data"
RESULTS_DIR = "results"
DB_PATH = os.path.join(DATA_DIR, "clinic.db")

# Random seed so everyone who runs the project gets the same fake data
SEED = 42
