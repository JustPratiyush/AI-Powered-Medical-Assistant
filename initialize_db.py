import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from app.vector_db import load_data_to_chromadb

if __name__ == "__main__":
    load_data_to_chromadb()