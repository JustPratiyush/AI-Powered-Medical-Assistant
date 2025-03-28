import chromadb
import numpy as np
import ollama
import os
from app.data_loader import load_csv_data, preprocess_text

# Initialize ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")
try:
    collection = client.get_collection(name="medical_assistant")
except Exception as e:
    print(f"Creating new collection: {e}")
    collection = client.create_collection(name="medical_assistant")

def embed_text(text):
    """Generate embeddings using nomic-embed-text:latest model."""
    response = ollama.embeddings(model="nomic-embed-text:latest", prompt=text)
    return np.array(response["embedding"])

def load_data_to_chromadb():
    """Load medical datasets into ChromaDB with embeddings."""
    # Get the absolute path to the datasets folder
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    datasets_dir = os.path.join(base_dir, "datasets")
    
    datasets = [
        os.path.join(datasets_dir, "dataset.csv"),
        os.path.join(datasets_dir, "symptom_Description.csv"),
        os.path.join(datasets_dir, "symptom_precaution.csv"),
        os.path.join(datasets_dir, "Symptom-severity.csv")
    ]
   
    print("🔄 Loading datasets into ChromaDB...")
   
    for file_path in datasets:
        print(f"📂 Processing: {file_path}")
        
        # Use data_loader to load and preprocess the data
        df = load_csv_data(file_path)
        print(f"🟢 Loaded {len(df)} rows from {file_path}")
        
        # Convert DataFrame to text using preprocess_text
        text_entries = preprocess_text(df)
        
        # Process each text entry
        for i, text_data in enumerate(text_entries):
            embedding = embed_text(text_data)

            if embedding is None:
                print(f"⚠️ Skipping entry, embedding is None: {text_data}")
                continue

            # Add to ChromaDB with unique ID
            file_name = os.path.basename(file_path)
            collection.add(
                embeddings=[embedding.tolist()], 
                documents=[text_data], 
                ids=[f"{file_name}_{i}"]
            )
            print(f"✅ Added to ChromaDB: {text_data[:50]}...")  # Print first 50 chars

    print("✅ Data successfully loaded into ChromaDB!")

def search_medical_data(collection, query_embedding):
    """Search ChromaDB for relevant medical data using the query embedding."""
    try:
        # Try the newer API first
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=3
        )
        return results["documents"][0]
    except TypeError:
        try:
            # Try alternative parameter name
            results = collection.query(
                query_vectors=[query_embedding.tolist()],
                n_results=3
            )
            return results["documents"][0]
        except TypeError:
            # Last resort for older versions
            print("Warning: Using fallback ChromaDB querying method")
            results = collection.query(
                include=["documents"],
                where={},
                where_document={},
                n_results=3
            )
            return results["documents"][0]