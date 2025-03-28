import numpy as np
import chromadb
from app.vector_db import search_medical_data
from app.model import query_llama

# Initialize ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")
try:
    collection = client.get_collection(name="medical_assistant")
except Exception as e:
    print(f"Error getting collection: {e}")
    # Create collection if it doesn't exist
    collection = client.create_collection(name="medical_assistant")

def embed_text(text):
    """Generate embeddings using nomic-embed-text model."""
    import ollama  # Import here to avoid circular imports
    response = ollama.embeddings(model="nomic-embed-text:latest", prompt=text)
    return np.array(response["embedding"])  # Convert to NumPy array

def get_medical_response(symptoms: str):
    """Generate a medical response based on user symptoms."""
    # Convert symptoms into embeddings
    query_embedding = embed_text(symptoms)

    # Search for relevant medical information in ChromaDB
    retrieved_docs = search_medical_data(collection, query_embedding)
    context = "\n".join(retrieved_docs)

    # Generate AI response using the query_llama function from model.py
    prompt = f"""
    Given the following symptoms: {symptoms}
   
    Medical Data:
    {context}
   
    Provide a possible diagnosis, causes, and recommended recovery methods.
    """

    return query_llama(prompt)