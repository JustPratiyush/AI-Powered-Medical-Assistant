import ollama

def query_llama(prompt):
    """Send a prompt to the Llama model and return its response."""
    response = ollama.chat(model="llama3.2:latest", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]