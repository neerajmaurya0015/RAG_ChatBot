import chromadb
import ollama
from sentence_transformers import SentenceTransformer
 
# Load LOCAL embedding model
model = SentenceTransformer(
    r"C:\Users\neeraj.maurya01\Documents\GE_Ai\QnA_Bot\models\all-MiniLM-L6-v2"
)
 
# Connect ChromaDB
client = chromadb.PersistentClient(
    path=r"C:\Users\neeraj.maurya01\Documents\GE_Ai\QnA_Bot\vector_db"
)
 
collection = client.get_collection("aveva_docs")
 
 
def ask(question):
    # Step 1: Embed query
    query_embedding = model.encode(question).tolist()
 
    # Step 2: Search in vector DB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )
 
    context = "\n".join(results["documents"][0])
 
    # Step 3: Create prompt
    prompt = f"""
You are an AVEVA E3D expert assistant.
 
Answer ONLY from the given context.
If answer is not present, say "Not found in context".
 
Context:
{context}
 
Question:
{question}
"""
 
    print("\nANSWER:\n")
 
    #  Step 4: Streaming response
    stream = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
 
    #  Step 5: Print chunks live
    for chunk in stream:
        print(chunk["message"]["content"], end="", flush=True)
 
# Test
ask("How to create PML form?")
