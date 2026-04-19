import chromadb
import ollama
from sentence_transformers import SentenceTransformer
 

# LOAD EMBEDDING MODEL

model = SentenceTransformer(
    r"C:\Users\neeraj.maurya01\Documents\GE_Ai\QnA_Bot\models\bge-small-en"
)
 

#  CONNECT DB

client = chromadb.PersistentClient(
    path=r"C:\Users\neeraj.maurya01\Documents\GE_Ai\QnA_Bot\vector_db"
)
 
collection = client.get_collection("aveva_docs")
 
 

# ASK FUNCTION

def ask(question):
    print(f"\n ? QUESTION: {question}")
 
    # Step 1: Embed query
    query_embedding = model.encode(question).tolist()
 
    # Step 2: Retrieve chunks
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=8   #  increased for better context
    )
 
    docs = results["documents"][0]
 
    # -------------------------------
    #  DEBUG: SEE RETRIEVED CHUNKS
    # -------------------------------
    print("\n TOP MATCHED CHUNKS:\n")
 
    for i, doc in enumerate(docs):
        print(f"\n--- Chunk {i+1} ---")
        print(doc[:400])
        print("------------")
 
    # Step 3: Build context
    context = "\n\n".join(docs)
 
    # -------------------------------
    #  PROMPT (IMPROVED)
    # -------------------------------
    prompt = f"""
You are a SENIOR AVEVA E3D / PML expert.
 
Rules:
- Answer ONLY from given context
- Provide detailed technical explanation
- Include commands / syntax if available
- If not found → say "Not found in context"
 
Context:
{context}
 
Question:
{question}
"""
 
    print("\n Generating answer...\n")
    print("\nANSWER:\n")
 
    # -------------------------------
    # STEP 4: LLM CALL
    # -------------------------------
    stream = ollama.chat(
        model="llama3",   #  use existing model (no download needed)
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
 
    # Step 5: Print streaming response
    for chunk in stream:
        print(chunk["message"]["content"], end="", flush=True)
 
 
# -------------------------------
#  TEST
# -------------------------------
ask("How to create PML form in AVEVA E3D with syntax?")