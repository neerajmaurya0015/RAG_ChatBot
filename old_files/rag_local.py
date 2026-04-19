import chromadb
import ollama
from sentence_transformers import SentenceTransformer

# Load same embedding model used during indexing
# model = SentenceTransformer("all-MiniLM-L6-v2")
#from sentence_transformers import SentenceTransformer

model = SentenceTransformer(r"C:\Users\neeraj.maurya01\Documents\GE_Ai\QnA_Bot\models\all-MiniLM-L6-v2")


client = chromadb.PersistentClient(
    path=r"C:\Users\neeraj.maurya01\Documents\GE_Ai\QnA_Bot\vector_db"
)

collection = client.get_collection("aveva_docs")


def ask(question):

    # Create embedding for the query
    query_embedding = model.encode(question).tolist()

    # Retrieve similar chunks
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )

    context = "\n".join(results["documents"][0])

    prompt = f"""
You are an AVEVA E3D expert assistant.

Answer the question using the context below.

Context:
{context}

Question:
{question}
"""

    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )

    print("\nANSWER:\n")
    print(response["message"]["content"])


ask("How to create PML form?")