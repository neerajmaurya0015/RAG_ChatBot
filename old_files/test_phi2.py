import chromadb
from sentence_transformers import SentenceTransformer
from gpt4all import GPT4All
 
# -------------------------------
# ✅ LOAD EMBEDDING MODEL
# -------------------------------
embed_model = SentenceTransformer(
    r"C:\Users\neeraj.maurya01\Documents\GE_Ai\QnA_Bot\models\all-MiniLM-L6-v2"
)
 
# -------------------------------
# ✅ LOAD LOCAL LLM
# -------------------------------
llm = GPT4All(
    model_name=r"C:\Users\neeraj.maurya01\Documents\GE_Ai\QnA_Bot\models\phi-2.Q4_K_M.gguf"
)
 
# -------------------------------
# ✅ CONNECT DB
# -------------------------------
client = chromadb.PersistentClient(
    path=r"C:\Users\neeraj.maurya01\Documents\GE_Ai\QnA_Bot\vector_db"
)
 
collection = client.get_collection("aveva_docs")
 
 
# -------------------------------
# 🚀 ASK FUNCTION
# -------------------------------
def ask(question):
    print(f"\n❓ QUESTION: {question}")
 
    # Step 1: Embed query
    query_embedding = embed_model.encode(question).tolist()
 
    # Step 2: Retrieve chunks
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=8
    )
 
    docs = results["documents"][0]
 
    print("\n🔎 TOP MATCHED CHUNKS:\n")
 
    for i, doc in enumerate(docs):
        print(f"\n--- Chunk {i+1} ---")
        print(doc[:400])
        print("------------")
 
    context = "\n\n".join(docs)
 
    # -------------------------------
    # 🧠 PROMPT
    # -------------------------------
    prompt = f"""
You are an AVEVA E3D / PML expert.
 
Use ONLY the context below.
 
Give:
- clear explanation
- commands if present
 
If not found → say "Not found in context"
 
Context:
{context}
 
Question:
{question}
"""
 
    print("\n🧠 Generating answer...\n")
 
    # -------------------------------
    # ⚡ RUN LOCAL LLM
    # -------------------------------
    response = llm.generate(
        prompt,
        max_tokens=500,
        temp=0.3
    )
 
    print("\nANSWER:\n")
    print(response)
 
 
# -------------------------------
# 🧪 TEST
# -------------------------------
ask("How to create PML form in AVEVA E3D?")
 