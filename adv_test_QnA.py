from retriever import retrieve
from llm import generate_stream
from utils import compress_context
 
 
def ask(question):
 
    docs = retrieve(question, top_k=3)
 
    print("\n================ RAW CHUNKS =================\n")
 
    for i, (doc, meta) in enumerate(docs):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Source: {meta.get('source')}")
        print(doc[:500])
 
    #  NEW: Compress context
    compressed_docs = compress_context(question, docs)
 
    print("\n================ COMPRESSED CHUNKS =================\n")
 
    for i, (doc, meta) in enumerate(compressed_docs):
        print(f"\n--- Compressed {i+1} ---")
        print(f"Source: {meta.get('source')}")
        print(doc)
 
    # build context
    context = "\n\n".join([doc for doc, _ in compressed_docs])
 
    prompt = f"""
You are an AVEVA E3D expert.
 
Use ONLY the context below.
 
Context:
{context}
 
Question:
{question}
 
Answer clearly:
"""
 
    print("\n================ ANSWER =================\n")
 
    for token in generate_stream(prompt):
        print(token, end="", flush=True)
 
    print("\n")
 
 
while True:
    q = input("\nAsk something (type 'exit'): ")
    if q == "exit":
        break
    ask(q)
 
 
