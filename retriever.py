import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from config import *
from utils import keyword_score
import numpy as np
 
#  Embedding model
embed_model = SentenceTransformer(MODEL_PATH)
 
#  Reranker (offline)
reranker = CrossEncoder(RERANKER_PATH)
 
# DB
client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
collection = client.get_collection(COLLECTION_NAME)
 
 
def retrieve(query, top_k=5):
 
    #  Query expansion
    queries = [
        query,
        f"explain {query}",
        f"definition of {query}"
    ]
 
    query_embeddings = embed_model.encode(queries)
    query_embedding = np.mean(query_embeddings, axis=0)
 
    #  Step 1: Retrieve candidates
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=25
    )
 
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
 
    candidates = []
 
    for doc, meta, dist in zip(docs, metas, distances):
 
        if not doc or len(doc.strip()) < 100:
            continue
 
        # similarity
        semantic_score = 1 - dist
 
        # keyword
        keyword = keyword_score(query, doc)
        keyword_norm = keyword / (len(query.split()) + 1)
 
        hybrid_score = (0.6 * semantic_score) + (0.4 * keyword_norm)
 
        candidates.append((hybrid_score, doc, meta))
 
    # 🔥 Step 2: shortlist
    candidates.sort(reverse=True, key=lambda x: x[0])
    candidates = candidates[:15]
 
    # 🔥 Step 3: RERANK
    pairs = [(query, doc) for _, doc, _ in candidates]
    rerank_scores = reranker.predict(pairs)
 
    reranked = []
    for score, (hybrid, doc, meta) in zip(rerank_scores, candidates):
        reranked.append((score, doc, meta))
 
        print(f"\n🔎 Rerank Score: {score:.3f}")
        print(f"📄 Source: {meta.get('source', 'unknown')}")
        print(doc[:200])
 
    # 🔥 Final sort
    reranked.sort(reverse=True, key=lambda x: x[0])
 
    top_docs = [(doc, meta) for _, doc, meta in reranked[:top_k]]
 
    return top_docs