import streamlit as st
from retriever import retrieve
from llm import generate_stream
from utils import compress_context
 
st.set_page_config(page_title="AVEVA QnA Bot")
 
st.title(" AVEVA RAG QnA Bot")
 
# Sidebar
st.sidebar.title(" Settings")
top_k = st.sidebar.slider("Top Chunks", 3, 10, 5)
 
question = st.text_input("Ask your question:")
 
if st.button("Get Answer"):
 
    if not question.strip():
        st.warning("Please enter a question")
        st.stop()
 
    #  STEP 1: Retrieve
    results = retrieve(question, top_k=top_k)
 
    # =========================
    #  RAW CHUNKS
    # =========================
    st.subheader(" Retrieved Chunks")
 
    context_parts = []
 
    for i, (doc, meta) in enumerate(results, 1):
        with st.expander(f"Chunk {i} | {meta.get('source')}"):
            st.write(doc)
 
        context_parts.append(doc)
 
    # =========================
    #  COMPRESSED CHUNKS
    # =========================
    compressed_docs = compress_context(question, results)
 
    st.subheader(" Compressed Chunks")
 
    compressed_parts = []
 
    for i, (doc, meta) in enumerate(compressed_docs, 1):
        with st.expander(f"Compressed {i} | {meta.get('source')}"):
            st.write(doc)
 
        compressed_parts.append(doc)
 
    # =========================
    #  FINAL CONTEXT
    # =========================
    context = "\n\n".join(compressed_parts)
 
    # =========================
    #  PROMPT
    # =========================
    prompt = f"""
You are an AVEVA E3D expert assistant.
 
Rules:
- Answer ONLY from the context
- Be precise and structured
- If not found, say "Not found in provided context"
 
Context:
{context}
 
Question:
{question}
 
Answer:
"""
 
    # =========================
    #  STREAMING ANSWER
    # =========================
    st.subheader(" Answer")
 
    response_area = st.empty()
    full_text = ""
 
    for token in generate_stream(prompt):
        full_text += token
        response_area.markdown(full_text)


