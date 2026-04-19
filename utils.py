import re
 
def clean_text(text):
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
 
 
def structured_chunk(text, chunk_size=120, overlap=30):
 
    sections = re.split(r'\n[A-Z][A-Z\s]{3,}\n', text)
 
    chunks = []
 
    for section in sections:
        words = section.split()
 
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
 
            if len(chunk.strip()) > 80:
                chunks.append(chunk)
 
    return chunks
 
 
def keyword_score(query, text):
    q_words = set(query.lower().split())
    t_words = set(text.lower().split())
    return len(q_words & t_words)


 
def compress_context(query, docs_with_meta, max_sentences=5):
    """
    Compress retrieved chunks → keep only relevant sentences
    """
 
    compressed_chunks = []
 
    query_words = set(query.lower().split())
 
    for doc, meta in docs_with_meta:
 
        # split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', doc)
 
        scored_sentences = []
 
        for sent in sentences:
            sent_lower = sent.lower()
 
            # keyword overlap score
            overlap = sum(1 for w in query_words if w in sent_lower)
 
            # remove junk
            if len(sent.strip()) < 30:
                continue
 
            if "table" in sent_lower and overlap == 0:
                continue
 
            scored_sentences.append((overlap, sent))
 
        # sort by relevance
        scored_sentences.sort(reverse=True, key=lambda x: x[0])
 
        # take top sentences
        top_sentences = [s for _, s in scored_sentences[:max_sentences]]
 
        compressed_text = " ".join(top_sentences)
 
        compressed_chunks.append((compressed_text, meta))
 
    return compressed_chunks

'''import re
 
def clean_text(text):
    text = text.replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()
 
 
def smart_chunk(text, chunk_size=400, overlap=120):
    chunks = []
    start = 0
 
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
 
        if len(chunk.strip()) > 50:
            chunks.append(chunk.strip())
 
        start += (chunk_size - overlap)
 
    return chunks
 
 
def keyword_score(query, doc):
    score = 0
    for word in query.lower().split():
        if word in doc.lower():
            score += 1
    return score


import re
 
# -------------------------------
# SMART CHUNKING (STRUCTURE AWARE)
# -------------------------------
def smart_chunk(text, chunk_size=500):
 
    # Split by sections / headings / line breaks
    sections = re.split(r"\n\s*\n", text)
 
    chunks = []
    current_chunk = ""
 
    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue
 
        # If section too big → split further
        if len(sec) > chunk_size:
            sub_parts = split_large_text(sec, chunk_size)
            chunks.extend(sub_parts)
            continue
 
        if len(current_chunk) + len(sec) < chunk_size:
            current_chunk += "\n\n" + sec
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sec
 
    if current_chunk:
        chunks.append(current_chunk.strip())
 
    return chunks
 
 
def split_large_text(text, chunk_size):
    words = text.split()
    chunks = []
    temp = ""
 
    for word in words:
        if len(temp) + len(word) < chunk_size:
            temp += " " + word
        else:
            chunks.append(temp.strip())
            temp = word
 
    if temp:
        chunks.append(temp.strip())
 
    return chunks
 
 
# -------------------------------
# BETTER KEYWORD SCORING
# -------------------------------
def keyword_score(query, doc):
    score = 0
    query_words = query.lower().split()
 
    for word in query_words:
        if word in doc.lower():
            score += 2   # stronger weight
 
    return score'''