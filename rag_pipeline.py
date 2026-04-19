def smart_chunk(text, chunk_size=400, overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += (chunk_size - overlap)

    return chunks


def keyword_score(query, doc):
    score = 0
    for word in query.lower().split():
        if word in doc.lower():
            score += 1
    return score