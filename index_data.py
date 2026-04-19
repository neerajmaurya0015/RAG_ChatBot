import os
import fitz
import pdfplumber
import chromadb
import pytesseract
from pdf2image import convert_from_path
from sentence_transformers import SentenceTransformer
from config import *
from utils import clean_text, structured_chunk
 
#  SET TESSERACT PATH
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\neeraj.maurya01\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
 
print(" INDEXING STARTED")
 
model = SentenceTransformer(MODEL_PATH)
client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
 
try:
    client.delete_collection(COLLECTION_NAME)
except:
    pass
 
collection = client.create_collection(COLLECTION_NAME)
 
 
def extract_text(pdf_path):
    text = ""
 
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text()
    except:
        pass
 
    text = clean_text(text)
 
    if len(text) < 500:
        print(" Using OCR...")
        images = convert_from_path(pdf_path)
        for img in images:
            text += pytesseract.image_to_string(img)
 
    return clean_text(text)
 
 
def extract_tables(pdf_path):
    tables_text = ""
 
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
 
                for table in tables:
                    for row in table:
                        row = [(c.replace("\n", " ").strip() if c else "") for c in row]
                        tables_text += " | ".join(row) + "\n"
    except:
        pass
 
    return clean_text(tables_text)
 
 
all_chunks, ids, metas = [], [], []
doc_id = 0
 
for file in os.listdir(DATA_PATH):
 
    if not file.endswith(".pdf"):
        continue
 
    print(f" Processing: {file}")
    path = os.path.join(DATA_PATH, file)
 
    text = extract_text(path)
    tables = extract_tables(path)
 
    chunks = structured_chunk(text)
 
    if tables:
        table_chunks = structured_chunk(tables)
        chunks.extend(["[TABLE]\n" + t for t in table_chunks])
 
    print(f"Chunks: {len(chunks)}")
 
    for i, chunk in enumerate(chunks):
 
        if len(chunk.strip()) < 80:
            continue
 
        all_chunks.append(chunk)
        ids.append(f"{doc_id}_{i}")
        metas.append({"source": file, "chunk_id": i})
 
    doc_id += 1
 
 
print(f"\nTotal chunks: {len(all_chunks)}")
 
embeddings = model.encode(all_chunks, batch_size=32,show_progress_bar=True).tolist()
print("Embedding completed")
for i in range(0, len(all_chunks), 500):
    collection.add(
        documents=all_chunks[i:i+500],
        embeddings=embeddings[i:i+500],
        ids=ids[i:i+500],
        metadatas=metas[i:i+500]
    )
 
print("INDEXING DONE")