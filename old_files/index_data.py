import os
import fitz  # PyMuPDF
import pdfplumber
import chromadb
import pytesseract
from sentence_transformers import SentenceTransformer
from pdf2image import convert_from_path
 
# -------------------------------
# ✅ CONFIG (UPDATE PATHS ONLY IF NEEDED)
# -------------------------------
 
print("🚀 Starting indexing pipeline...")
 
# Tesseract path (VERY IMPORTANT)
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\neeraj.maurya01\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
 
# Poppler path
POPPLER_PATH = r"C:\poppler\Library\bin"
 
# PDF folder
DATA_PATH = r"C:\Users\neeraj.maurya01\Documents\GE_Ai\QnA_Bot\data\pdfs"
 
# Embedding model
model = SentenceTransformer(
    r"C:\Users\neeraj.maurya01\Documents\GE_Ai\QnA_Bot\models\all-MiniLM-L6-v2"
)
 
# Chroma DB
client = chromadb.PersistentClient(
    path=r"C:\Users\neeraj.maurya01\Documents\GE_Ai\QnA_Bot\vector_db"
)
 
# Reset collection
try:
    client.delete_collection("aveva_docs")
    print("🗑 Old collection deleted")
except:
    pass
 
collection = client.create_collection("aveva_docs")
 
# -------------------------------
# 📌 TEXT EXTRACTION
# -------------------------------
 
def extract_text_pymupdf(pdf_path):
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text()
    except Exception as e:
        print(f"❌ PyMuPDF error: {e}")
    return text.strip()
 
 
def extract_tables(pdf_path):
    tables_text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for table in tables:
                    for row in table:
                        tables_text += " | ".join(
                            [str(cell) if cell else "" for cell in row]
                        ) + "\n"
    except Exception as e:
        print(f"⚠️ Table extraction error: {e}")
    return tables_text
 
 
def extract_ocr(pdf_path):
    print(f"🔍 OCR running: {pdf_path}")
 
    try:
        images = convert_from_path(
            pdf_path,
            poppler_path=POPPLER_PATH
        )
    except Exception as e:
        print(f"❌ OCR failed: {e}")
        return ""
 
    text = ""
 
    for img in images:
        try:
            text += pytesseract.image_to_string(img)
        except Exception as e:
            print(f"⚠️ OCR error: {e}")
 
    return text
 
 
def process_pdf(pdf_path):
    print(f"\n📄 Processing: {pdf_path}")
 
    text = extract_text_pymupdf(pdf_path)
    tables = extract_tables(pdf_path)
 
    print(f"📊 PyMuPDF text length: {len(text)}")
    print(f"📊 Tables length: {len(tables)}")
 
    # OCR fallback
    if len(text) < 100:
        print("⚠️ Using OCR fallback...")
        text = extract_ocr(pdf_path)
 
    combined = text + "\n\n[TABLE DATA]\n" + tables
 
    return combined
 
 
# -------------------------------
# 📌 SMART CHUNKING
# -------------------------------
 
def smart_chunk(text, chunk_size=500):
    if not text.strip():
        return []
 
    sentences = text.split(".")
    chunks = []
    current = ""
 
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
 
        if len(current) + len(sentence) < chunk_size:
            current += sentence + ". "
        else:
            chunks.append(current.strip())
            current = sentence + ". "
 
    if current:
        chunks.append(current.strip())
 
    return chunks
 
 
# -------------------------------
# 🚀 MAIN PIPELINE
# -------------------------------
 
files = os.listdir(DATA_PATH)
print(f"\n📁 FILES FOUND: {files}")
 
all_chunks = []
ids = []
 
doc_id = 0
 
for file in files:
    if file.endswith(".pdf"):
        path = os.path.join(DATA_PATH, file)
 
        content = process_pdf(path)
 
        print(f"\n--- DEBUG: {file} ---")
        print(f"Content length: {len(content)}")
 
        chunks = smart_chunk(content)
 
        print(f"Chunks created: {len(chunks)}")
 
        if len(chunks) == 0:
            print("⚠️ Skipping empty document")
            continue
 
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            ids.append(f"{doc_id}_{i}")
 
        doc_id += 1
 
 
print(f"\n📦 Total chunks: {len(all_chunks)}")
 
# -------------------------------
# ❌ SAFETY CHECK
# -------------------------------
 
if len(all_chunks) == 0:
    print("❌ ERROR: No chunks created. Check PDF extraction.")
    exit()
 
# -------------------------------
# 📌 EMBEDDING
# -------------------------------
 
print("🧠 Creating embeddings...")
 
embeddings = model.encode(all_chunks, batch_size=32).tolist()
 
# -------------------------------
# 📌 STORE
# -------------------------------
 
print("💾 Storing in ChromaDB...")
 
collection.add(
    documents=all_chunks,
    embeddings=embeddings,
    ids=ids
)
 
print("\n✅ INDEXING COMPLETED SUCCESSFULLY!")
print(f"📊 Total stored chunks: {len(all_chunks)}")