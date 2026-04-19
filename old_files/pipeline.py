import hashlib
import math
import os
import sys
import types

import chromadb
import fitz
import pdfplumber
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Skip model hoster connectivity checks in restricted networks.
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except Exception:
    RecursiveCharacterTextSplitter = None


def ensure_langchain_docstore_compat():
    """Provide compatibility for packages that still import langchain.docstore."""
    try:
        import langchain.docstore.document  # noqa: F401
        return
    except Exception:
        pass

    try:
        from langchain_core.documents import Document
    except Exception:
        return

    if "langchain.docstore" not in sys.modules:
        sys.modules["langchain.docstore"] = types.ModuleType("langchain.docstore")

    doc_module = types.ModuleType("langchain.docstore.document")
    doc_module.Document = Document
    sys.modules["langchain.docstore.document"] = doc_module


def ensure_langchain_text_splitter_compat():
    """Provide compatibility for packages that still import langchain.text_splitter."""
    try:
        import langchain.text_splitter  # noqa: F401
        return
    except Exception:
        pass

    class CompatRecursiveCharacterTextSplitter:
        def __init__(self, chunk_size=800, chunk_overlap=150):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap

        def split_text(self, text):
            step = max(1, self.chunk_size - self.chunk_overlap)
            chunks = []
            for start in range(0, len(text), step):
                chunk = text[start:start + self.chunk_size].strip()
                if chunk:
                    chunks.append(chunk)
            return chunks

    splitter_module = types.ModuleType("langchain.text_splitter")
    splitter_module.RecursiveCharacterTextSplitter = CompatRecursiveCharacterTextSplitter
    sys.modules["langchain.text_splitter"] = splitter_module


ensure_langchain_docstore_compat()
ensure_langchain_text_splitter_compat()

# ---- PaddleOCR import & CPU-only, deprecation-safe initialization ----
try:
    from paddleocr import PaddleOCR
    OCR_IMPORT_ERROR = None
except Exception as exc:
    PaddleOCR = None
    OCR_IMPORT_ERROR = exc

# (Optional) Hint Paddle to use CPU if available
try:
    import paddle
    try:
        paddle.set_device("cpu")
    except Exception:
        pass
except Exception:
    paddle = None

PDF_FOLDER = "data/pdfs"
IMAGE_FOLDER = "output/images"
VECTOR_DB_PATH = "vector_db"
COLLECTION_NAME = "aveva_docs"
MIN_EXTRACTED_TEXT = 100
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs(VECTOR_DB_PATH, exist_ok=True)

# Initialize OCR (CPU-only, version-tolerant)
def init_ocr():
    if PaddleOCR is None:
        return None
    # Newer paddlex-based API: use device="cpu" + use_textline_orientation
    try:
        return PaddleOCR(use_textline_orientation=True, lang="en", device="cpu")
    except Exception:
        # Older API: fall back to use_angle_cls + use_gpu flag
        try:
            return PaddleOCR(use_angle_cls=True, lang="en", use_gpu=False)
        except Exception as e:
            print(f"Warning: failed to initialize PaddleOCR: {e}")
            return None

ocr = init_ocr()
if ocr is None:
    print(f"Warning: OCR disabled due to import error: {OCR_IMPORT_ERROR}")

# Embedding model
try:
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as exc:
    embedding_model = None
    print(f"Warning: embedding model unavailable, using fallback embeddings: {exc}")

# Persistent Vector DB
client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
collection = client.get_or_create_collection(
    COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)

if RecursiveCharacterTextSplitter is not None:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
else:
    splitter = None


def extract_text_pdf(pdf_path):
    text_content = []

    # Fast path: extract page text with PyMuPDF first.
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text = page.get_text("text")
                if text and text.strip():
                    text_content.append(text)
    except Exception as exc:
        print(f"Warning: fitz text extraction failed for {pdf_path}: {exc}")

    # Best-effort table extraction.
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables() or []
                for table in tables:
                    table_text = "\n".join(
                        [" | ".join("" if cell is None else str(cell) for cell in row) for row in table if row]
                    )
                    if table_text.strip():
                        text_content.append(table_text)
    except Exception as exc:
        print(f"Warning: pdfplumber table extraction failed for {pdf_path}: {exc}")

    return "\n".join(text_content)


def ocr_scanned_pages(pdf_path):
    if ocr is None:
        return ""

    ocr_text = []

    with fitz.open(pdf_path) as doc:
        for page_index in range(len(doc)):
            page = doc.load_page(page_index)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))

            safe_name = os.path.splitext(os.path.basename(pdf_path))[0].replace(" ", "_")
            img_path = os.path.join(IMAGE_FOLDER, f"{safe_name}_page_{page_index}.png")
            pix.save(img_path)

            try:
                # Newer API: no `cls` param; Older API: accepts `cls`
                result = ocr.ocr(img_path)
            except TypeError:
                # Backward compatibility
                result = ocr.ocr(img_path, cls=False)
            except Exception as exc:
                print(f"Warning: OCR failed for {img_path}: {exc}")
                continue

            if not result:
                continue

            lines = result[0] if isinstance(result, list) and len(result) > 0 else result
            page_lines = []
            for line in lines:
                if isinstance(line, (list, tuple)) and len(line) > 1:
                    line_info = line[1]
                    if isinstance(line_info, (list, tuple)) and len(line_info) > 0:
                        text = line_info[0]
                        if isinstance(text, str) and text.strip():
                            page_lines.append(text.strip())

            if page_lines:
                ocr_text.append(" ".join(page_lines))

    return "\n".join(ocr_text)


def chunk_text(text):
    if splitter is not None:
        return splitter.split_text(text)

    step = max(1, CHUNK_SIZE - CHUNK_OVERLAP)
    chunks = []
    for start in range(0, len(text), step):
        end = start + CHUNK_SIZE
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def fallback_embedding(text, dim=128):
    vec = [0.0] * dim
    for idx, byte_val in enumerate(text.encode("utf-8", errors="ignore")):
        vec[idx % dim] += byte_val / 255.0

    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


def store_embeddings(chunks, source):
    if not chunks:
        print(f"No chunks generated for {source}; skipping embedding storage.")
        return

    if embedding_model is not None:
        embeddings = embedding_model.encode(chunks, normalize_embeddings=True)
        embeddings = [emb.tolist() if hasattr(emb, "tolist") else emb for emb in embeddings]
    else:
        embeddings = [fallback_embedding(chunk) for chunk in chunks]

    ids = []
    metadatas = []
    for i, chunk in enumerate(chunks):
        digest = hashlib.md5(chunk.encode("utf-8", errors="ignore")).hexdigest()[:12]
        ids.append(f"{source}_{i}_{digest}")
        metadatas.append({"source": source, "chunk_index": i, "chunk_length": len(chunk)})

    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )


def process_pdf(pdf_path):
    print(f"Processing: {pdf_path}")

    text = extract_text_pdf(pdf_path)

    if len(text.strip()) < MIN_EXTRACTED_TEXT:
        print("Running OCR...")
        ocr_text = ocr_scanned_pages(pdf_path)
        text = "\n".join(part for part in [text, ocr_text] if part and part.strip())

    if not text.strip():
        print(f"No text extracted from {pdf_path}; skipping.")
        return

    chunks = chunk_text(text)
    store_embeddings(chunks, os.path.basename(pdf_path))


def run_pipeline():
    pdf_files = sorted([f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")])
    if not pdf_files:
        print(f"No PDF files found in {PDF_FOLDER}")
        return

    for file in tqdm(pdf_files):
        pdf_path = os.path.join(PDF_FOLDER, file)
        try:
            process_pdf(pdf_path)
        except Exception as exc:
            print(f"Error processing {pdf_path}: {exc}")


if __name__ == "__main__":
    run_pipeline()
    print(f"Pipeline complete. Collection '{COLLECTION_NAME}' now has {collection.count()} records.")