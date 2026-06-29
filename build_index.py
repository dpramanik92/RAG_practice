import re

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_PATH = "data/Indias_Ancient_history_full.pdf"
INDEX_PATH = "data/faiss_index"
SKIP_PAGES = 14
CHAPTER_PATTERN = r"(\d+)\.\s*\n([^\n]+)"
NUM_CHAPTERS = 33
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
EMBED_MODEL = "BAAI/bge-small-en-v1.5"


def build_and_save_index():
    print(f"Loading PDF: {PDF_PATH}")
    loader = PyMuPDFLoader(PDF_PATH)
    documents = loader.load()

    full_text = " ".join([doc.page_content for doc in documents])
    full_text_shorten = " ".join([doc.page_content for doc in documents[SKIP_PAGES:]])

    matches = list(re.finditer(CHAPTER_PATTERN, full_text))[:NUM_CHAPTERS]
    chapters = [m.group(2) for m in matches]
    print(f"Found {len(chapters)} chapters")

    positions = []
    for chapter in chapters:
        match = re.search(re.escape(chapter), full_text_shorten.replace("\n", " "), re.IGNORECASE)
        if match:
            positions.append((chapter, match.start()))

    chapters_text = []
    for i, (chapter_name, start) in enumerate(positions):
        end = positions[i + 1][1] if i < len(positions) - 1 else len(full_text_shorten)
        text = full_text_shorten[start:end].strip()
        chapters_text.append({"chapter": chapter_name, "text": text})

    docs = [
        Document(page_content=ch["text"], metadata={"chapter": ch["chapter"]})
        for ch in chapters_text
        if ch["text"]
    ]

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks")

    print(f"Embedding with {EMBED_MODEL}…")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    vectorstore.save_local(INDEX_PATH)
    print(f"Index saved to {INDEX_PATH}/")


if __name__ == "__main__":
    build_and_save_index()
