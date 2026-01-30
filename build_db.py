import json
import os

from local_loader import load_txt_files
from splitter import split_documents
from langchain_community.embeddings import HuggingFaceEmbeddings
from vector_store import create_vector_db, DEFAULT_EMBEDDING_MODEL

STORE_DIR = "store"
CHROMA_COLLECTION = "chroma"
CHUNK_TEXTS_PATH = os.path.join(STORE_DIR, "chunk_texts.json")


def main():
    os.makedirs(STORE_DIR, exist_ok=True)
    print("=== 1. load_txt_files ===")
    docs = load_txt_files()
    print(f"  loaded {len(docs)} documents")

    print("=== 2. split_documents ===")
    texts = split_documents(docs)
    print(f"  got {len(texts)} chunks")

    print("=== 3. HuggingFaceEmbeddings + Chroma add_documents ===")
    embeddings = HuggingFaceEmbeddings(model_name=DEFAULT_EMBEDDING_MODEL)
    create_vector_db(texts, embeddings=embeddings, collection_name=CHROMA_COLLECTION)
    print("  Chroma 已写入 store/chroma")

    chunk_texts = [t.page_content for t in texts]
    with open(CHUNK_TEXTS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunk_texts, f, ensure_ascii=False, indent=0)
    print(f"  chunk 文本已写入 {CHUNK_TEXTS_PATH}")

    print("=== 建库完成 ===")


if __name__ == "__main__":
    main()
