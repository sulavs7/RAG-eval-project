import os
import re
import glob

from dotenv import load_dotenv
# Import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()

DATA_DIR = "data"
DB_DIR = "chroma_store"

# 1. LOAD ---- read each transcript
def load_transcripts():
    docs = []
    for path in glob.glob(f"{DATA_DIR}/*.vtt"):
        lines = []
        for line in open(path):
            line = line.strip()
            if not line or line == "WEBVTT" or "-->" in line:
                continue
            lines.append(line)
        text = " ".join(lines)

        session = re.search(r"Session[ _]*(\d+)", path).group(1)
        docs.append(Document(page_content=text, metadata={"session": session}))
    return docs

# 2. BUILD ---- chunk, embed, and store
def load_store():
    
    embeddings = HuggingFaceEndpointEmbeddings(
    model="BAAI/bge-small-en-v1.5",
)

    if os.path.exists(DB_DIR):
        return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

    docs = load_transcripts()

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=750,
        chunk_overlap=100,
    ).split_documents(docs)

    # This will download the model once (~100MB) and cache it
    return Chroma.from_documents(chunks, embeddings, persist_directory=DB_DIR)

def build_retriever():
    return load_store().as_retriever(search_kwargs={"k": 5})

# 3. TRY IT
if __name__ == "__main__":
    retriever = build_retriever()
    results = retriever.invoke("what is regression testing?")
    
    for r in results:
        print(f"[Session {r.metadata['session']}] {r.page_content[:150]}...\n")   