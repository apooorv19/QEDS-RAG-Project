import os
import shutil
import pickle
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from colorama import Fore, init

init(autoreset=True)

# Define Paths
DATA_FOLDER = "./data"  # <--- Points to your new folder
DB_PATH = "./chroma_db_advanced"

def main():
    # 1. Load all txt files from the data folder
    documents = []
    if not os.path.exists(DATA_FOLDER):
        print(f"❌ Error: Data folder '{DATA_FOLDER}' not found.")
        return

    for file in os.listdir(DATA_FOLDER):
        if file.endswith(".txt"):
            file_path = os.path.join(DATA_FOLDER, file)
            loader = TextLoader(file_path, encoding="utf-8")
            documents.extend(loader.load())

    print(f"✅ Loaded {len(documents)} documents.")

    # 2. Split chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    # 3. Create/Update Vector DB
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3", model_kwargs={"device": "cpu"})
    
    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=DB_PATH
    )
    print("✅ Vector Database Created Successfully!")

if __name__ == "__main__":
    main()