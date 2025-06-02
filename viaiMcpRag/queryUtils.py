# # import os
# # from typing import Dict, Any
# # from pymongo import MongoClient
# # from fastapi import HTTPException
# # from vectorstore_utils import unzip_to_memory
# # # --- Constants ---
# # MONGO_URI = os.getenv("MONGODB_URI")
# # DB_NAME = os.getenv("DB_NAME")
# # COLLECTION_NAME = os.getenv("COLLECTION_NAME")

# # # --- MongoDB Retrieval Function ---
# # def get_vector_store_by_user_and_folder(user_id: str, folder_id: str) -> Dict[str, Any]:
# #     print(f"\n[GetMongoDB] Fetching vector store with user_id: {user_id}, folder_id: {folder_id}")
# #     client = MongoClient(MONGO_URI)
# #     db = client[DB_NAME]
# #     collection = db[COLLECTION_NAME]
# #     document = collection.find_one({"user_id": user_id, "folder_id": folder_id})
# #     client.close()
# #     if not document:
# #         print("[GetMongoDB] Error: Vector store not found")
# #         raise ValueError("Vector store not found")
# #     print("[GetMongoDB] Successfully retrieved vector store")
# #     return document

# # # --- Download and Save Function ---
# # def download_and_save_vector_store_by_folder(user_id: str, folder_id: str):
# #     print(f"\n[Download Process] Starting download for user_id: {user_id}, folder_id: {folder_id}")
# #     try:
# #         print("[Download Process] Fetching vector store from MongoDB using user_id and folder_id")
# #         document = get_vector_store_by_user_and_folder(user_id, folder_id)
# #         if not document:
# #             print("[Download Process] Error: No document found for user_id and folder_id")
# #             raise HTTPException(status_code=404, detail="No document found for provided user_id and folder_id")

# #         zip_data = document.get("vector_store")
# #         if not zip_data:
# #             print("[Download Process] Error: No zip data found in document")
# #             raise HTTPException(status_code=404, detail="No zip data found in document")

# #         metadata = document.get("metadata", {})
# #         vectorstore_info = metadata.get("vectorstore_info", {})
# #         persist_dir = vectorstore_info.get("persist_dir", folder_id)
# #         print(f"[Download Process] Found vector store with persist_dir: {persist_dir}")

# #         VECTOR_STORE_BASE_DIR = os.getenv("VECTOR_STORE_BASE_DIR", "./vectorstores")
# #         extract_path = os.path.join(VECTOR_STORE_BASE_DIR, user_id, persist_dir)
# #         print(f"[Download Process] Extracting to path: {extract_path}")
        
# #         saved_path = unzip_to_memory(zip_data, extract_path)
# #         print(f"[Download Process] Successfully extracted to: {saved_path}")

# #         return {
# #             "status": "success",
# #             "saved_path": saved_path,
# #             "metadata": metadata
# #         }
# #     except Exception as e:
# #         print(f"[Download Process] Error occurred: {str(e)}")
# #         raise HTTPException(status_code=500, detail=str(e))

# # # --- Static Main Function ---
# # def main():
# #     user_id = "12345"
# #     folder_id = "my1"
    
# #     result = download_and_save_vector_store_by_folder(user_id, folder_id)
# #     print("\n[Result]")
# #     print(result)

# # if __name__ == "__main__":
# #     main()


# import time
# import logging
# from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings
# # Setup logging
# log = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)

# # Path to your saved FAISS vector store
# vector_store_path = "/Users/macbook/Desktop/MCP_S copy/viaiMcpRag/vector_stores/12367/VDB_1748430604605-sample%20%281%29_my1276"

# # Choose the same embedding model you used to create the vector store
# embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# # Load the FAISS vector store from disk
# start_time = time.time()
# vectorstore = FAISS.load_local(vector_store_path, embeddings=embedding_model,allow_dangerous_deserialization = True)
# retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
# log.info(f"Vector store ready in {time.time() - start_time:.2f} seconds.")

# # Define the retrieval function
# def retrieve_doc(query: str) -> str:
#     log.info(f"Tool call: retrieve_doc(query='{query}')")
#     start = time.time()

#     # Retrieve relevant documents
#     relevant_docs = retriever.invoke(query)
#     if not relevant_docs:
#         return "No relevant documents found for your query."

#     duration = time.time() - start
#     log.info(f"Retrieved {len(relevant_docs)} documents in {duration:.2f} seconds.")

#     # Format the output
#     return "\n\n".join([
#         f"==DOCUMENT {i+1}==\nSource: {doc.metadata.get('source', 'Unknown')}\n\n{doc.page_content}"
#         for i, doc in enumerate(relevant_docs)
#     ])

# if __name__ == "__main__":
#     query = "this document is about"
#     result = retrieve_doc(query)
#     print(result)



import os
import io
import time
import zipfile
import logging
from typing import Dict, Any

from pymongo import MongoClient
from fastapi import HTTPException
from vectorstore_utils import unzip_to_memory

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# -------------------------
# Configuration
# -------------------------
MONGO_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("DB_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_STORE_BASE_DIR = os.getenv("VECTOR_STORE_BASE_DIR", "./vectorstores")

# -------------------------
# Logging
# -------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("VectorPipeline")

# -------------------------
# MongoDB Fetch
# -------------------------
def get_vector_store_by_user_and_folder(user_id: str, folder_id: str) -> Dict[str, Any]:
    log.info(f"Fetching vector store for user_id={user_id}, folder_id={folder_id}")
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    document = collection.find_one({"user_id": user_id, "folder_id": folder_id})
    client.close()
    if not document:
        raise ValueError("Vector store not found in MongoDB.")
    return document

# -------------------------
# Download and Save
# -------------------------
def download_and_save_vector_store_by_folder(user_id: str, folder_id: str) -> str:
    try:
        document = get_vector_store_by_user_and_folder(user_id, folder_id)
        zip_data = document.get("vector_store")
        if not zip_data:
            raise ValueError("No zip data found in document")

        metadata = document.get("metadata", {})
        vectorstore_info = metadata.get("vectorstore_info", {})
        persist_dir = vectorstore_info.get("persist_dir", folder_id)
        extract_path = os.path.join(VECTOR_STORE_BASE_DIR, user_id, persist_dir)

        saved_path = unzip_to_memory(zip_data, extract_path)
        return saved_path
    except Exception as e:
        log.error(f"Download error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------
# Load Vector Store
# -------------------------
def load_vector_store(vector_store_path: str):
    log.info(f"Loading FAISS vector store from: {vector_store_path}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectorstore = FAISS.load_local(vector_store_path, embeddings=embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return retriever

# -------------------------
# Retrieve Documents
# -------------------------
def retrieve_documents(query: str, retriever) -> str:
    log.info(f"Running query: '{query}'")
    start = time.time()
    docs = retriever.invoke(query)
    duration = time.time() - start
    log.info(f"Retrieved {len(docs)} documents in {duration:.2f} seconds.")

    if not docs:
        return "No relevant documents found for your query."

    return "\n\n".join([
        f"==DOCUMENT {i + 1}==\nSource: {doc.metadata.get('source', 'Unknown')}\n\n{doc.page_content}"
        for i, doc in enumerate(docs)
    ])

# -------------------------
# Main Execution
# -------------------------
def main():
    user_id = "12345"
    folder_id = "my1"
    query = "this document is about"

    try:
        vector_store_path = download_and_save_vector_store_by_folder(user_id, folder_id)
        retriever = load_vector_store(vector_store_path)
        result = retrieve_documents(query, retriever)

        print("\n=== Retrieved Documents ===\n")
        print(result)

    except Exception as e:
        log.error(f"Failed: {e}")

if __name__ == "__main__":
    main()
