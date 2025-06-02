from fastapi import FastAPI, UploadFile, File, Form, HTTPException,Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import os, shutil
from langchain.docstore.document import Document

from fileUploader_Utils import FileUploader
from vectorstore_utils import (
    VectorStoreBuilder,
    store_in_mongodb,
    get_vector_store_from_mongodb,
    unzip_to_memory
)
from queryResponse import (
    download_and_save_vector_store_by_folder,
    load_vector_store,
    pipeline_query_with_groq,
    batch_process_queries
)

app = FastAPI()

# ============================
# Upload Endpoint
# ============================
@app.post("/upload/")
async def upload_and_process(
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None),
    user_id: str = Form(...),
    folder_id: str = Form(...)
):
    print(f"\n[Upload Process] Starting upload process for user_id: {user_id}, folder_id: {folder_id}")
    if not file and not url:
        print("[Upload Process] Error: Neither file nor URL provided")
        raise HTTPException(status_code=400, detail="Either file or URL must be provided.") 
    try:
        uploader = FileUploader()

        if file:
            print(f"[Upload Process] Processing file: {file.filename}")
            temp_path = file.filename
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            text = uploader.load_file(temp_path)
            os.remove(temp_path)
            source_name = file.filename
            print(f"[Upload Process] File processed successfully: {source_name}")

        else:
            print(f"[Upload Process] Processing URL: {url}")
            text = uploader.extract_text_from_url(url)
            source_name = os.path.basename(url)
            print(f"[Upload Process] URL processed successfully: {source_name}")

        print("[Upload Process] Creating vector store")
        builder = VectorStoreBuilder(file_name=source_name, folder_id=folder_id)
        document = Document(page_content=text, metadata={"source": source_name})
        split_docs = builder.split_documents([document])
        print(f"[Upload Process] Documents split into {len(split_docs)} chunks")
        
        builder.create_vectorstore(split_docs)
        print("[Upload Process] Vector store created successfully")

        vectorstore_info = builder.get_vectorstore_info()
        print(f"[Upload Process] Vector store info: uploading...")
        
        zip_data = builder.create_zip_archive()
        print("[Upload Process] Created zip archive of vector store")

        print("[Upload Process] Storing in MongoDB")
        mongo_id = store_in_mongodb(
            zip_data=zip_data,
            user_id=user_id,
            folder_id=folder_id,
            metadata={
                "vectorstore_info": vectorstore_info,
                "document_info": {
                    "source": source_name,
                    "chunks": len(split_docs),
                    "chunk_size": builder.chunk_size,
                    "chunk_overlap": builder.chunk_overlap
                }
            }
        )
        print(f"[Upload Process] Stored in MongoDB with ID: {mongo_id}")

        persist_path = os.path.join(os.getcwd(), builder.persist_dir)
        if os.path.exists(persist_path):
            shutil.rmtree(persist_path)
            print(f"[Upload Process] Cleaned up temporary directory: {persist_path}")

        return JSONResponse(content={
            "status": "success",
            "mongo_id": mongo_id,
            "user_id": user_id,
            "folder_id": folder_id,
            "vectorstore_info": vectorstore_info
        })

    except Exception as e:
        print(f"[Upload Process] Error occurred: {str(e)}")
        if 'builder' in locals() and hasattr(builder, 'persist_dir'):
            persist_path = os.path.join(os.getcwd(), builder.persist_dir)
            if os.path.exists(persist_path):
                shutil.rmtree(persist_path)
                print(f"[Upload Process] Cleaned up temporary directory after error: {persist_path}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================
# Download Vector Store Endpoint
# ============================
@app.get("/download-and-save/{mongo_id}")
async def download_and_save_vector_store(mongo_id: str, user_id: str):
    print(f"\n[Download Process] Starting download for mongo_id: {mongo_id}, user_id: {user_id}")
    try:
        print("[Download Process] Fetching vector store from MongoDB")
        document = get_vector_store_from_mongodb(mongo_id, user_id)
        zip_data = document.get("vector_store")
        if not zip_data:
            print("[Download Process] Error: No zip data found in DB")
            raise HTTPException(status_code=404, detail="No zip data found in DB")

        metadata = document.get("metadata", {})
        vectorstore_info = metadata.get("vectorstore_info", {})
        persist_dir = vectorstore_info.get("persist_dir", "vector_store")
        print(f"[Download Process] Found vector store with persist_dir: {persist_dir}")

        VECTOR_STORE_BASE_DIR = os.getenv("VECTOR_STORE_BASE_DIR", "./vectorstores")
        extract_path = os.path.join(VECTOR_STORE_BASE_DIR, user_id, persist_dir)
        print(f"[Download Process] Extracting to path: {extract_path}")
        
        saved_path = unzip_to_memory(zip_data, extract_path)
        print(f"[Download Process] Successfully extracted to: {saved_path}")

        return {
            "status": "success",
            "saved_path": saved_path,
            "metadata": metadata
        }
    except Exception as e:
        print(f"[Download Process] Error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================
# Get Vector Store Info Endpoint
# ============================
@app.get("/vector-store-info/{mongo_id}")
async def get_vector_store_info(mongo_id: str, user_id: str):
    print(f"\n[Info Process] Fetching info for mongo_id: {mongo_id}, user_id: {user_id}")
    try:
        print("[Info Process] Fetching vector store from MongoDB")
        document = get_vector_store_from_mongodb(mongo_id, user_id)
        print("[Info Process] Successfully retrieved vector store info")
        return {
            "status": "success",
            "user_id": document["user_id"],
            "folder_id": document["folder_id"],
            "created_at": document["created_at"],
            "metadata": document["metadata"]
        }
    except Exception as e:
        print(f"[Info Process] Error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================
# Query Vector Store (JSON Input)
# ============================
from fastapi.exceptions import RequestValidationError
from fastapi.exception_handlers import request_validation_exception_handler
import json
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # Prepare a dict with just the URL
    url_dict = {"url": str(request.url)}
    print("\n[Validation Error] on request:")
    print(json.dumps(url_dict, indent=4))
    raw_body = await request.body()
    try:
        print(f"Raw request body (str): {raw_body.decode('utf-8')}")
    except Exception:
        print("Cannot decode request body as UTF-8")
    print("Validation errors:")
    print(exc.errors())
    
    return await request_validation_exception_handler(request, exc)

class QueryRequest(BaseModel):
    query: str
    user_id: str
    folder_id: str
    batch_mode: bool = False

@app.post("/query/")
async def query_documents(request: QueryRequest):
    # Print expected schema
    print("\n[Expected Schema]")
    print({
        "query": "string",
        "user_id": "string",
        "folder_id": "string",
        "batch_mode": "boolean"
    })

    # Print actual received values
    print("\n[Received Request Body]")
    print(request.dict())  # or use request.json() if it's a raw body

    # Proceed with original logic
    query = request.query
    user_id = request.user_id
    folder_id = request.folder_id
    batch_mode = request.batch_mode

    print(f"\n[Query Process] Starting query process for user_id: {user_id}, folder_id: {folder_id}")
    try:
        print("[Query Process] Downloading and saving vector store")
        vector_store_path = await download_and_save_vector_store_by_folder(user_id, folder_id)
        
        print("[Query Process] Loading vector store")
        retriever = load_vector_store(vector_store_path)
        
        if batch_mode:
            print("[Query Process] Processing batch queries")
            queries = query.split('\n')
            results = await batch_process_queries(retriever, queries, user_id, folder_id)
        else:
            print("[Query Process] Processing single query")
            results = await pipeline_query_with_groq(retriever, query, user_id, folder_id)
        
        print("[Query Process] Query completed successfully")
        return JSONResponse(content=results)
        
    except Exception as e:
        print(f"[Query Process] Error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    import os

    port = int(os.environ.get("PORT", 8000))  # Use PORT from env or default to 8000
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)