import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from loguru import logger

# Import RAG components
from app.core.vector_store import VectorStoreManager
from app.core.query_processor import QueryProcessor
from app.core.document_processor import DocumentProcessor

# Initialize FastAPI app
app = FastAPI(
    title="College Admission Chatbot API",
    description="API for handling admission-related queries using RAG architecture",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request and Response models
class Query(BaseModel):
    question: str
    context: Optional[str] = None  # For future use in tracking conversation context

class Response(BaseModel):
    answer: str
    sources: List[str]
    confidence: float

# Initialize RAG pipeline components
vector_store = VectorStoreManager()
query_processor = QueryProcessor()

def initialize_data():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
    logger.info(f"Processing and indexing documents from: {data_dir}")
    processor = DocumentProcessor()
    chunks = processor.process_directory(data_dir)
    vector_store.clear_collection()
    vector_store.add_documents(chunks)
    logger.info(f"Indexed {len(chunks)} chunks from all documents.")

@app.on_event("startup")
def on_startup():
    initialize_data()

@app.get("/")
async def root():
    return {"message": "College Admission Chatbot API is running"}

@app.post("/query", response_model=Response)
async def process_query(query: Query):
    try:
        # Retrieve relevant documents
        docs = vector_store.similarity_search(query.question, k=5, score_threshold=0.6)
        # Generate answer using RAG pipeline
        result = query_processor.process_query(query.question, docs)
        return Response(
            answer=result["answer"],
            sources=result["sources"],
            confidence=result["confidence"]
        )
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 