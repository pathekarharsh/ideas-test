from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from langchain_community.embeddings import HuggingFaceEmbeddings
from loguru import logger
import os
from dotenv import load_dotenv

load_dotenv()

class VectorStoreManager:
    def __init__(
        self,
        persist_directory: str = "backend/data/vector_store",
        collection_name: str = "admission_docs"
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Initialize HuggingFace embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to the vector store with their embeddings.
        """
        try:
            # Prepare documents for batch processing
            ids = []
            texts = []
            metadatas = []

            for i, doc in enumerate(documents):
                ids.append(f"doc_{i}")
                texts.append(doc["content"])
                metadatas.append(doc["metadata"])

            # Add documents to collection
            self.collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas
            )

            logger.info(f"Successfully added {len(documents)} documents to vector store")

        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            raise

    def similarity_search(
        self,
        query: str,
        k: int = 3,
        score_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents using the query.
        """
        try:
            # Get query embedding
            query_embedding = self.embeddings.embed_query(query)

            # Search in collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                include=["documents", "metadatas", "distances"]
            )

            # Process results
            documents = []
            for i in range(len(results["documents"][0])):
                distance = results["distances"][0][i]
                # Convert distance to similarity score (cosine similarity)
                similarity = 1 - distance
                
                if similarity >= score_threshold:
                    documents.append({
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "score": similarity
                    })

            logger.info(f"Found {len(documents)} relevant documents for query")
            return documents

        except Exception as e:
            logger.error(f"Error performing similarity search: {str(e)}")
            raise

    def clear_collection(self) -> None:
        """
        Clear all documents from the collection.
        """
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("Successfully cleared vector store collection")
        except Exception as e:
            logger.error(f"Error clearing vector store: {str(e)}")
            raise

if __name__ == "__main__":
    # Example usage
    vector_store = VectorStoreManager()
    # Test query
    results = vector_store.similarity_search("What are the admission requirements?")
    for doc in results:
        print(f"Score: {doc['score']:.2f}")
        print(f"Content: {doc['content'][:200]}...")
        print("---") 