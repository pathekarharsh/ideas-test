from typing import List, Dict, Any
from groq import Groq
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings
from loguru import logger
import os
from dotenv import load_dotenv
import re

load_dotenv()

class QueryProcessor:
    def __init__(self, model_name: str = "meta-llama/llama-4-scout-17b-16e-instruct"):
        # Initialize Groq client
        self.client = Groq(
            api_key="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"
        )
        self.model_name = model_name
        
        # Initialize embeddings model (using HuggingFace's all-MiniLM-L6-v2)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            cache_folder=os.path.join(os.path.dirname(__file__), "cache")
        )
        
        # Define the prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful VNIT (Visvesvaraya National Institute of Technology) admission assistant. 
            Use the following context to answer the question. 
            If you cannot find the answer in the context, respond with:
            "Sorry, I don't have that information. Please contact the VNIT admission team at admissions@vnit.ac.in or visit the official website: https://vnit.ac.in"
            
            Important: Always mention that this information is for the academic year 2025-26.
            
            Context: {context}
            
            Question: {question}"""),
            ("human", "{question}")
        ])

    def format_context(self, documents: List[Dict[str, Any]]) -> str:
        """
        Format retrieved documents into a context string.
        """
        context_parts = []
        for doc in documents:
            source = doc["metadata"]["file_name"]
            page = doc["metadata"]["page"]
            content = doc["content"]
            program_type = doc["metadata"].get("program_type", "General")
            context_parts.append(f"Source: {source} (Page {page}, Program: {program_type})\n{content}\n")
        
        return "\n".join(context_parts)

    def process_query(
        self,
        query: str,
        documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Process a query using the RAG pipeline.
        """
        try:
            # Format context from retrieved documents
            context = self.format_context(documents)
            
            # Prepare the messages for the chat completion
            messages = [
                {
                    "role": "system",
                    "content": f"""You are a helpful VNIT (Visvesvaraya National Institute of Technology) admission assistant. 
                    Use the following context to answer the question. 
                    If you cannot find the answer in the context, respond with:
                    "Sorry, I don't have that information. Please contact the VNIT admission team at admissions@vnit.ac.in or visit the official website: https://vnit.ac.in"
                    
                    Important: Always mention that this information is for the academic year 2025-26.
                    
                    Context: {context}"""
                },
                {
                    "role": "user",
                    "content": query
                }
            ]
            
            # Get response from Groq
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model=self.model_name,
                temperature=0.1,
                max_tokens=1024
            )
            
            response = chat_completion.choices[0].message.content
            # Strip HTML tags from the response
            response = re.sub(r'<[^>]+>', '', response)
            
            # Calculate confidence based on document scores
            confidence = max([doc["score"] for doc in documents]) if documents else 0.0
            
            # Get source documents
            sources = [
                f"{doc['metadata']['file_name']} (Page {doc['metadata']['page']})"
                for doc in documents
            ]
            
            return {
                "answer": response,
                "sources": sources,
                "confidence": confidence
            }

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise

if __name__ == "__main__":
    # Example usage
    processor = QueryProcessor()
    test_documents = [
        {
            "content": "The admission process requires a minimum GPA of 3.0.",
            "metadata": {
                "file_name": "admission_requirements.pdf",
                "page": 1,
                "program_type": "UG"
            },
            "score": 0.85
        }
    ]
    result = processor.process_query(
        "What is the minimum GPA requirement?",
        test_documents
    )
    print(f"Answer: {result['answer']}")
    print(f"Sources: {result['sources']}")
    print(f"Confidence: {result['confidence']:.2f}") 