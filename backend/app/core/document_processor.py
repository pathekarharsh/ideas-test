import os
from typing import List, Dict, Any
from pathlib import Path
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from loguru import logger

class DocumentProcessor:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        supported_extensions: List[str] = [".pdf"]
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.supported_extensions = supported_extensions
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False
        )

    def process_document(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Process a single document and return chunks with metadata.
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            file_extension = Path(file_path).suffix.lower()
            if file_extension not in self.supported_extensions:
                raise ValueError(f"Unsupported file type: {file_extension}")

            # Load document using PyMuPDF
            loader = PyMuPDFLoader(file_path)
            pages = loader.load()

            # Process each page and create chunks
            chunks = []
            for page in pages:
                # Add metadata
                metadata = {
                    "source": file_path,
                    "page": page.metadata.get("page", 0),
                    "total_pages": len(pages),
                    "file_name": os.path.basename(file_path),
                    "program_type": self._get_program_type(file_path),
                    "year": "2025"  # For future admissions
                }

                # Split text into chunks
                page_chunks = self.text_splitter.split_text(page.page_content)
                
                # Create chunk documents with metadata
                for i, chunk in enumerate(page_chunks):
                    chunk_doc = {
                        "content": chunk,
                        "metadata": {
                            **metadata,
                            "chunk_index": i,
                            "total_chunks": len(page_chunks)
                        }
                    }
                    chunks.append(chunk_doc)

            logger.info(f"Successfully processed {file_path}: {len(chunks)} chunks created")
            return chunks

        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            raise

    def _get_program_type(self, file_path: str) -> str:
        """
        Determine the program type from the filename.
        """
        filename = os.path.basename(file_path).lower()
        if "ug" in filename or "undergraduate" in filename:
            return "UG"
        elif "mtech" in filename:
            return "MTech"
        elif "phd" in filename:
            return "PhD"
        else:
            return "General"

    def process_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """
        Process all supported documents in a directory.
        """
        all_chunks = []
        try:
            for root, _, files in os.walk(directory_path):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in self.supported_extensions):
                        file_path = os.path.join(root, file)
                        chunks = self.process_document(file_path)
                        all_chunks.extend(chunks)

            logger.info(f"Processed {len(all_chunks)} total chunks from directory {directory_path}")
            return all_chunks

        except Exception as e:
            logger.error(f"Error processing directory {directory_path}: {str(e)}")
            raise

if __name__ == "__main__":
    # Example usage
    processor = DocumentProcessor()
    # Get the absolute path to the data directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "..", "data"))
    logger.info(f"Processing documents from: {data_dir}")
    chunks = processor.process_directory(data_dir)
    print(f"Processed {len(chunks)} chunks from all documents") 