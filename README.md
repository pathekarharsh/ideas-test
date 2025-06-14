# VNIT Admission Chatbot

A chatbot application that provides information about VNIT (Visvesvaraya National Institute of Technology) admissions using RAG (Retrieval-Augmented Generation) technology.

## Project Structure

```
vnit-admission-chatbot/
├── backend/
│   ├── app/
│   │   ├── core/
│   │   │   ├── cache/                 # Cache directory for embeddings
│   │   │   ├── document_processor.py  # Handles PDF document processing
│   │   │   ├── query_processor.py     # Processes user queries using RAG
│   │   │   └── vector_store.py        # Manages vector storage for documents
│   │   └── main.py                    # FastAPI backend server
│   └── data/                          # Directory for processed documents
├── frontend/
│   └── app.py                         # Streamlit frontend application
├── data/                              # Directory for raw PDF documents
├── requirements.txt                   # Python dependencies
└── README.md                          # Project documentation
```

## Setup Instructions

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start the backend server:
   ```bash
   python backend/app/main.py
   ```

4. In a new terminal, start the frontend:
   ```bash
   streamlit run frontend/app.py
   ```

## Features

- PDF Document Processing: Automatically processes VNIT admission-related PDF documents
- RAG-based Responses: Uses Retrieval-Augmented Generation for accurate answers
- Source Attribution: Provides sources for each response
- Confidence Scoring: Indicates confidence level for each response

## Technology Stack

- Backend:
  - FastAPI
  - Groq LLM
  - LangChain
  - HuggingFace Embeddings
  - FAISS Vector Store

- Frontend:
  - Streamlit
  - Python

## API Endpoints

- `POST /query`: Process user queries
  - Input: `{"query": "your question here"}`
  - Output: `{"answer": "response", "sources": ["source1", "source2"], "confidence": 0.95}`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
