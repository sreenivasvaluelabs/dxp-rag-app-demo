# Moment Platform DXP - RAG Application Demo

A **Retrieval Augmented Generation (RAG)** application built with Streamlit, ChromaDB, and Ollama, featuring the **Moment Platform DXP** branding and enterprise-ready functionality.

## 🚀 Features

- **📄 Document Processing**: Upload and process PDF, TXT, and DOCX files
- **🧠 Vector Search**: ChromaDB integration for semantic document retrieval
- **🤖 AI-Powered Responses**: Ollama LLM integration for intelligent Q&A
- **🎨 Professional UI**: Moment Platform DXP branded interface
- **💾 Persistent Storage**: Documents persist across sessions
- **⚡ Real-time Processing**: Live document chunking and embedding
- **🔍 Dual Mode**: Compare responses with and without RAG
- **📊 Analytics**: Collection statistics and vector counts

## 🛠️ Technology Stack

### Core Framework
- **Frontend**: Streamlit v1.31.1 (Python web framework for rapid prototyping)
- **Backend**: Python 3.9+ with async/await support
- **Vector Database**: ChromaDB v0.6.3 (High-performance vector similarity search)
- **Document Processing**: LlamaIndex v0.12.11 (RAG orchestration framework)
- **Containerization**: Docker & Docker Compose (Scalable deployment)

### AI/ML Components
- **LLM Orchestration**: Ollama v0.6.1 (Local LLM serving platform)
- **Primary LLM**: **qwen2.5-coder:1.5b** (1.5B parameters)
  - **Purpose**: Main conversational AI for question answering
  - **Strengths**: Code understanding, technical documentation, multilingual support
  - **Use Case**: Generating human-like responses to user queries
- **Fallback LLM**: **deepseek-r1:latest** (Larger model for complex queries)
  - **Purpose**: Advanced reasoning for complex document analysis
  - **Strengths**: Deep reasoning, mathematical concepts, research tasks
  - **Use Case**: Complex technical questions requiring multi-step reasoning

### Embedding & Vector Processing
- **Embedding Model**: **mxbai-embed-large** (Large-scale embeddings)
  - **Purpose**: Convert documents and queries into 1024-dimensional vectors
  - **Strengths**: High-quality semantic understanding, multilingual support
  - **Use Case**: Document similarity search and semantic retrieval
- **Vector Operations**: FAISS-compatible similarity search with cosine distance

### Document Processing Pipeline
- **PDF Processing**: pypdf v4.1.0 (Modern PDF parsing, Unicode support)
- **DOCX Processing**: python-docx v1.1.0 + docx2txt v0.8 (Dual extraction methods)
- **Text Processing**: Custom chunking with configurable overlap
- **Data Analysis**: pandas v2.2.1 (Collection analytics and statistics)

### Supporting Libraries
- **HTTP Client**: requests v2.32.0 (API communications)
- **Environment**: python-dotenv v1.0.1 (Configuration management)
- **Utilities**: numpy v1.26.4 (Mathematical operations)
- **Logging**: Enhanced error tracking and debugging

## 📋 Prerequisites

Before running this application, ensure you have:

1. **Python 3.9+** installed with pip package manager
2. **Ollama v0.6.1+** installed and running locally
   ```bash
   # Install Ollama (if not already installed)
   curl -fsSL https://ollama.com/install.sh | sh
   ```
3. **Required AI Models** downloaded via Ollama:
   ```bash
   # Primary LLM for conversational AI
   ollama pull qwen2.5-coder:1.5b
   
   # Embedding model for document vectorization  
   ollama pull mxbai-embed-large:latest
   
   # Optional: Advanced LLM for complex reasoning
   ollama pull deepseek-r1:latest
   ```
4. **System Requirements**:
   - **RAM**: 8GB minimum (16GB recommended for larger models)
   - **Storage**: 5GB free space for models and vector database
   - **CPU**: Multi-core processor (GPU optional but recommended)

## 🚀 Quick Start

### Option 1: Local Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/sreenivasvaluelabs/dxp-rag-app-demo.git
   cd dxp-rag-app-demo
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start Ollama** (in separate terminal):
   ```bash
   ollama serve
   ```

4. **Run the application**:
   ```bash
   streamlit run rag.py
   ```

5. **Access the app**: Open http://localhost:8501 in your browser

### Option 2: Docker Deployment

1. **Clone and navigate**:
   ```bash
   git clone https://github.com/sreenivasvaluelabs/dxp-rag-app-demo.git
   cd dxp-rag-app-demo
   ```

2. **Run with Docker Compose**:
   ```bash
   docker-compose up -d
   ```

3. **Access the app**: http://localhost:8501

## 📚 Usage Guide

### 1. **Document Upload**
   - Click "Browse Files" to select documents
   - Choose from PDF, TXT, or DOCX formats
   - Create a new collection or use existing one
   - Click "Upload and Index" to process documents

### 2. **Querying Documents**
   - Select a collection from the dropdown
   - Enter your question in the text box
   - Choose comparison mode (with/without RAG)
   - Click "Submit Query" to get AI responses

### 3. **Demo Data**
   - Click "Ingest Demo Data" for pre-loaded examples
   - Includes sample documents about Mars, Saturn, Solar System
   - Try example queries about planetary science

## ⚙️ RAG Architecture & Model Configuration

### Document Processing Pipeline
```
PDF/DOCX/TXT → Text Extraction → Chunking (256 chars) → Embeddings → ChromaDB Storage
```

### Query Processing Flow  
```
User Query → Embedding → Vector Search → Context Retrieval → LLM Generation → Response
```

### Model Configuration
- **LLM Model**: qwen2.5-coder:1.5b (or deepseek-r1:latest)
  - **Context Window**: 4,096 tokens
  - **Output Tokens**: 256 tokens per response
  - **Temperature**: Optimized for factual responses
- **Embedding Model**: mxbai-embed-large:latest
  - **Dimensions**: 1024-dimensional vectors
  - **Max Input**: 512 tokens per chunk
  - **Similarity**: Cosine distance matching
- **Chunk Configuration**: 
  - **Size**: 256 characters (configurable 64-512)
  - **Overlap**: 32 characters (prevents context loss)
  - **Top-K Retrieval**: 4 most similar documents per query

### Advanced Settings
Access the "Advanced Settings" panel to modify:
- Document parsing chunk size and overlap
- Number of similar documents retrieved (similarity_top_k)
- LLM context window and output length
- Search similarity thresholds and embedding parameters

## 📱 API Endpoints

The application runs on Streamlit but includes these key functionalities:
- Document upload and processing
- Vector similarity search
- LLM query processing
- Collection management

## 🐳 Docker Configuration

```yaml
# docker-compose.yml structure
services:
  rag-app:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./chroma_db:/app/chroma_db
    environment:
      - OLLAMA_BASE_URL=http://host.docker.internal:11434
```

## 📊 Project Structure

```
dxp-rag-app-demo/
├── rag.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── Dockerfile            # Container configuration
├── docker-compose.yml    # Multi-service deployment
├── demo_docs/            # Sample PDF documents
├── chroma_db/           # Persistent vector database
├── help_content.py      # Help documentation
├── tutorial_content.py  # User tutorial content
└── README.md           # Project documentation
```

## 🔧 Troubleshooting

### Common Issues:

1. **"No collections available"**
   - Ensure documents are uploaded successfully
   - Check ChromaDB persistence in `chroma_db/` folder

2. **"Model not found" errors**
   - Verify Ollama is running: `ollama list`
   - Download required models if missing

3. **Embedding errors**
   - Reduce chunk size in Advanced Settings
   - Check Ollama service status

4. **Slow performance**
   - Use smaller embedding models
   - Reduce number of retrieved documents
   - Optimize chunk size for your content

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -m 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏢 About Moment Platform DXP

**Moment Platform DXP** delivers AI-powered digital solutions for enterprise document intelligence and conversational AI applications.

## 📞 Support

For questions or issues:
- Create an issue in this repository
- Contact: [+91 9603668136/sreenivas.valuelabs@gmail.com]

---

**Built with ❤️ by Sreeni**
