import streamlit as st
import requests
import os
import base64
import warnings
import logging
import time
from typing import List
from help_content import HELP_CONTENT
from tutorial_content import TUTORIAL_CONTENT

# Setup comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Comprehensive error suppression
warnings.filterwarnings("ignore")
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
os.environ['CHROMA_TELEMETRY'] = 'False'

# Suppress ChromaDB and other library logs completely
logging.getLogger('chromadb').setLevel(logging.CRITICAL)
logging.getLogger('chromadb.telemetry').setLevel(logging.CRITICAL)
logging.getLogger('chromadb.telemetry.product.posthog').setLevel(logging.CRITICAL)
logging.getLogger('PyPDF2').setLevel(logging.ERROR)
logging.getLogger('pydantic').setLevel(logging.ERROR)
logging.getLogger('httpx').setLevel(logging.ERROR)

import chromadb
from chromadb.config import Settings as ChromaSettings

# Additional telemetry suppression
try:
    import chromadb.telemetry.product.posthog
    # Completely disable the telemetry module
    chromadb.telemetry.product.posthog.capture = lambda *args, **kwargs: None
except:
    pass

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

# Custom CSS for Cognizant-inspired white and blue theme
st.markdown(
    """
<style>
/* Cognizant-inspired color scheme */
:root {
    --cognizant-blue: #0074D9;
    --cognizant-dark-blue: #005bb5;
    --cognizant-light-blue: #e6f3ff;
    --white: #ffffff;
}

.download-button {
    display: inline-block;
    padding: 8px 16px;
    background-color: var(--cognizant-blue);
    color: white;
    text-decoration: none;
    border-radius: 6px;
    margin: 5px 0;
    transition: background-color 0.3s ease;
}
.download-button:hover {
    background-color: var(--cognizant-dark-blue);
}

/* Streamlit button styling */
.stButton > button {
    width: 100%;
    margin: 0;
    background-color: var(--cognizant-blue) !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover {
    background-color: var(--cognizant-dark-blue) !important;
    transform: translateY(-1px);
    box-shadow: 0 2px 4px rgba(0,116,217,0.3);
}

/* Response and context boxes */
.response-box {
    border: 2px solid var(--cognizant-blue);
    border-radius: 12px;
    padding: 15px;
    margin: 15px 0;
    background-color: var(--white);
    box-shadow: 0 2px 8px rgba(0,116,217,0.1);
}
.context-box {
    border: 2px solid var(--cognizant-light-blue);
    border-radius: 12px;
    padding: 15px;
    margin: 15px 0;
    background-color: var(--cognizant-light-blue);
    box-shadow: 0 2px 8px rgba(0,116,217,0.05);
}

/* Sidebar styling */
.css-1d391kg {
    background-color: var(--white);
}

/* Header styling */
h1 {
    color: var(--cognizant-blue) !important;
    font-weight: 600 !important;
}

/* Selectbox and input styling */
.stSelectbox > div > div {
    border-color: var(--cognizant-blue) !important;
}
.stTextInput > div > div > input {
    border-color: var(--cognizant-blue) !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--cognizant-dark-blue) !important;
    box-shadow: 0 0 0 2px rgba(0,116,217,0.2) !important;
}

/* Checkbox styling */
.stCheckbox > label > div {
    background-color: var(--white) !important;
    border-color: var(--cognizant-blue) !important;
}

/* Success and info messages */
.stSuccess {
    background-color: var(--cognizant-light-blue) !important;
    color: var(--cognizant-dark-blue) !important;
}
</style>
""",
    unsafe_allow_html=True
)

# Fetch available Ollama models
def get_ollama_models():
    ollama_base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
    try:
        response = requests.get(f"{ollama_base_url}/api/tags")
        if response.status_code == 200:
            models = response.json()["models"]
            return [model["name"] for model in models]
        else:
            st.error(f"Failed to fetch models: {response.status_code}")
            return []
    except requests.RequestException as e:
        st.error(f"Error fetching models: {e}")
        return []

# Initialize ChromaDB client with telemetry disabled
def get_chroma_client():
    """ChromaDB client with persistent storage"""
    try:
        # Use persistent storage so data survives page reloads
        persist_path = "chroma_db"
        if not os.path.exists(persist_path):
            os.makedirs(persist_path)
        client = chromadb.PersistentClient(path=persist_path)
        return client
    except Exception as e:
        st.error(f"ChromaDB connection error: {e}")
        # Fallback to temporary storage
        try:
            client = chromadb.EphemeralClient()
            st.warning("⚠️ Using temporary storage - data will be lost on refresh!")
            return client
        except Exception as e2:
            st.error(f"ChromaDB fallback failed: {e2}")
            return None

# Retrieve Chroma collection names
def get_collections() -> List[str]:
    try:
        chroma_client = get_chroma_client()
        collections = chroma_client.list_collections()
        # ChromaDB v0.6.0+ returns CollectionName objects, extract the name 
        return [str(col) for col in collections]  # CollectionName objects have __str__ method
    except Exception:
        return []

# Create a new Chroma collection
def create_collection(name: str):
    # Validate and fix collection name
    original_name = name
    
    # Auto-fix common naming issues
    name = name.lower()  # Convert to lowercase
    name = name.replace(' ', '_')  # Replace spaces with underscores
    name = name.replace('.', '_')  # Replace dots with underscores  
    name = ''.join(c for c in name if c.isalnum() or c in '_-')  # Keep only allowed characters
    
    # Ensure it starts and ends with alphanumeric
    name = name.strip('_-')
    
    # Ensure minimum length
    if len(name) < 3:
        name = f"col_{name}"
    
    # Ensure maximum length  
    if len(name) > 63:
        name = name[:63]
    
    # Final validation
    if not name or len(name) < 3:
        raise ValueError("Collection name too short after cleaning")
    
    chroma_client = get_chroma_client()
    chroma_client.create_collection(name)
    
    return name, original_name

# Get ChromaDB statistics
def get_chromadb_stats():
    chroma_client = get_chroma_client()
    collections = chroma_client.list_collections()

    stats = {
        "num_collections": len(collections),
        "total_vectors": 0,
    }
    
    for coll_name in collections:
        collection_obj = chroma_client.get_collection(coll_name)
        stats["total_vectors"] += collection_obj.count()

    return stats

# Upload and index files into a specified collection
def upload_files(files, collection_name: str, chunk_size: int, chunk_overlap: int):
    try:
        with st.spinner('Processing your file... Please wait.'):
            # Get ChromaDB client and collection
            chroma_client = get_chroma_client()
            chroma_collection = chroma_client.get_or_create_collection(collection_name)
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
            all_documents = []
    
            # Process each file
            for file in files:
                try:
                    st.write(f"📄 Processing {file.name}...")
                    
                    # Save uploaded file temporarily
                    temp_file = f"temp_{file.name}"
                    with open(temp_file, "wb") as f:
                        f.write(file.getbuffer())
                        
                    # Read the file
                    documents = SimpleDirectoryReader(input_files=[temp_file]).load_data()
                    all_documents.extend(documents)
                    
                    # Remove temporary file
                    os.remove(temp_file)
                    
                    st.success(f"✅ {file.name}: {len(documents)} chunks loaded")
                    
                except Exception as e:
                    st.error(f"❌ Error with {file.name}: {str(e)}")
                    continue

            if not all_documents:
                st.error("No documents were processed successfully.")
                return None

            st.write(f"🧠 Creating embeddings for {len(all_documents)} chunks...")
            
            # Create index with all documents and better error handling
            try:
                index = VectorStoreIndex.from_documents(
                    all_documents,
                    storage_context=storage_context,
                    embed_model=Settings.embed_model,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
                st.success(f"🎉 Successfully indexed {len(all_documents)} chunks in collection '{collection_name}'!")
                return index
            except Exception as embed_error:
                if "input length exceeds" in str(embed_error) or "status code: 500" in str(embed_error):
                    st.error(f"❌ Text chunks too large for embedding model. Try reducing chunk size to 128-256.")
                else:
                    st.error(f"❌ Embedding error: {str(embed_error)}")
                return None

    except Exception as e:
        st.error(f"❌ Upload failed: {str(e)}")
        return None
        return None

# Show demo files with download buttons
def show_demo_files():
    st.header("Available Demo Files")
    demo_dir = "demo_docs"
    
    for filename in os.listdir(demo_dir):
        filepath = os.path.join(demo_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(filepath, 'r', encoding='latin-1') as f:
                content = f.read()
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**{filename}**")
        with col2:
            b64 = base64.b64encode(content.encode()).decode()
            href = f'data:text/plain;base64,{b64}'
            st.download_button(
                label="Download",
                data=content,
                file_name=filename,
                mime="text/plain",
                key=filename
            )

# Ingest demo documents into Chroma
def ingest_demo_data():
    start_time = time.time()
    logger.info("Starting demo data ingestion")
    
    with st.spinner('Ingesting demo data... This may take a few moments.'):
        chroma_client = get_chroma_client()
        logger.info("ChromaDB client initialized for demo data")

        demo_collection = chroma_client.get_or_create_collection("demo")
        vector_store = ChromaVectorStore(chroma_collection=demo_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        documents = SimpleDirectoryReader("demo_docs").load_data()
        logger.info(f"Loaded {len(documents)} demo documents")

        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            embed_model=Settings.embed_model,
        )
        
        total_time = time.time() - start_time
        logger.info(f"Demo data ingestion completed in {total_time:.2f} seconds")

        st.session_state.chromadb_stats = get_chromadb_stats()

        return index

# Main function to set up the Streamlit app
def main():
    st.title("Retrieval Augmented Generation")

    # Add logo to the sidebar
    with st.sidebar:
        # Moment Platform DXP branding with white background
        st.markdown("""
        <div style="text-align: center; padding: 20px; background: white; border-radius: 10px; margin-bottom: 20px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); border: 2px solid #0074D9;">
            <h1 style="color: #0074D9 !important; margin: 0; font-family: 'Arial', sans-serif; font-weight: 800; font-size: 20px; letter-spacing: 1px;">
                MOMENT PLATFORM DXP
            </h1>
            <p style="color: #005bb5 !important; margin: 8px 0 0 0; font-size: 14px; font-weight: 600;">
                By: Sreeni
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Controls for demo data and help/tutorial
    with st.sidebar:

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Ingest Demo Data", use_container_width=True):
                st.session_state.demo_index = ingest_demo_data()
                st.success("Demo data ingested successfully!")
                st.session_state.chromadb_stats = get_chromadb_stats()
        
        with col2:
            if st.button("Show Demo Files", use_container_width=True):
                st.session_state.show_demo_files = True

        col3, col4 = st.columns([1, 1])
        with col3:
            if st.button("Help", use_container_width=True):
                st.session_state.show_help = not st.session_state.get('show_help', False)
        with col4:
            if st.button("Tutorial", use_container_width=True):
                st.session_state.show_tutorial = not st.session_state.get('show_tutorial', False)

    if st.session_state.get('show_help', False):
        with st.expander("Help", expanded=True):
            st.markdown(HELP_CONTENT)
            if st.button("Close Help"):
                st.session_state.show_help = False

    if st.session_state.get('show_tutorial', False):
        with st.expander("Tutorial", expanded=True):
            st.markdown(TUTORIAL_CONTENT)
            if st.button("Close Tutorial"):
                st.session_state.show_tutorial = False

    if st.session_state.get('show_demo_files', False):
        show_demo_files()
        if st.button("Close Demo Files"):
            st.session_state.show_demo_files = False
        st.divider()

    # Fetch available models
    available_models = get_ollama_models()

    # Sidebar settings for LLM and embedding models
    st.sidebar.header("Settings")
    default_model_index = 0
    default_embed_index = 0
    
    # Set proper defaults - LLM should NOT be an embedding model
    chat_models = [m for m in available_models if not ('embed' in m.lower())]
    embed_models = [m for m in available_models if ('embed' in m.lower())]
    
    if not chat_models:
        chat_models = available_models  # Fallback to all models
    if not embed_models:
        embed_models = available_models  # Fallback to all models
    
    # Set better defaults
    if "qwen2.5-coder:1.5b" in chat_models:
        default_model_index = chat_models.index("qwen2.5-coder:1.5b")
    elif "deepseek-r1:latest" in chat_models:
        default_model_index = chat_models.index("deepseek-r1:latest")
        
    if "mxbai-embed-large:latest" in embed_models:
        default_embed_index = embed_models.index("mxbai-embed-large:latest")

    model = st.sidebar.selectbox("LLM Model", options=chat_models, index=default_model_index)
    embed_model = st.sidebar.selectbox("Embedding Model", options=embed_models, index=default_embed_index)

    # Advanced settings configuration
    with st.sidebar.expander("Advanced Settings"):
        similarity_top_k = st.number_input("Number of similar documents", value=4, min_value=1)
        context_window = st.number_input("Maximum input size to LLM", value=4096, min_value=1)
        num_output = st.number_input("Number of tokens for generation", value=256, min_value=1)
        chunk_size = st.number_input("Chunk size for document parsing", value=256, min_value=64, max_value=512)
        chunk_overlap = st.number_input("Chunk overlap for document parsing", value=32, min_value=0, max_value=128)

    # File upload and collection management
    st.sidebar.header("Document Upload")
    uploaded_files = st.sidebar.file_uploader("Choose files", type=["txt", "pdf", "docx"], accept_multiple_files=True)
    
    # Store uploaded files in session state for estimation
    if uploaded_files:
        st.session_state['uploaded_files'] = uploaded_files
        total_size = sum(len(file.getbuffer()) for file in uploaded_files)
        st.sidebar.info(f"📊 {len(uploaded_files)} files selected ({total_size/1024:.1f} KB total)")

    collections = get_collections()
    
    # Handle collection selection with proper state management
    if 'selected_collection' in st.session_state and st.session_state.selected_collection in collections:
        # If we have a valid selected collection, use it
        default_index = collections.index(st.session_state.selected_collection)
        collection_name = st.sidebar.selectbox("Select Collection", options=collections + ["New Collection"], index=default_index)
    else:
        # Default selection
        collection_name = st.sidebar.selectbox("Select Collection", options=collections + ["New Collection"])
    
    if collection_name == "New Collection":
        new_collection_name = st.sidebar.text_input(
            "Enter new collection name",
            help="Rules: 3-63 chars, alphanumeric/underscore/hyphen only, no spaces"
        )
        
        if st.sidebar.button("Create Collection"):
            if new_collection_name:
                try:
                    final_name, original_name = create_collection(new_collection_name)
                    st.sidebar.success(f"✅ Collection '{final_name}' created!")
                    # Set the selected collection and force refresh
                    st.session_state.selected_collection = final_name
                    st.session_state.chromadb_stats = get_chromadb_stats()
                    st.rerun()  # Force refresh to update the interface
                except Exception as e:
                    st.sidebar.error(f"❌ Failed: {str(e)}")
            else:
                st.sidebar.error("Please enter a name.")
    else:
        # Store the selected collection
        st.session_state.selected_collection = collection_name
    
    # Show upload section - use the actual selected collection name
    actual_collection = st.session_state.get('selected_collection', collection_name)
    
    if uploaded_files and actual_collection and actual_collection != "New Collection":
        st.sidebar.success(f"Ready to upload to '{actual_collection}'")
        if st.sidebar.button("🚀 Upload and Index", type="primary"):
            with st.spinner("Processing documents..."):
                try:
                    st.session_state.index = upload_files(uploaded_files, actual_collection, chunk_size, chunk_overlap)
                    if st.session_state.index:
                        st.session_state.chromadb_stats = get_chromadb_stats()
                        # Force refresh of collections for querying
                        st.session_state.pop('selected_query_collection', None)
                        st.success("🎉 Upload completed successfully!")
                        st.balloons()
                        st.rerun()  # Refresh to update query collections
                except Exception as e:
                    st.error(f"Upload failed: {e}")
    elif uploaded_files:
        st.sidebar.warning("⚠️ Create a collection first!")
    elif not uploaded_files:
        st.sidebar.info("📁 Select files to upload")

    # Configure global settings for embeddings and LLM
    ollama_base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
    Settings.embed_model = OllamaEmbedding(
        model_name=embed_model,
        base_url=ollama_base_url,
        ollama_additional_kwargs={"mirostat": 0},
    )
    Settings.llm = Ollama(model=model, base_url=ollama_base_url, request_timeout=360.0)
    Settings.context_window = context_window
    Settings.num_output = num_output
    Settings.chunk_size = chunk_size
    Settings.chunk_overlap = chunk_overlap

    # Initialize index
    if "index" not in st.session_state:
        st.session_state.index = None

    # Collection selection for querying
    all_collections_for_query = get_collections()
    if "demo_index" in st.session_state:
        all_collections_for_query.append("demo")
    
    # Add default selection if no collection is selected but collections exist
    if all_collections_for_query and not st.session_state.get('selected_query_collection'):
        st.session_state.selected_query_collection = all_collections_for_query[0]
    
    # Handle the case where collection might be empty
    if not all_collections_for_query:
        st.warning("⚠️ No collections available. Please upload documents first.")
        return
    
    query_collection = st.selectbox(
        "Select Collection for Querying", 
        options=all_collections_for_query,
        index=0 if all_collections_for_query else 0
    )

    # Show tutorial if no collection is selected
    if not query_collection:
        st.markdown(TUTORIAL_CONTENT)
        return

    # Get the index for the selected collection
    if query_collection == "demo":
        if "demo_index" in st.session_state:
            index = st.session_state.demo_index
        else:
            st.warning("Please ingest the demo data first.")
            return
    else:
        chroma_client = get_chroma_client()
        chroma_collection = chroma_client.get_collection(query_collection)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            storage_context=storage_context,
            embed_model=Settings.embed_model
        )

    # Query input field
    query = st.text_input("Enter your query:")

    # Options for query processing
    col1, col2, col3 = st.columns(3)
    with col1:
        use_rag = st.checkbox("Use RAG", value=True)
    with col2:
        compare = st.checkbox("Without RAG", value=True)
    with col3:
        print_context = st.checkbox("Print RAG context", value=True)

    # Handle query submission
    if st.button("Submit Query"):
        if query:
            with st.spinner('Generating response...'):

                query_engine = index.as_query_engine(
                    similarity_top_k=similarity_top_k,
                    use_async=True,
                    llm=Settings.llm,
                    streaming=True
                )

                if not use_rag:
                    st.subheader("Response without RAG:")
                    response_placeholder = st.empty()
                    full_response = ""
                    for response_chunk in Settings.llm.stream_complete(query):
                        full_response += response_chunk.delta
                        response_placeholder.markdown(
                            f'<div class="response-box">{full_response}</div>',
                            unsafe_allow_html=True
                        )

                elif compare:
                    col1, col2 = st.columns(2)

                    # Left column: no RAG
                    with col1:
                        st.subheader("Response without RAG:")
                        response_placeholder_norag = st.empty()
                        full_response_norag = ""
                        for response_chunk in Settings.llm.stream_complete(query):
                            full_response_norag += response_chunk.delta
                            response_placeholder_norag.markdown(
                                f'<div class="response-box">{full_response_norag}</div>',
                                unsafe_allow_html=True
                            )

                    # Right column: with RAG
                    with col2:
                        st.subheader("Response with RAG:")
                        response_placeholder_rag = st.empty()
                        full_response_rag = ""
                        response_rag = query_engine.query(query)
                        for text in response_rag.response_gen:
                            full_response_rag += text
                            response_placeholder_rag.markdown(
                                f'<div class="response-box">{full_response_rag}</div>',
                                unsafe_allow_html=True
                            )

                    # Print RAG context if requested
                    if print_context and response_rag.source_nodes:
                        st.subheader("RAG Context:")
                        context_placeholder = st.empty()
                        full_context = ""
                        for node_entry in response_rag.source_nodes:
                            full_context += f"File: {node_entry.node.metadata.get('file_path', 'N/A')}\n"
                            full_context += f"Text: {node_entry.node.text}\n\n"
                        context_placeholder.markdown(
                            f'<div class="context-box">{full_context}</div>',
                            unsafe_allow_html=True
                        )

                else:
                    st.subheader("Response with RAG:")
                    response_placeholder = st.empty()
                    full_response = ""
                    response = query_engine.query(query)
                    for text in response.response_gen:
                        full_response += text
                        response_placeholder.markdown(
                            f'<div class="response-box">{full_response}</div>',
                            unsafe_allow_html=True
                        )

                    if print_context and response.source_nodes:
                        st.subheader("RAG Context:")
                        context_placeholder = st.empty()
                        full_context = ""
                        for node_entry in response.source_nodes:
                            full_context += f"File: {node_entry.node.metadata.get('file_path', 'N/A')}\n"
                            full_context += f"Text: {node_entry.node.text}\n\n"
                        context_placeholder.markdown(
                            f'<div class="context-box">{full_context}</div>',
                            unsafe_allow_html=True
                        )
        else:
            st.warning("Please enter a query.")
            
    # Display ChromaDB statistics
    if 'chromadb_stats' not in st.session_state:
        st.session_state.chromadb_stats = get_chromadb_stats()

    stats = st.session_state.chromadb_stats
    st.sidebar.header("ChromaDB Stats")
    st.sidebar.write(f"Number of Collections: {stats['num_collections']}")
    st.sidebar.write(f"Total Vectors: {stats['total_vectors']}")
    
    # Add clear collections button
    if stats['num_collections'] > 0:
        if st.sidebar.button("🗑️ Clear All Collections"):
            try:
                chroma_client = get_chroma_client()
                collections = chroma_client.list_collections()
                for collection_name in collections:
                    chroma_client.delete_collection(collection_name)
                st.sidebar.success(f"✅ Cleared {len(collections)} collections")
                st.session_state.chromadb_stats = get_chromadb_stats()
                # Clear session state
                if 'demo_index' in st.session_state:
                    del st.session_state.demo_index
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"❌ Error clearing collections: {e}")

    # Copyright notice
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div style='text-align: center; color: #666; font-size: 0.8em;'>
        © 2024 Dennis Kruyt<br>
        AT Computing
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == '__main__':
    main()
