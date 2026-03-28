#!/usr/bin/env python3
"""
Minimal RAG Demo - Simplified version that should work
"""
import streamlit as st
import chromadb
import warnings
import os

warnings.filterwarnings('ignore')
os.environ['ANONYMIZED_TELEMETRY'] = 'False'

st.set_page_config(page_title="Simple RAG Demo")
st.title("🚀 Simple RAG Demo")

# Test ChromaDB connection
@st.cache_resource
def get_chromadb_client():
    """Get ChromaDB client with simple fallback"""
    try:
        return chromadb.EphemeralClient()
    except:
        try:
            return chromadb.Client()
        except Exception as e:
            st.error(f"ChromaDB Error: {e}")
            st.stop()

# Test the connection
def test_chromadb():
    try:
        client = get_chromadb_client()
        collections = client.list_collections()
        st.success(f"✅ ChromaDB working! Found {len(collections)} collections")
        return client
    except Exception as e:
        st.error(f"❌ ChromaDB failed: {e}")
        
        with st.expander("🔧 Troubleshooting"):
            st.write("**Try these commands:**")
            st.code("pip uninstall chromadb -y && pip install chromadb==0.6.3")
            st.code("python test_chromadb.py")
        
        st.stop()

# Main app
client = test_chromadb()

st.write("### 🧪 Test ChromaDB Operations")
if st.button("Create Test Collection"):
    try:
        collection = client.create_collection(f"test_{len(client.list_collections())}")
        
        # Add some test data
        collection.add(
            embeddings=[[1.2, 2.3], [3.4, 5.6]],
            documents=["Hello world", "Goodbye world"],
            ids=["doc1", "doc2"]
        )
        
        st.success("✅ Test collection created and data added!")
        st.json({
            "collection_count": len(client.list_collections()),
            "test_data_added": 2
        })
        
    except Exception as e:
        st.error(f"❌ Test failed: {e}")

collections = client.list_collections()
if collections:
    st.write("### 📁 Existing Collections")
    for i, collection_name in enumerate(collections):
        col = client.get_collection(collection_name)
        st.write(f"{i+1}. {collection_name} ({col.count()} items)")

st.write("### ℹ️ System Info")
st.write(f"Python version: {st.__version__}")

# Simple instructions
st.write("---")
st.write("### 🎯 Next Steps")
st.write("If this simple test works:")
st.write("1. ChromaDB is properly installed")
st.write("2. You can run the main RAG application")
st.write("3. If the main app still fails, there may be a specific configuration issue")

if st.button("🔄 Refresh"):
    st.rerun()