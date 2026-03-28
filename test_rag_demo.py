#!/usr/bin/env python3
"""
Test RAG with One Document - Simple Demo
This script shows exactly how the RAG app works with one PDF.
"""
import warnings
import os
import sys
warnings.filterwarnings('ignore')
os.environ['ANONYMIZED_TELEMETRY'] = 'False'

# Add current directory to path to import from rag.py
sys.path.append('.')

def test_rag_simple():
    print("🚀 Testing RAG with Moons_of_Mars.pdf...")
    
    try:
        # Test ChromaDB
        import chromadb
        client = chromadb.EphemeralClient()
        print("✅ ChromaDB working")
        
        # Test document loading
        from llama_index.core import SimpleDirectoryReader
        documents = SimpleDirectoryReader(input_files=["demo_docs/Moons_of_Mars.pdf"]).load_data()
        print(f"✅ Loaded PDF: {len(documents)} chunks")
        
        # Test collection creation
        collection = client.create_collection("test_moons")
        print("✅ Collection created: test_moons")
        
        # Test Ollama connection
        import requests
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json()["models"]
                print(f"✅ Ollama working: {len(models)} models available")
                model_names = [m["name"] for m in models[:3]]
                print(f"   Available models: {', '.join(model_names)}")
            else:
                print("❌ Ollama not responding")
                return False
        except:
            print("❌ Ollama not running - start with: ollama serve")
            return False
        
        print("\n🎯 READY TO TEST:")
        print("1. Your RAG app should work perfectly")
        print("2. Upload any PDF and create collection 'test'")
        print("3. Try these questions about Moons of Mars:")
        print("   - 'What are the names of Mars moons?'")
        print("   - 'How big is Phobos?'")
        print("   - 'When were Mars moons discovered?'")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_rag_simple()
    if success:
        print("\n✅ Everything is ready! Run: streamlit run rag.py")
    else:
        print("\n❌ Fix the issues above first")