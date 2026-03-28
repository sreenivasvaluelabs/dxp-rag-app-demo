#!/usr/bin/env python3
"""
Simple ChromaDB Test Script
This script tests if ChromaDB is working properly on your system.
"""
import sys
import warnings
warnings.filterwarnings("ignore")

def test_chromadb():
    print("🔍 Testing ChromaDB installation...")
    
    try:
        print("   ✓ Importing chromadb...")
        import chromadb
        print(f"   ✓ ChromaDB version: {chromadb.__version__}")
    except Exception as e:
        print(f"   ❌ Failed to import ChromaDB: {e}")
        return False
    
    try:
        print("   ✓ Creating EphemeralClient...")
        client = chromadb.EphemeralClient()
        print("   ✓ EphemeralClient created successfully")
    except Exception as e:
        print(f"   ❌ Failed to create EphemeralClient: {e}")
        try:
            print("   ⚡ Trying basic Client...")
            client = chromadb.Client()
            print("   ✓ Basic Client created successfully")
        except Exception as e2:
            print(f"   ❌ Failed to create basic Client: {e2}")
            return False
    
    try:
        print("   ✓ Testing basic operations...")
        collection = client.create_collection("test_collection")
        print("   ✓ Collection created")
        
        collection.add(
            embeddings=[[1.2, 2.3, 4.5], [6.7, 8.2, 9.2]],
            documents=["This is the first document", "This is the second document"],
            ids=["id1", "id2"]
        )
        print("   ✓ Documents added")
        
        results = collection.query(
            query_embeddings=[[1.1, 2.3, 4.6]],
            n_results=1
        )
        print("   ✓ Query successful")
        print(f"   ✓ Found {len(results['documents'][0])} documents")
        
        collections = client.list_collections()
        print(f"   ✓ Listed collections: {len(collections)} found")
        
    except Exception as e:
        print(f"   ❌ Basic operations failed: {e}")
        return False
    
    print("🎉 All ChromaDB tests passed! Your installation is working correctly.")
    return True

def suggest_fixes():
    print("\n🔧 If tests fail, try these fixes:")
    print("1. Reinstall ChromaDB:")
    print(f"   {sys.executable} -m pip uninstall chromadb -y")
    print(f"   {sys.executable} -m pip install chromadb==0.6.3")
    print("\n2. Update dependencies:")
    print(f"   {sys.executable} -m pip install --upgrade numpy pandas sqlite3")
    print("\n3. Clear Python cache:")
    print(f"   {sys.executable} -m pip cache purge")
    print("\n4. Check Python version (needs 3.8+):")
    print(f"   Current Python version: {sys.version}")

if __name__ == "__main__":
    success = test_chromadb()
    if not success:
        suggest_fixes()
        sys.exit(1)
    else:
        print("\n✅ ChromaDB is ready for your RAG application!")
        sys.exit(0)