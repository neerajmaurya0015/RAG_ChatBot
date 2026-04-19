import chromadb

client = chromadb.PersistentClient(
    path=r"C:\Users\neeraj.maurya01\Documents\GE_Ai\QnA_Bot\vector_db"
)

collections = client.list_collections()

print("Collections:", collections)

if collections:
    collection = client.get_collection(collections[0].name)
    print("Document count:", collection.count())