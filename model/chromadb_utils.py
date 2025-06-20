import chromadb

class ChromaDBClient:
    def __init__(
            self,
            persist_directory: str,
            collection_name: str,
        ):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name=collection_name)
    
    def add_to_collection(self, ids, embeddings, documents, metadatas):
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
    
    def query_collection(self, query_embeddings, top_k):
        results = self.collection.query(
            query_embeddings=query_embeddings,
            n_results=top_k
        )
        return results
