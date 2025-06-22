"""
Contains utility class to interact with ChromaDB Client.
"""
import chromadb

class ChromaDBClient:
    """
    Wraps the ChromaDB Client and allows to
    query and add to collection.
    """
    def __init__(
            self,
            persist_directory: str,
            collection_name: str,
        ):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_to_collection(self, ids, embeddings, documents, metadatas):
        """
        Adds item(s) to the collection.
        """
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

    def query_collection(self, query_embeddings, top_k):
        """
        Queries the collection and returns the best top_k results.
        """
        results = self.collection.query(
            query_embeddings=query_embeddings,
            n_results=top_k
        )
        return results
