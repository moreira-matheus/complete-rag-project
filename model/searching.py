
import json
from typing import List
from sentence_transformers import SentenceTransformer

from model.chromadb_utils import ChromaDBClient

class SearchResult:
    def __init__(self, rank: int, score: float, text: str, metadata: dict):
        self.rank = rank
        self.score = score
        self.text = text
        self.metadata = metadata
    
    def __repr__(self):
        repr = {
            k: str(v) for k, v in self.__dict__.items()
        }
        return json.dumps(repr, indent=2)

class SearchEngine:
    def __init__(self, cfg: dict):
        self.__check_cfg(cfg)
        self.cfg = cfg
        self.client = ChromaDBClient(
            persist_directory=self.cfg["INDEX_DIR"],
            collection_name=self.cfg["COLLECTION_NAME"]
        )
    
    def __check_cfg(self, cfg: dict):
        fields = [
            "EMBEDDING_MODEL_NAME", "EMBEDDING_ENCODE_KWARGS",
            "INDEX_DIR", "COLLECTION_NAME"
        ]

        for field in fields:
            assert field in cfg.keys()
    
    def embed_query(self, query: str):
        return SentenceTransformer(self.cfg["EMBEDDING_MODEL_NAME"])\
            .encode(query, **self.cfg["EMBEDDING_ENCODE_KWARGS"])
    
    def search_index(self, query: str, top_k: int = 3) -> List[SearchResult]:
        results = []

        embedded_query = self.embed_query(query)
        query_results = self.client.query_collection(
            query_embeddings=[embedded_query],  # List of 1 or more embeddings
            top_k=top_k
        )
        for rank in range(top_k):
            results.append(
                SearchResult(
                    rank=rank+1,
                    score=query_results["distances"][0][rank],
                    text=query_results["documents"][0][rank],
                    metadata=query_results["metadatas"][0][rank]
                )
            )
        
        return results
