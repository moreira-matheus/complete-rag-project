"""
def search_faiss(query, model, index, chunks, k=5):
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding).astype('float32'), k)
    results = []
    for idx in I[0]:
        results.append(chunks[idx])
    return results
"""

import os, faiss, pickle
from typing import List
from sentence_transformers import SentenceTransformer

class SearchResult:
    def __init__(self, rank: int, score: float, text: str, metadata: dict):
        self.rank = rank
        self.score = score
        self.text = text
        self.metadata = metadata
        
class SearchEngine:
    def __init__(self, cfg: dict):
        self.__check_cfg(cfg)
        self.cfg = cfg
    
    def __check_cfg(self, cfg: dict):
        fields = ["EMBEDDING_MODEL_NAME", "INDEX_PATH", "METADATA_PATH"]

        for field in fields:
            assert field in cfg.keys()
    
    def embed_query(self, query: str):
        return SentenceTransformer(self.cfg["EMBEDDING_MODEL_NAME"])\
            .encode([query], **self.cfg["EMBEDDING_ENCODE_KWARGS"])
    
    def load_index(self) -> faiss.IndexFlatL2:
        if os.path.exists(self.cfg["INDEX_PATH"]):
            return faiss.read_index(self.cfg["INDEX_PATH"])
        
        raise FileNotFoundError(
            f"Could not find index @ {self.cfg['INDEX_PATH']}."
        )
    
    def load_metadata(self) -> list:
        if os.path.exists(self.cfg["METADATA_PATH"]):
            with open(self.cfg["METADATA_PATH"], "rb") as f:
                return pickle.load(f)
        
        raise FileNotFoundError(
            f"Could not find metadata @ {self.cfg['METADATA_PATH']}."
        )

    def search_index(self, query: str, top_k: int = 3) -> List[SearchResult]:
        embedded_query = self.embed_query(query)
        index = self.load_index()
        metadata = self.load_metadata()
        results = []

        distances, indices = index.search(
            embedded_query, k=top_k
        )
        for rank, idx in enumerate(indices[0]):
            if idx == -1:
                continue

            try:
                result = SearchResult(**{
                    "rank": rank + 1,
                    "score": distances[0][rank],
                    "text": metadata[idx].get("text", ""),
                    "metadata": {
                        "file_path": metadata[idx].get("file_path", None),
                        "page_num": metadata[idx].get("page_num", None)
                    }
                })
                results.append(result)
            except KeyError as e:
                print(e)
                print(f"Idx: {idx}")
                print(f"Metadata length: {len(metadata)}")
            except Exception as e:
                print(e)
        
        return results