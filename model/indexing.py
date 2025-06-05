import os, faiss, pickle, pymupdf, unicodedata
import numpy as np
from typing import Generator, Tuple, Optional
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

class TextChunk:
    def __init__(self, text: str, file_path: str, page_num: int):
        self.text = text
        self.file_path = file_path
        self.page_num = page_num

class EmbeddedChunk(TextChunk):
    def __init__(self, chunk: TextChunk, embedding: np.ndarray):
        super().__init__(chunk.text, chunk.file_path, chunk.page_num)
        self.embedding = embedding

class TextChunker:
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunker = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def read_pages(
            self, file_path
        ) -> Generator[Tuple[int, Optional[str]], None, None]:
        doc = pymupdf.open(file_path)
        for page_num, page in enumerate(doc, start=1):
            yield page_num, page.get_text()

    def normalize_text(self, text):
        return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    
    #TODO
    def clean_text(self, text2clean: Optional[str]) -> str:
        cleaned = text2clean[:]
        cleaned = self.normalize_text(cleaned)
        cleaned = cleaned.replace("-\n", "")
        cleaned = cleaned.replace("\n", " ")

        return cleaned

    def chunk_text(self, text2chunk: Optional[str]) -> Generator[str, None, None]:
        if text2chunk is None:
            text2chunk = ""

        chunks = self.chunker.split_text(text2chunk)
        for chunk in chunks:
            yield chunk

    def process_text(self, file_path: str) -> Generator[TextChunk, None, None]:
        for page_num, page_txt in self.read_pages(file_path):
            clean_text = self.clean_text(page_txt)
            for chunk in self.chunk_text(clean_text):
                yield TextChunk(
                    text=chunk,
                    file_path=file_path,
                    page_num=page_num
                )

class ChunkEmbedder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
    
    def embed_chunk(
            self, chunk: TextChunk, encode_kwargs: dict = {}
        ) -> EmbeddedChunk:
        embedding = self.model.encode(
            [chunk.text],
            show_progress_bar=False,
            **encode_kwargs,
        )
        return EmbeddedChunk(chunk, embedding)

class Indexer:
    def __init__(self, index_path: str, metadata_path: str):
        self.index_path = index_path
        self.index = None
        self.metadata_path = metadata_path

    def index_chunk(self, embedded_chunk: EmbeddedChunk) -> None:
        self._add_to_index(
            embedded_chunk.embedding, 
            embedded_chunk.embedding.shape[1]
        )
        self._add_to_metadata({
            "text": embedded_chunk.text,
            "file_path": embedded_chunk.file_path,
            "page_num": embedded_chunk.page_num
        })

    def _load_index(self, dimensions):
        if self.index is None:
            if os.path.exists(self.index_path):
                self.index = faiss.read_index(self.index_path)
            else:
                self.index = faiss.IndexFlatIP(dimensions)

    def _save_index(self):
        faiss.write_index(self.index, self.index_path)
        self.index = None

    def _add_to_index(self, embedding: np.ndarray, dimensions: int) -> None:
        self._load_index(dimensions)
        self.index.add(embedding.astype('float32'))
        self._save_index()
    
    def _add_to_metadata(self, metadata: dict) -> None:
        existing_data = []

        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "rb") as f:
                existing_data = pickle.load(f)
        
        if isinstance(existing_data, list):
            metadata_list = existing_data + [metadata]
        else:
            metadata_list = [metadata]

        with open(self.metadata_path, "wb") as f:
            pickle.dump(metadata_list, f)


class IndexingPipeline:
    def __init__(self, cfg: dict):
        self.__check_cfg(cfg)
        self.cfg = cfg

    def __check_cfg(self, cfg: dict):
        fields = [
            "RAW_INPUT_FOLDER", "METADATA_PATH", "INDEX_PATH",
            "EMBEDDING_MODEL_NAME", "EMBEDDING_ENCODE_KWARGS",
            "CHUNK_SIZE", "CHUNK_OVERLAP"
        ]
        for field in fields:
            assert field in cfg.keys()

    def run(self):
        chunker = TextChunker(
            self.cfg["CHUNK_SIZE"],
            self.cfg["CHUNK_OVERLAP"]
        )
        embedder = ChunkEmbedder(
            self.cfg["EMBEDDING_MODEL_NAME"]
        )
        indexer = Indexer(
            self.cfg["INDEX_PATH"],
            self.cfg["METADATA_PATH"]
        )

        for fname in os.listdir(self.cfg["RAW_INPUT_FOLDER"]):
            if fname.endswith(".pdf"):
                file_path = os.path.join(self.cfg["RAW_INPUT_FOLDER"], fname)
                print(f"File path: {file_path}")
                for chunk in chunker.process_text(file_path):
                    embedded_chunk = embedder.embed_chunk(
                        chunk,
                        self.cfg["EMBEDDING_ENCODE_KWARGS"]
                    )
                    indexer.index_chunk(embedded_chunk)
        
        print("Done.")
