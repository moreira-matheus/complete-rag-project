"""
Contains all classes used in the indexing process.
"""

import os
import unicodedata
import uuid
from typing import Generator, Tuple, Optional
from dataclasses import dataclass
import pymupdf
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

from model.chromadb_utils import ChromaDBClient

@dataclass
class TextChunk:
    """
    Dataclass to store information about a text chunk.
    """
    text: str
    file_path: str
    page_num: int

@dataclass
class EmbeddedChunk(TextChunk):
    """
    Dataclass to store information about a text chunk
    after embedding.
    """
    embedding: np.ndarray

class TextChunker:
    """
    Reads the pages off a PDF file, processes the text,
    and yields the text chunks.
    """
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
        """
        Reads the PDF file at file_path
        and yields its pages.
        """
        doc = pymupdf.open(file_path)
        for page_num, page in enumerate(doc, start=1):
            yield page_num, page.get_text()

    def normalize_text(self, text):
        """
        Normalizes a string into ascii.
        """
        return unicodedata.normalize("NFKD", text)\
            .encode("ascii", "ignore").decode("ascii")

    def clean_text(self, text2clean: Optional[str]) -> str:
        """
        Cleans text2clean of unwanted characters.
        """
        cleaned = text2clean[:]
        cleaned = self.normalize_text(cleaned)
        cleaned = cleaned.replace("-\n", "")
        cleaned = cleaned.replace("\n", " ")

        for symbol in list("†‡⋆"):
            cleaned = cleaned.replace(symbol, "")

        return cleaned

    def chunk_text(self, text2chunk: Optional[str]) -> Generator[str, None, None]:
        """
        Chunks text2chunk into pieces of size self.chunk_size
        with overlaps of size self.chunk_overlap.
        """
        if text2chunk is None:
            text2chunk = ""

        chunks = self.chunker.split_text(text2chunk)
        yield from chunks

    def process_text(self, file_path: str) -> Generator[TextChunk, None, None]:
        """
        Reads pages, cleans and chenks the text,
        then yields the chunks as TextChunk objects.
        """
        for page_num, page_txt in self.read_pages(file_path):
            clean_text = self.clean_text(page_txt)
            for chunk in self.chunk_text(clean_text):
                yield TextChunk(
                    text=chunk,
                    file_path=file_path,
                    page_num=page_num
                )

# pylint: disable=too-few-public-methods
class ChunkEmbedder:
    """
    Embeds chunks of texts using a SentenceTransformer model. 
    """
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_chunk(
            self, chunk: TextChunk, encode_kwargs: Optional[dict] = None
        ) -> EmbeddedChunk:
        """
        Embeds chunk into numpy.ndarrays
        and returns it as a EmbeddedChunk object.
        """
        if encode_kwargs is None:
            encode_kwargs = {}

        embedding = self.model.encode(
            chunk.text,
            show_progress_bar=False,
            **encode_kwargs,
        )
        return EmbeddedChunk(
            chunk.text, chunk.file_path, chunk.page_num, embedding
        )

# pylint: disable=too-few-public-methods
class Indexer:
    """
    Receives an EmbeddedChunk and indexes it
    in a ChromaDB collection.
    """
    def __init__(
            self,
            persist_directory: str,
            collection_name: str,
        ):
        self.client = ChromaDBClient(
            persist_directory=persist_directory,
            collection_name=collection_name
        )

    def _generate_id(self):
        return str(uuid.uuid4())

    def index_chunk(self, embedded_chunk: EmbeddedChunk) -> None:
        """
        Receives embedded_chunk, generates an id for it,
        then adds it to a ChromaDB collection.
        """
        self.client.add_to_collection(
            ids=[self._generate_id()],
            embeddings=[embedded_chunk.embedding.tolist()],
            documents=[embedded_chunk.text],
            metadatas=[
                {
                    "page_num": embedded_chunk.page_num,
                    "file_path": embedded_chunk.file_path
                }
            ]
        )

# pylint: disable=too-few-public-methods
class IndexingPipeline:
    """
    Wraps all steps required for indexing text chunks.
    """
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
        """
        Runs the indexing pipeline.
        Returns None.
        """
        chunker = TextChunker(
            self.cfg["CHUNK_SIZE"],
            self.cfg["CHUNK_OVERLAP"]
        )
        embedder = ChunkEmbedder(
            self.cfg["EMBEDDING_MODEL_NAME"]
        )
        indexer = Indexer(
            self.cfg["INDEX_DIR"],
            self.cfg["COLLECTION_NAME"]
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
