"""
Contains all classes used in the indexing process.
"""
import os
import uuid
from itertools import repeat
from dataclasses import dataclass
from typing import Generator, Tuple, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

from model.utils.chromadb_utils import ChromaDBClient
from model.utils.text_utils import clean_text, chunk_text_with_overlap, PageAwareSentencizer

@dataclass
class TextChunk:
    """
    Dataclass to store information about a text chunk.
    """
    text: str
    file_path: str
    page_num: Tuple[int]

@dataclass
class EmbeddedChunk(TextChunk):
    """
    Dataclass to store information about a text chunk
    after embedding.
    """
    embedding: np.ndarray

class TextProcessor:
    """
    Reads the pages off a PDF file, processes the text,
    and yields the text chunks.
    """
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_sentences(self, sentences_with_pages) -> Generator[Tuple[str, Tuple], None, None]:
        chunks = []
        pages = set((1,))

        for sentence, page_nums in sentences_with_pages:
            current_length = len(' '.join(chunks))
            future_length = len(' '.join(chunks + [sentence]))

            if current_length < self.chunk_size:
                if future_length < self.chunk_size:
                    chunks.append(sentence)
                    pages.union(page_nums)
                else:
                    if chunks:
                        yield (
                            ' '.join(chunks),
                            tuple(pages)
                        )
                    chunks = [sentence]
                    pages = set(page_nums)
            else:
                overflow_chunk = ' '.join(chunks)
                yield from zip(
                    chunk_text_with_overlap(
                        overflow_chunk, self.chunk_size, self.chunk_overlap
                    ),
                    repeat(tuple(pages))
                )
                chunks = [sentence]
                pages = set(page_nums)
        else:
            yield from zip(
                chunk_text_with_overlap(
                    ' '.join(chunks), self.chunk_size, self.chunk_overlap
                ),
                repeat(tuple(pages))
            )

    def process_text(self, file_path):
        sentencizer = PageAwareSentencizer(file_path)
        for text, pages in self.chunk_sentences(
            sentencizer.sentencize_with_page_num()
        ):
            yield TextChunk(
                text=text,
                file_path=file_path,
                page_num=pages
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
