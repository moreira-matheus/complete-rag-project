"""
Contains useful functions for text processing.
"""
import re
from typing import Optional, Tuple, List, Generator
import pymupdf
import spacy
import unicodedata

def chunk_text_with_overlap(text, chunk_size, chunk_overlap):
    step = chunk_size - chunk_overlap
    for i in range(0, len(text), step):
        yield text[i:i+chunk_size]

def sentencize_text(text: str) -> Generator[str, None, None]:
    # python -m spacy download en_core_web_sm
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    yield from [sent.text for sent in doc.sents]

def normalize_text(text: str) -> str:
    """
    Normalizes a string into ascii.
    """
    return unicodedata.normalize("NFKD", text)\
        .encode("ascii", "ignore").decode("ascii")

def clean_text(text2clean: Optional[str]) -> str:
    """
    Cleans text2clean of unwanted characters.
    """
    cleaned = text2clean[:]
    cleaned = normalize_text(cleaned)
    cleaned = re.sub("\[\d{1,}\]", "", cleaned)
    cleaned = cleaned.replace("-\n", "")
    cleaned = cleaned.replace("\n", " ")

    for symbol in list("†‡⋆"):
        cleaned = cleaned.replace(symbol, "")

    return cleaned

class PageAwareSentencizer:
    def __init__(self, file_path):
        self.file_path = file_path

    def _read_words(self) -> List[Tuple[str, int]]:
        words = []
        with pymupdf.open(self.file_path) as doc:
            for page_num, page in enumerate(doc,start=1):
                # list of (x0, y0, x1, y1, word, block_no, line_no, word_no)
                raw_words = page.get_text("words")
                # sort top-to-bottom, left-to-right
                raw_words.sort(key=lambda w: (w[1], w[0]))
                words.extend(
                    [(clean_text(w[4]), page_num) for w in raw_words]
                )

        return words
        
    def _concat_text(self, words: List[Tuple[str, int]]) -> Tuple[str, List]:
        full_text = ""
        char_to_page = []

        for word, page in words:
            if full_text:
                full_text += " "
                char_to_page.append(None)
            for char in word:
                full_text += char
                char_to_page.append(page)
        
        return full_text, char_to_page

    def _sentencize_full_text(self, full_text: str, char_to_page: list):
        sentences_with_pages = []
        curr_idx = 0

        for sentence in sentencize_text(full_text):
            sentence_len = len(sentence)
            sentence_pages = set([page_num
                    for page_num in char_to_page[curr_idx : curr_idx + sentence_len]
                    if page_num is not None
            ])
            sentence_pages = tuple(sorted(sentence_pages))

            sentences_with_pages.append((sentence, sentence_pages))
            curr_idx += sentence_len  
            
        return sentences_with_pages

    def sentencize_with_page_num(self) -> Generator[List[Tuple[str, Tuple]], None, None]:
        words = self._read_words()
        full_text, char_to_page = self._concat_text(words)
        sentences_with_pages = self._sentencize_full_text(full_text, char_to_page)

        yield from sentences_with_pages