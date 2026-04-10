from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []

        # Split on sentence-ending punctuation followed by a space or newline.
        # Use a regex that keeps the delimiter attached to the preceding sentence.
        sentences = re.split(r'(?<=[.!?])\s+|(?<=\.)\n', text)
        # Remove empty strings and strip whitespace
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return []

        chunks: list[str] = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            group = sentences[i : i + self.max_sentences_per_chunk]
            chunks.append(" ".join(group))
        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        # Base case: text fits within chunk_size
        if len(current_text) <= self.chunk_size:
            return [current_text]

        # No separators left — fall back to character-level splitting
        if not remaining_separators:
            chunks: list[str] = []
            for start in range(0, len(current_text), self.chunk_size):
                chunks.append(current_text[start : start + self.chunk_size])
            return chunks

        separator = remaining_separators[0]
        next_separators = remaining_separators[1:]

        # Try splitting with the current separator
        if separator == "":
            # Character-level split
            chunks = []
            for start in range(0, len(current_text), self.chunk_size):
                chunks.append(current_text[start : start + self.chunk_size])
            return chunks

        parts = current_text.split(separator)

        result: list[str] = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if len(part) <= self.chunk_size:
                result.append(part)
            else:
                # Recursively split with the next separator
                result.extend(self._split(part, next_separators))

        return result if result else [current_text]


class SemanticChunker:
    """
    Split text into chunks based on semantic similarity between adjacent sentences.

    Algorithm:
        1. Split text into sentences.
        2. Embed each sentence using the provided embedding function.
        3. Compute cosine similarity between consecutive sentence embeddings.
        4. Start a new chunk when similarity drops below `breakpoint_threshold`.
        5. Merge chunks that are too small (below `min_chunk_size` chars) with neighbors.

    Why it's better:
        - Chunks stay on-topic instead of cutting mid-idea.
        - Retrieval precision in RAG improves because each chunk covers one coherent concept.

    Args:
        embed_fn: Callable[[str], list[float]] — returns a dense vector for a string.
        breakpoint_threshold: Similarity below this value triggers a chunk boundary.
        min_chunk_size: Chunks shorter than this are merged with the next one.
    """

    def __init__(
        self,
        embed_fn,
        breakpoint_threshold: float = 0.75,
        min_chunk_size: int = 100,
    ) -> None:
        self.embed_fn = embed_fn
        self.breakpoint_threshold = breakpoint_threshold
        self.min_chunk_size = min_chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []

        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|(?<=\.)\n', text) if s.strip()]
        if not sentences:
            return []
        if len(sentences) == 1:
            return sentences

        embeddings = [self.embed_fn(s) for s in sentences]

        # Build initial groups by detecting semantic breaks
        groups: list[list[str]] = [[sentences[0]]]
        for i in range(1, len(sentences)):
            sim = compute_similarity(embeddings[i - 1], embeddings[i])
            if sim < self.breakpoint_threshold:
                groups.append([sentences[i]])
            else:
                groups[-1].append(sentences[i])

        # Merge groups that are too small into the next group
        merged: list[list[str]] = []
        for group in groups:
            text_len = sum(len(s) for s in group)
            if merged and text_len < self.min_chunk_size:
                merged[-1].extend(group)
            else:
                merged.append(group)

        return [" ".join(g) for g in merged if g]


class ParagraphMergingChunker:
    """
    Chunk text by respecting paragraph boundaries, then merging small paragraphs
    and splitting oversized ones at sentence boundaries.

    Algorithm:
        1. Split on blank lines (double newline) to get natural paragraphs.
        2. Accumulate paragraphs into a buffer until adding the next one would
           exceed `target_size`.
        3. Flush the buffer as a chunk, then start fresh.
        4. If a single paragraph exceeds `target_size`, split it into sentences
           and pack them into sub-chunks up to `target_size`.

    Why it's better than RecursiveChunker:
        - Paragraphs are never cut in the middle unless they're oversized.
        - Produces more uniform chunk sizes while keeping logical units intact.
        - Better preserves document structure for structured text (reports, articles).

    Args:
        target_size: Soft ceiling on chunk character length.
        overlap_sentences: Number of sentences to carry over from the previous
                           chunk to maintain context across boundaries.
    """

    def __init__(self, target_size: int = 500, overlap_sentences: int = 1) -> None:
        self.target_size = target_size
        self.overlap_sentences = max(0, overlap_sentences)

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []

        paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
        if not paragraphs:
            return []

        chunks: list[str] = []
        buffer: list[str] = []
        buffer_len = 0
        carry_sentences: list[str] = []  # overlap context from previous chunk

        def flush(buf: list[str]) -> None:
            if buf:
                joined = " ".join(buf)
                chunks.append(joined)

        for para in paragraphs:
            if len(para) > self.target_size:
                # Flush current buffer first
                flush(buffer)
                buffer, buffer_len = [], 0

                # Split oversized paragraph into sentence sub-chunks
                sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', para) if s.strip()]
                sub_buf: list[str] = list(carry_sentences)
                sub_len = sum(len(s) for s in sub_buf)
                for sent in sents:
                    if sub_len + len(sent) + 1 > self.target_size and sub_buf:
                        chunks.append(" ".join(sub_buf))
                        carry_sentences = sub_buf[-self.overlap_sentences:] if self.overlap_sentences else []
                        sub_buf = list(carry_sentences) + [sent]
                        sub_len = sum(len(s) for s in sub_buf)
                    else:
                        sub_buf.append(sent)
                        sub_len += len(sent) + 1
                if sub_buf:
                    flush(sub_buf)
                    carry_sentences = sub_buf[-self.overlap_sentences:] if self.overlap_sentences else []
            else:
                addition = len(para) + (1 if buffer else 0)
                if buffer and buffer_len + addition > self.target_size:
                    flush(buffer)
                    carry_sentences = []
                    # Extract trailing sentences for overlap
                    if self.overlap_sentences and chunks:
                        last_sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', chunks[-1]) if s.strip()]
                        carry_sentences = last_sents[-self.overlap_sentences:]
                    buffer = list(carry_sentences) + [para]
                    buffer_len = sum(len(s) for s in buffer)
                else:
                    buffer.append(para)
                    buffer_len += addition

        flush(buffer)
        return chunks


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    mag_a = math.sqrt(_dot(vec_a, vec_a))
    mag_b = math.sqrt(_dot(vec_b, vec_b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return _dot(vec_a, vec_b) / (mag_a * mag_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200, embed_fn=None) -> dict:
        strategies = {
            "fixed_size": FixedSizeChunker(chunk_size=chunk_size, overlap=0),
            "by_sentences": SentenceChunker(max_sentences_per_chunk=3),
            "recursive": RecursiveChunker(chunk_size=chunk_size),
            "paragraph_merging": ParagraphMergingChunker(target_size=chunk_size, overlap_sentences=1),
        }
        if embed_fn is not None:
            strategies["semantic"] = SemanticChunker(embed_fn=embed_fn, breakpoint_threshold=0.75)

        result: dict = {}
        for name, chunker in strategies.items():
            chunks = chunker.chunk(text)
            count = len(chunks)
            avg_length = sum(len(c) for c in chunks) / count if count > 0 else 0.0
            result[name] = {
                "count": count,
                "avg_length": avg_length,
                "chunks": chunks,
            }
        return result
