"""
Document Processing Service
Handles parsing and chunking of various document formats (PDF, DOCX, CSV, JSON).

Now supports context-aware chunking with Docling for improved RAG quality.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

import tiktoken
from unstructured.partition.auto import partition

from app.config import settings

logger = logging.getLogger("rag_app.document_service")


def parse_document(file_path: str) -> str:
    """
    Parse any document type and return extracted text.
    Uses fast direct read for simple text files (.txt, .md, .csv).
    Uses Unstructured.io for complex formats (PDF, DOCX, JSON, etc.).

    Args:
        file_path: Path to the document file

    Returns:
        str: Extracted text content from the document

    Raises:
        FileNotFoundError: If the file doesn't exist
        Exception: If parsing fails
    """
    # Verify file exists
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File Not Found: {file_path}")

    file_extension = Path(file_path).suffix.lower()
    if file_extension in [".txt", ".md", ".csv", ".log", ".json"]:
        try:
            logger.info(f"Using fast text read for {file_extension} file")
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, "r", encoding="latin-1") as f:
                    return f.read()
            except Exception as e:
                logger.warning(
                    f"Fast text read failed : {e} , falling back to unstructured."
                )
        except Exception as e:
            logger.warning(
                f"Fast text read failed : {e} ,falling back to unstructured."
            )
    """
    “Fast read” here means:

    Directly reading the file as plain text using Python’s built-in open() instead of using the Unstructured parser.
    """
    try:
        # Use Unstructured.io's auto partition for complex formats (PDF, DOCX, etc.)
        # strategy="fast" disables OCR (tesseract) for Lambda compatibility
        # OCR can be enabled by adding tesseract Lambda layer and using strategy="hi_res"
        logger.info(f"Using unstructured library for {file_extension} file")
        elements = partition(
            filename=file_path,
            strategy="fast",  # Fast mode: no OCR, works without tesseract
        )

        # Combine all elements into a single text string
        text = "\n\n".join([str(el) for el in elements])
        return text

    except Exception as e:
        raise Exception(f"Failed to parse document {file_path}: {str(e)}")


def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50,
    encoding_name: str = "cl100k_base",  # GPT-4 encoding
) -> List[Dict[str, Any]]:
    """
    Split text into overlapping chunks based on token count.

    Args:
        text: The text to chunk
        chunk_size: Maximum tokens per chunk (default: 512)
        overlap: Number of overlapping tokens between chunks (default: 50)
        encoding_name: Tokenizer encoding to use (default: cl100k_base for GPT-4)

    Returns:
        List of dictionaries containing:
            - text: The chunk text
            - chunk_index: Index of the chunk
            - token_count: Number of tokens in the chunk
            - start_char: Starting character position
            - end_char: Ending character position
    """
    # Initialize tokenizer
    try:
        tokenizer = tiktoken.get_encoding(encoding_name)
    except Exception:
        # fallback to default tokenizer
        tokenizer = tiktoken.encoding_for_model("gpt-4")

    tokens = tokenizer.encode(text)

    chunks = []
    start_idx = 0

    while start_idx < len(tokens):
        # Get chunk tokens
        end_idx = min(start_idx + chunk_size, len(tokens))
        chunk_tokens = tokens[start_idx:end_idx]

        chunk_text = tokenizer.decode(chunk_tokens)

        # Calculate character positions(approximate)
        if chunks:
            start_char = chunks[-1]["end_char"] - (overlap * 4)  # Rough estimate
            # Since we generally see in bpe , that 1 token is equal to four chars
            start_char = max(0, start_char)
        else:
            start_char = 0

        end_char = start_char + len(chunk_text)

        chunk_data = {
            "text": chunk_text,
            "chunk_index": len(chunks),
            "token_count": len(chunk_tokens),
            "start_char": start_char,
            "end_char": end_char,
        }
        chunks.append(chunk_data)

        # Move to next chunk with overlap
        start_idx += chunk_size - overlap

        # Break if we've reached the end
        if end_idx >= len(tokens):
            break

    return chunks


def chunk_text_semantic(
    text: str, chunk_size: int = 512, encoding_name: str = "cl100k_base"
) -> List[Dict[str, Any]]:
    """
    Split text into semantic chunks using semchunk library.

    Better than token-based chunking because it:
    - Respects sentence boundaries (no mid-sentence splits)
    - Maintains semantic coherence
    - Still lightweight (pure Python, no PyTorch)

    Falls back to token-based chunking if semchunk unavailable.

    Args:
        text: The text to chunk
        chunk_size: Maximum tokens per chunk (default: 512)
        encoding_name: Tokenizer encoding to use (default: cl100k_base for GPT-4)

    Returns:
        List of dictionaries containing chunk metadata
    """

    tokenizer = tiktoken.get_encoding(encoding_name=encoding_name)

    try:
        from semchunk.semchunk import chunkerify

        chunker = chunkerify(tokenizer, chunk_size=chunk_size)
        # use semchunk for semantic boundaries
        semantic_chunks = chunker(text)

        chunks = []
        char_position = 0

        # Meta data
        for idx, chunk_sen in enumerate(semantic_chunks):
            # tokens = tokenizer.encode(chunk_text)

            chunk_data = {
                "text": chunk_sen,
                "chunk_index": idx,
                "token_count": len(tokenizer.encode(chunk_sen)),
                "start_char": char_position,
                "end_char": char_position + len(chunk_sen),
                # Add empty metadata for compatibility with Docling format
                "headings": [],
                "page_numbers": [],
                "doc_items": [],
                "captions": [],
            }

            chunks.append(chunk_data)
            char_position += len(chunk_sen)

        logger.info(f"Semantic Chunking complete: {len(chunks)} chunks (semchunk)")
        return chunks

    except Exception as e:
        if isinstance(e, ImportError):
            logger.warning(
                "semchunk not available, falling back to token-based chunking"
            )
        else:
            logger.warning(
                f"Semantic chunking failed: {e}, falling back to token-based"
            )

        fallback_chunks = chunk_text(text, chunk_size=chunk_size, overlap=50)
        for chunk in fallback_chunks:
            chunk["headings"] = []
            chunk["page_numbers"] = []
            chunk["doc_items"] = []
            chunk["captions"] = []
        return fallback_chunks


def get_document_stats(file_path: str) -> Dict[str, Any]:
    """
    Get statistics about a document.

    Args:
        file_path: Path to the document

    Returns:
        Dictionary with document statistics
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Parse document
    text = parse_document(file_path)

    # Get token count
    tokenizer = tiktoken.encoding_for_model("gpt-4")
    tokens = tokenizer.encode(text)

    return {
        "filename": path.name,
        "file_size_bytes": path.stat().st_size,
        "file_type": path.suffix,
        "character_count": len(text),
        "token_count": len(tokens),
        "estimated_chunks_512": (len(tokens) // 512) + 1,
    }


# File types Docling handles well (rich document formats with layout structure)
_DOCLING_SUPPORTED = {".pdf", ".docx", ".doc", ".pptx", ".ppt", ".html", ".htm"}

# Plain-text formats — no layout to analyse, Docling adds no value
_PLAIN_TEXT = {".txt", ".md", ".csv", ".log", ".json"}


def parse_and_chunk_with_context(
    file_path: str, chunk_size: int = 512, min_chunk_size: int = 256
) -> List[Dict[str, Any]]:
    """
    Parse and chunk a document, always attempting Docling first.

    Processing order
    ----------------
    1. Plain-text files (.txt / .md / .csv / .log / .json)
       Docling adds no value here (no layout to analyse).
       → direct read → semchunk

    2. Rich document formats (.pdf / .docx / .pptx / .html …)
       a. Docling  — structure-aware chunking with heading context,
                     page numbers, captions, and semantic boundaries.
       b. Unstructured + semchunk  — fallback when Docling is
                     unavailable or fails.

    Returns
    -------
    List of chunk dicts compatible with the vector / cache services.
    Each chunk contains at minimum: text, chunk_index, token_count,
    start_char, end_char, headings, page_numbers, doc_items, captions.

    Raises
    ------
    ValueError  — when no text can be extracted at all (e.g. image-only PDF)
                  or when every chunking strategy produces 0 chunks.
    """
    file_name = Path(file_path).name
    file_extension = Path(file_path).suffix.lower()

    # ------------------------------------------------------------------ #
    # 1. Plain-text fast path                                              #
    # ------------------------------------------------------------------ #
    if file_extension in _PLAIN_TEXT:
        logger.info(
            f"[Docling SKIP] Plain-text file — using semchunk directly: {file_name}"
        )
        text = parse_document(file_path)
        if not text.strip():
            raise ValueError(
                f"No text extracted from '{file_name}'. The file appears to be empty."
            )
        chunks = chunk_text_semantic(text, chunk_size=chunk_size)
        logger.info(f"[semchunk] {len(chunks)} chunks created for '{file_name}'")
        return chunks

    # ------------------------------------------------------------------ #
    # 2a. Primary parser: Docling                                          #
    # ------------------------------------------------------------------ #
    if settings.USE_DOCLING and file_extension in _DOCLING_SUPPORTED:
        logger.info(f"[Docling START] Parsing '{file_name}' with Docling (primary)")
        try:
            from app.services.docling_service import parse_and_chunk_document

            chunks = parse_and_chunk_document(
                file_path, chunk_size=chunk_size, min_chunk_size=min_chunk_size
            )

            if chunks:
                logger.info(
                    f"[Docling OK] {len(chunks)} chunks created for '{file_name}'"
                )
                return chunks

            # Docling ran without error but produced nothing
            logger.warning(
                f"[Docling WARN] 0 chunks returned for '{file_name}' — "
                "falling back to Unstructured + semchunk"
            )

        except ImportError as exc:
            logger.warning(
                f"[Docling UNAVAILABLE] {exc} — falling back to Unstructured + semchunk"
            )
        except Exception as exc:
            logger.error(
                f"[Docling ERROR] '{file_name}': {exc} — "
                "falling back to Unstructured + semchunk"
            )
    elif not settings.USE_DOCLING:
        logger.info("[Docling DISABLED] USE_DOCLING=false — using Unstructured + semchunk")
    else:
        logger.info(
            f"[Docling SKIP] '{file_extension}' not in supported set — using Unstructured + semchunk"
        )

    # ------------------------------------------------------------------ #
    # 2b. Fallback: Unstructured + semchunk                                #
    # ------------------------------------------------------------------ #
    logger.info(f"[Unstructured START] Parsing '{file_name}'")
    text = parse_document(file_path)

    if not text.strip():
        raise ValueError(
            f"No text could be extracted from '{file_name}'. "
            "The file may be a scanned/image-only PDF, or in an unsupported format. "
            "Convert it to a text-based PDF and re-upload."
        )

    chunks = chunk_text_semantic(text, chunk_size=chunk_size)

    if not chunks:
        raise ValueError(
            f"Chunking produced 0 chunks for '{file_name}' even though text was extracted. "
            "Check the CHUNK_SIZE setting."
        )

    logger.info(f"[semchunk FALLBACK] {len(chunks)} chunks created for '{file_name}'")
    return chunks
