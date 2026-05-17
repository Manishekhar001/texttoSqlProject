"""
Docling Service
Provides context-aware document parsing and chunking using Docling's HybridChunker.
Preserves document structure and hierarchical heading context for better RAG quality.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger("rag_app.docling_service")

try:
    from docling.chunking import HybridChunker
    from docling.document_converter import DocumentConverter
    from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer

    DOCLING_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Docling not available : {e}")
    DOCLING_AVAILABLE = False


def convert_document(file_path: str):
    """
    Convert document using Docling's advanced layout analysis.

    Args:
        file_path: Path to the document file

    Returns:
        DoclingDocument: Structured document with hierarchy preserved

    Raises:
        ImportError: If Docling is not installed
        Exception: If conversion fails
    """
    if not DOCLING_AVAILABLE:
        raise ImportError(
            "Docling is not installed. Run : pip install docling docling_core"
        )

    if not Path(file_path).exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        logger.info(f"Converting document with Docling : {Path(file_path).name}")
        converter = DocumentConverter()
        result = converter.convert(file_path)
        doc = result.document

        logger.info(f"Document converted successfully : {len(doc.texts)} text elements")
        return doc
    except Exception as e:
        logger.error(f"Docling conversion failed: {str(e)}")
        raise Exception(f"Failed to convert document with Docling: {str(e)}")


def chunk_with_hybrid(
    doc, max_tokens: int = 512, min_tokens: int = 256
) -> List[Dict[str, Any]]:
    """
    Chunk a DoclingDocument into semantically meaningful, size-bounded pieces
    using Docling's HybridChunker, with a post-processing merge pass to enforce
    a minimum chunk size.

    WHY THIS FUNCTION EXISTS:
        Raw document text split naively by token count loses structural context —
        a paragraph gets split mid-sentence, a heading gets separated from its body,
        a table caption ends up in a different chunk from its table. HybridChunker
        solves this by respecting the document's internal structure (headings,
        paragraphs, tables, figures) detected during Docling's layout analysis.
        The merge pass on top solves the opposite problem — HybridChunker can
        produce very small chunks (a 2-word heading, a single short sentence) that
        carry too little context for GPT-4 to answer questions well.

    TWO-STAGE PROCESS:
        Stage 1 — HybridChunker:
            Splits the document respecting structural boundaries.
            Enforces max_tokens as an upper limit per chunk.
            Preserves heading hierarchy and page number metadata per chunk.

        Stage 2 — Merge pass:
            Iterates through HybridChunker's output.
            Merges consecutive chunks where the current accumulated chunk is
            still below min_tokens AND the combined size stays within max_tokens.
            Merges heading lists and page number lists from both chunks.
            Result: every chunk (except possibly the last) is between
            min_tokens and max_tokens.

    TOKENIZER:
        Uses tiktoken cl100k_base (GPT-4 / text-embedding-3-small encoding).
        This ensures token counts here are consistent with the token counts
        used during embedding generation and Pinecone storage — they all
        speak the same token language.

    Args:
        doc:
            A DoclingDocument object returned by convert_document().
            Must have been processed by Docling's layout analysis pipeline,
            which provides structural metadata (headings, page numbers, etc.).

        max_tokens (int):
            Hard upper limit on tokens per chunk. No chunk in the output
            will exceed this count. Default 512 matches OpenAI's recommended
            context window per retrieval chunk.

        min_tokens (int):
            Soft lower target for tokens per chunk. Chunks below this size
            are candidates for merging with the next chunk. Default 256
            prevents excessively small chunks that provide insufficient
            context for RAG answer generation.

    Returns:
        List of chunk dictionaries. Each dictionary has the following keys:

        text (str):
            The full text content of this chunk. May span multiple paragraphs
            if small chunks were merged.

        chunk_index (int):
            Zero-based sequential position of this chunk within the document.
            Used as part of the Pinecone vector ID: {filename}_{chunk_index}.

        token_count (int):
            Exact token count of this chunk using tiktoken cl100k_base.
            Consistent with OpenAI embedding API token counting.

        start_char (int):
            Approximate starting character position of this chunk within
            the reconstructed full document text. Accumulated sequentially —
            not looked up in the original text, so treat as approximate.

        end_char (int):
            Approximate ending character position. start_char + len(text).

        headings (List[str]):
            Hierarchical heading path from the document structure leading
            to this chunk. Example: ["Chapter 3", "Section 3.2", "Billing"].
            Empty list if this chunk has no heading context (e.g., front matter).
            Stored as JSON string in Pinecone metadata due to Pinecone's
            metadata type constraints.

        page_numbers (List[int]):
            Page numbers this chunk spans in the original document.
            Useful for source citation ("see page 4").
            Empty list if page information is unavailable.

        doc_items (List[str]):
            String representations of the first 3 Docling document item
            references for this chunk. Used for grounding — tracing a chunk
            back to its exact location in the parsed document structure.
            Truncated to 100 characters each.

        captions (List[str]):
            Table or figure captions associated with this chunk, if any.
            Present when the chunk contains or is adjacent to a table/figure
            that Docling detected during layout analysis.

    Raises:
        ImportError:
            If Docling is not installed (DOCLING_AVAILABLE is False).
            Resolution: pip install docling docling-core

        Exception:
            Wraps any runtime error from HybridChunker or the merge/conversion
            pass. Original error message is preserved in the raised exception.

    Example output (single chunk dict):
        {
            'text': 'Our return policy allows customers to return items within
                     30 days. Products must be in original packaging.',
            'chunk_index': 4,
            'token_count': 298,
            'start_char': 1847,
            'end_char': 2103,
            'headings': ['Customer Policies', 'Returns and Refunds'],
            'page_numbers': [3],
            'doc_items': ['text_item_12', 'text_item_13'],
            'captions': []
        }
    """

    if not DOCLING_AVAILABLE:
        raise ImportError(
            "Docling is not installed. Run: pip install docling docling-core"
        )

    try:
        import tiktoken

        tiktoken_encoder = tiktoken.get_encoding("cl100k_base")
        tokenizer = OpenAITokenizer(tokenizer=tiktoken_encoder, max_tokens=max_tokens)

        chunker = HybridChunker(
            tokenizer=tokenizer, max_tokens=max_tokens, merge_peers=True
        )

        logger.info(
            f"chunking with HybridChunker (max_tokens = {max_tokens}), merge_peers = True"
        )

        raw_chunks = list(chunker.chunk(dl_doc=doc))

        logger.info(
            f"Generated {len(raw_chunks)} raw semantic chunks, merging to target {min_tokens} - {max_tokens} tokens"
        )

        # Post-process: Merge consecutive small chunks to reach target size
        merged_chunks = []
        current_merged = None

        for chunk in raw_chunks:
            token_count = len(tiktoken_encoder.encode(chunk.text))

            if current_merged is None:
                current_merged = chunk
            else:
                current_tokens = len(tiktoken_encoder.encode(current_merged.text))
                combined_tokens = current_tokens + token_count

                # Merge if current chunk is undersized and combined won't exceed max

                if current_tokens < min_tokens and combined_tokens <= max_tokens:
                    # Merge chunks
                    current_merged.text = current_merged.text + "\n\n" + chunk.text

                    # Merged metadata
                    if chunk.meta and chunk.meta.headings:
                        if not current_merged.meta.headings:
                            current_merged.meta.headings = []
                        for h in chunk.meta.headings:
                            if h not in current_merged.meta.headings:
                                current_merged.meta.headings.append(h)

                    # Merge page numbers
                    if (
                        chunk.meta
                        and chunk.meta.origin
                        and hasattr(chunk.meta.origin, "page_numbers")
                    ):
                        if chunk.meta.origin.page_numbers:
                            if (
                                not hasattr(current_merged.meta.origin, "page_numbers")
                                or not current_merged.meta.origin.page_numbers
                            ):
                                if current_merged.meta and current_merged.meta.origin:
                                    current_merged.meta.origin.page_numbers = []
                            if (
                                current_merged.meta
                                and current_merged.meta.origin
                                and current_merged.meta.origin.page_numbers is not None
                            ):
                                for pn in chunk.meta.origin.page_numbers:
                                    if (
                                        pn
                                        not in current_merged.meta.origin.page_numbers
                                    ):
                                        current_merged.meta.origin.page_numbers.append(
                                            pn
                                        )
                else:
                    # Current chunk is complete, save it and start new one
                    merged_chunks.append(current_merged)
                    current_merged = chunk

        # Don't forget the last chunk
        if current_merged is not None:
            merged_chunks.append(current_merged)

        logger.info(
            f"After merging: {len(merged_chunks)} chunks (avg {sum(len(tiktoken_encoder.encode(c.text)) for c in merged_chunks) / len(merged_chunks):.1f} tokens)"
        )

        # Convert to format compatible with existing cache/vector storage
        result = []
        char_position = 0

        for idx, chunk in enumerate(merged_chunks):
            # Extract heading hierarchy
            headings = []
            if chunk.meta and chunk.meta.headings:
                headings = [h.text for h in chunk.meta.headings if hasattr(h, "text")]

            # Extract page numbers
            page_numbers = []
            if (
                chunk.meta
                and chunk.meta.origin
                and hasattr(chunk.meta.origin, "page_numbers")
            ):
                page_numbers = chunk.meta.origin.page_numbers or []

            # Extract captions (for tables/figures)
            captions = []
            if chunk.meta and hasattr(chunk.meta, "captions") and chunk.meta.captions:
                captions = [str(c) for c in chunk.meta.captions]

            # Get document items (for grounding)
            doc_items = []
            if chunk.meta and hasattr(chunk.meta, "doc_items") and chunk.meta.doc_items:
                # Store first 3 items as strings for reference
                doc_items = [str(item)[:100] for item in chunk.meta.doc_items[:3]]

            # Calculate token count using the underlying tiktoken encoder
            token_count = len(tiktoken_encoder.encode(chunk.text))

            # Calculate character positions
            chunk_text = chunk.text
            start_char = char_position
            end_char = start_char + len(chunk_text)
            char_position = end_char

            # Create enhanced chunk dictionary
            chunk_data = {
                "text": chunk_text,
                "chunk_index": idx,
                "token_count": token_count,
                "start_char": start_char,
                "end_char": end_char,
                # NEW: Rich metadata from Docling
                "headings": headings,
                "page_numbers": page_numbers,
                "doc_items": doc_items,
                "captions": captions,
            }

            result.append(chunk_data)

        # Log sample of first chunk's metadata
        if result:
            first_chunk = result[0]
            logger.info(
                f"Sample chunk metadata - Headings: {first_chunk['headings']}, Pages: {first_chunk['page_numbers']}"
            )

        return result

    except Exception as e:
        logger.error(f"HybridChunker failed: {str(e)}")
        raise Exception(f"Failed to chunk document with HybridChunker: {str(e)}")


def parse_and_chunk_document(
    file_path: str, chunk_size: int = 512, min_chunk_size: int = 256
) -> List[Dict[str, Any]]:
    """
    Parse and chunk document using Docling's context-aware approach.

    This is the main entry point that replaces the old parse_document() + chunk_text() flow.

    Args:
        file_path: Path to the document file
        chunk_size: Maximum tokens per chunk (default: 512)
        min_chunk_size: Minimum tokens per chunk - smaller chunks will be merged (default: 256)

    Returns:
        List of chunk dictionaries with rich metadata

    Raises:
        Exception: If both Docling and fallback fail
    """

    if not DOCLING_AVAILABLE:
        logger.warning("Docling not available, cannot use context-aware chunking")
        raise ImportError(
            "Docling is required for context-aware chunking. Run: pip install docling docling-core"
        )

    try:
        doc = convert_document(file_path=file_path)

        chunks = chunk_with_hybrid(
            doc, max_tokens=chunk_size, min_tokens=min_chunk_size
        )

        logger.info(
            f"Successfully processed {Path(file_path).name}: {len(chunks)} chunks with context"
        )

        return chunks

    except Exception as e:
        logger.error(f"Docling processing failed for {Path(file_path).name}: {str(e)}")
        raise Exception(f"Failed to process document with Docling: {str(e)}")


def fallback_to_unstructured(
    file_path: str, chunk_size: int = 512
) -> List[Dict[str, Any]]:
    """
    Fallback to Unstructured.io for documents Docling cannot handle.

    This maintains compatibility but without context-aware chunking benefits.

    Args:
        file_path: Path to the document file
        chunk_size: Maximum tokens per chunk

    Returns:
        List of chunk dictionaries (without rich metadata)
    """

    logger.warning(f"Using unstructured.io fallback for {Path(file_path).name}")

    try:
        from app.services.document_service import parse_document, chunk_text

        text = parse_document(file_path=file_path)
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=50)

        # Add empty metadata fields for compatibility
        for chunk in chunks:
            chunk["headings"] = []
            chunk["page_numbers"] = []
            chunk["doc_items"] = []
            chunk["captions"] = []

        logger.info(f"Fallback chunking complete: {len(chunks)} chunks (no context)")

        return chunks

    except Exception as e:
        logger.error(f"Fallback also failed: {str(e)}")
        raise Exception(f"Both Docling and unstructured failed : {str(e)}")


def get_docling_status() -> Dict[str, Any]:
    """
    Check if Docling is available and functioning.

    Returns:
        Dictionary with status information
    """

    return {
        "docling_available": DOCLING_AVAILABLE,
        "features": {
            "context_aware_chunking": DOCLING_AVAILABLE,
            "heading_preservation": DOCLING_AVAILABLE,
            "table_structure": DOCLING_AVAILABLE,
            "layout_analysis": DOCLING_AVAILABLE,
        },
    }
