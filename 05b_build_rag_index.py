"""
05b_build_rag_index.py
======================
Builds the FAISS vector index from all policy .docx files.

This script must be run ONCE before 05_run_llm_experiments.py.
It outputs two files:
  - data/rag/policy_faiss.index  : The FAISS flat-IP (cosine) index
  - data/rag/policy_chunks.json  : All chunk text + metadata

Design decisions:
  - Uses python-docx (not raw XML zip) to correctly read tables.
  - Converts each document to Markdown, preserving table structure (| col | col |).
  - Chunks by heading/section so each chunk is semantically coherent.
  - Tags every chunk with: policy_code, policy_name, year, section_title, chunk_id.
  - The ground-truth decision document is EXCLUDED from the index (its rules are
    hardcoded into the System Prompt — Requirement #4).
  - Embeds with sentence-transformers all-MiniLM-L6-v2 (small, offline, free).
  - Normalises embeddings for cosine similarity via FAISS IndexFlatIP.

Dependencies (add to requirements.txt):
  python-docx
  sentence-transformers
  faiss-cpu
"""

print("Building RAG index... (Loading libraries)")

import os
import re
import json
import numpy as np

try:
    from docx import Document
    from docx.oxml.ns import qn
except ImportError:
    raise ImportError("python-docx not installed. Run: pip install python-docx")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("sentence-transformers not installed. Run: pip install sentence-transformers")

try:
    import faiss
except ImportError:
    raise ImportError("faiss-cpu not installed. Run: pip install faiss-cpu")


# ==============================================================================
# CONFIGURATION
# ==============================================================================

POLICY_DOCS_DIR = "policy_documents"
OUTPUT_DIR      = "data/rag"
INDEX_PATH      = os.path.join(OUTPUT_DIR, "policy_faiss.index")
CHUNKS_PATH     = os.path.join(OUTPUT_DIR, "policy_chunks.json")
EMBED_MODEL     = "all-MiniLM-L6-v2"   # small, fast, runs fully offline
MIN_CHUNK_CHARS = 80                    # drop chunks smaller than this

# EXCLUDED: Decision matrix hardcoded into System Prompt (Requirement #4)
EXCLUDED_FILES = {"Rule_Based_Decision_Engine_Frozen_Ground_Truth.docx"}

# Filename pattern: POL-XX_YYYY-N_Policy_Name.docx
FILENAME_PATTERN = re.compile(r"^(POL-\d+)_(\d{4})-\d+_(.+?)\.docx$")


# ==============================================================================
# STEP 1: PARSE DOCX TO MARKDOWN (preserving tables)
# ==============================================================================

def _table_to_markdown(table) -> str:
    """Convert a python-docx Table to a Markdown table string (| col | col |)."""
    rows = []
    for i, row in enumerate(table.rows):
        cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
        rows.append("| " + " | ".join(cells) + " |")
        if i == 0:
            rows.append("| " + " | ".join(["---"] * len(cells)) + " |")
    return "\n".join(rows)


def docx_to_markdown(doc_path: str) -> str:
    """
    Parse a .docx file to Markdown.
    Headings -> # / ## / ###
    Tables   -> Markdown pipe tables (| col | col |)
    Paragraphs -> plain text
    """
    doc  = Document(doc_path)
    body = doc.element.body
    lines = []

    for child in body:
        tag = child.tag.split("}")[-1]

        if tag == "p":
            para_text = "".join(
                node.text for node in child.iter(qn("w:t")) if node.text
            ).strip()
            if not para_text:
                continue

            style_elem = child.find(
                ".//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}pStyle"
            )
            style_val = ""
            if style_elem is not None:
                style_val = style_elem.get(
                    "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val", ""
                )

            if "Heading1" in style_val or style_val == "1":
                lines.append(f"# {para_text}")
            elif "Heading2" in style_val or style_val == "2":
                lines.append(f"## {para_text}")
            elif "Heading3" in style_val or style_val == "3":
                lines.append(f"### {para_text}")
            else:
                lines.append(para_text)

        elif tag == "tbl":
            for tbl in doc.tables:
                if tbl._tbl is child:
                    lines.append("")
                    lines.append(_table_to_markdown(tbl))
                    lines.append("")
                    break

    return "\n".join(lines)


# ==============================================================================
# STEP 2: CHUNK MARKDOWN BY SECTION HEADING
# ==============================================================================

def chunk_markdown(markdown_text: str, policy_code: str,
                   policy_name: str, year: int, filename: str) -> list:
    """
    Split Markdown into chunks on heading boundaries.
    Each chunk carries full metadata and a policy-citation header
    so the LLM always knows which policy section it is reading.
    """
    heading_pattern = re.compile(r"^(#{1,3} .+)$", re.MULTILINE)
    parts = heading_pattern.split(markdown_text)

    chunks  = []
    chunk_id = 0
    current_heading = f"{policy_name} — Introduction"
    current_body    = ""

    def flush(heading, body):
        nonlocal chunk_id
        body = body.strip()
        if len(body) >= MIN_CHUNK_CHARS:
            # Prepend a citation header so the LLM knows provenance
            citation = f"[SOURCE: {policy_code} | Year: {year} | Section: {heading}]"
            chunks.append({
                "chunk_id":      f"{policy_code}_{year}_{chunk_id:03d}",
                "policy_code":   policy_code,
                "policy_name":   policy_name,
                "year":          year,
                "source_file":   filename,
                "section_title": heading,
                "text":          f"{citation}\n\n{body}",
            })
            chunk_id += 1

    for part in parts:
        part = part.strip()
        if not part:
            continue
        if re.match(r"^#{1,3} ", part):
            flush(current_heading, current_body)
            current_heading = part.lstrip("#").strip()
            current_body    = ""
        else:
            current_body += "\n" + part

    flush(current_heading, current_body)
    return chunks


# ==============================================================================
# STEP 3: PARSE METADATA FROM FILENAME
# ==============================================================================

def parse_filename_metadata(filename: str):
    match = FILENAME_PATTERN.match(filename)
    if not match:
        return None
    return {
        "policy_code": match.group(1),
        "year":        int(match.group(2)),
        "policy_name": match.group(3).replace("_", " "),
    }


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_chunks = []

    # ── 1. Parse every eligible policy document ───────────────────────────────
    doc_files = sorted(os.listdir(POLICY_DOCS_DIR))
    print(f"\nFound {len(doc_files)} files in '{POLICY_DOCS_DIR}/'")

    for filename in doc_files:
        if not filename.endswith(".docx"):
            continue
        if filename in EXCLUDED_FILES:
            print(f"  SKIP (excluded from RAG): {filename}")
            continue

        meta = parse_filename_metadata(filename)
        if meta is None:
            print(f"  WARN: Cannot parse metadata from '{filename}' — skipping.")
            continue

        doc_path = os.path.join(POLICY_DOCS_DIR, filename)
        print(f"  Parsing: {filename}")
        print(f"    -> Policy: {meta['policy_code']} | Year: {meta['year']} | Name: {meta['policy_name']}")

        markdown = docx_to_markdown(doc_path)
        chunks   = chunk_markdown(
            markdown_text = markdown,
            policy_code   = meta["policy_code"],
            policy_name   = meta["policy_name"],
            year          = meta["year"],
            filename      = filename,
        )
        print(f"    -> {len(chunks)} chunks extracted")
        all_chunks.extend(chunks)

    if not all_chunks:
        print("\nERROR: No chunks were produced. Verify policy_documents/ has valid .docx files.")
        return

    print(f"\nTotal chunks: {len(all_chunks)}")

    # ── 2. Embed all chunks ───────────────────────────────────────────────────
    print(f"\nLoading embedding model '{EMBED_MODEL}' (downloads on first run)...")
    model = SentenceTransformer(EMBED_MODEL)

    texts      = [c["text"] for c in all_chunks]
    print(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype=np.float32)

    # ── 3. Build FAISS index (Inner Product on L2-normalised = cosine sim) ────
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"\nFAISS index built: {index.ntotal} vectors, dim={dim}")

    # ── 4. Save index + metadata ──────────────────────────────────────────────
    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"\nFAISS index saved  -> {INDEX_PATH}")
    print(f"Chunk metadata saved -> {CHUNKS_PATH}")

    # ── 5. Summary ────────────────────────────────────────────────────────────
    from collections import Counter
    summary = Counter(f"{c['policy_code']} ({c['year']})" for c in all_chunks)
    print("\n── Chunk Distribution ──")
    for key, count in sorted(summary.items()):
        print(f"  {key:35s}: {count} chunks")
    print("\nDone. Run 05_run_llm_experiments.py next.\n")


if __name__ == "__main__":
    main()
