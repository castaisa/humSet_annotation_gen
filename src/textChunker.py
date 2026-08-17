import re
import os
import sys
import math
import pdfplumber


def _split_text_evenly(text, max_chunk_length=2000, min_chunk_length=1000, separators=None,
                       overlap_sentences=1):
    """Split `text` into chunks of at most `max_chunk_length` chars (before overlap
    is prepended), cutting only at the highest-priority separator available
    (paragraph > sentence > line > clause), never mid-sentence just to even out lengths.

    The old version scored every separator by chunk-length variance, which meant the
    plain-space separator always won (it allows perfectly even chunks) and every cut
    landed mid-sentence — producing extraction errors like a quantity in one chunk
    and its predicate in the next. Boundary quality now beats evenness.

    `overlap_sentences` prepends the last N sentences of the previous chunk to the
    next one, so facts straddling a boundary are seen whole at least once.
    """
    if separators is None:
        # strict priority order: paragraph breaks, sentence ends, newlines, clauses
        separators = [r"\n\s*\n", r"(?<=[.!?])\s+", r"\n", r"(?<=[;:,])\s+", r" "]

    for sep in separators:
        matches = list(re.finditer(sep, text))
        split_points = [m.end() for m in matches] + [len(text)]

        chunks = []
        prev = 0
        last_valid = None
        for p in split_points:
            if p - prev > max_chunk_length:
                # cut at the last separator that kept us within the limit
                if last_valid is None:
                    chunks = None  # a single segment exceeds the limit: try finer separator
                    break
                chunks.append(text[prev:last_valid])
                prev = last_valid
            last_valid = p
        if chunks is None:
            continue
        if prev < len(text):
            chunks.append(text[prev:])

        if all(len(c) <= max_chunk_length for c in chunks):
            if overlap_sentences and len(chunks) > 1:
                chunks = _add_overlap(chunks, overlap_sentences)
            return chunks

    return [text]


def _add_overlap(chunks, n_sentences):
    """Prepend the last `n_sentences` sentences of chunk i-1 to chunk i (marked so
    downstream deduplication can recognize repeated context)."""
    out = [chunks[0]]
    for prev, cur in zip(chunks, chunks[1:]):
        sentences = re.split(r"(?<=[.!?])\s+", prev.strip())
        tail = " ".join(sentences[-n_sentences:]) if sentences else ""
        out.append((tail + "\n" + cur) if tail else cur)
    return out


def _normalize(line):
    line = re.sub(r"\d+", "#", line.strip())   # 2 y 3 -> # para que matcheen
    return re.sub(r"\s+", " ", line)


def _is_section_header(line):
    """Short ALL-CAPS lines (JAMAICA, CUBA, HAITI, RESPONSE, KEY POINTS...) are section
    headers, not page furniture — they may legitimately repeat across pages (e.g. one
    IMPACT and one RESPONSE section per country) and carry the location context that
    downstream extraction needs. Never treat them as noise."""
    stripped = line.strip()
    return (0 < len(stripped) <= 40
            and stripped == stripped.upper()
            and any(c.isalpha() for c in stripped)
            and not any(c.isdigit() for c in stripped))


def _detect_noise(pages, min_fraction=0.5):
    """Lineas que se repiten (normalizadas) en >= min_fraction de las paginas = header/footer.
    Section headers (short ALL-CAPS lines) are exempt even when repeated."""
    from collections import defaultdict
    seen = defaultdict(set)
    headers = set()
    for i, page in enumerate(pages):
        for line in page.split("\n"):
            norm = _normalize(line)
            if norm:
                seen[norm].add(i)
                if _is_section_header(line):
                    headers.add(norm)
    threshold = max(2, math.ceil(len(pages) * min_fraction))
    return {norm for norm, page_set in seen.items()
            if len(page_set) >= threshold and norm not in headers}


def extract_clean_text(path, min_fraction=0.5):
    with pdfplumber.open(path) as pdf:
        pages = [(p.extract_text() or "") for p in pdf.pages]

    noise = _detect_noise(pages, min_fraction=min_fraction)
    clean_pages = []
    for page in pages:
        kept = [l for l in page.split("\n") if _normalize(l) not in noise]
        clean_pages.append("\n".join(kept))
    return "\n".join(clean_pages)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "documento.pdf"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "chunks"

    raw = extract_clean_text(path)
    chunks = _split_text_evenly(raw, max_chunk_length=2000, min_chunk_length=1000)

    base = os.path.splitext(os.path.basename(path))[0]
    os.makedirs(out_dir, exist_ok=True)

    for i, c in enumerate(chunks):
        fname = os.path.join(out_dir, f"{base}_chunk_{i:03d}.txt")
        with open(fname, "w", encoding="utf-8") as f:
            f.write(c)

    print(f"Texto limpio: {len(raw)} chars")
    print(f"Guardados {len(chunks)} chunks en: {os.path.abspath(out_dir)}/")