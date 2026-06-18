import re
import os
import sys
import math
import pdfplumber


def _split_text_evenly(text, max_chunk_length=2000, min_chunk_length=1000, separators=None):
    if separators is None:
        separators = [r"\n\s*\n", r"\. ", r"\n\n", r"\n", r"\t", r", ", r" "]

    best_chunks = None
    best_score = float("inf")

    for sep in separators:
        matches = list(re.finditer(sep, text))
        if not matches:
            continue

        split_points = [m.end() for m in matches]
        split_points.append(len(text))

        chunks = []
        prev = 0
        total_length = len(text)
        target_chunks = max(total_length // max_chunk_length, 1)
        ideal_chunk_length = total_length / target_chunks

        for p in split_points:
            current_length = p - prev
            if current_length > max_chunk_length:
                chunks.append(text[prev:p])
                prev = p
            elif current_length >= min_chunk_length and abs(current_length - ideal_chunk_length) < ideal_chunk_length * 0.5:
                chunks.append(text[prev:p])
                prev = p

        if prev < len(text):
            chunks.append(text[prev:])

        lengths = [len(c) for c in chunks]
        if max(lengths) <= max_chunk_length and all(len(c) >= min_chunk_length for c in chunks[:-1]):
            avg = sum(lengths) / len(lengths)
            score = sum((l - avg) ** 2 for l in lengths) / len(lengths)
            if score < best_score:
                best_chunks = chunks
                best_score = score

    return best_chunks if best_chunks else [text]


def _normalize(line):
    line = re.sub(r"\d+", "#", line.strip())   # 2 y 3 -> # para que matcheen
    return re.sub(r"\s+", " ", line)


def _detect_noise(pages, min_fraction=0.5):
    """Lineas que se repiten (normalizadas) en >= min_fraction de las paginas = header/footer."""
    from collections import defaultdict
    seen = defaultdict(set)
    for i, page in enumerate(pages):
        for line in page.split("\n"):
            norm = _normalize(line)
            if norm:
                seen[norm].add(i)
    threshold = max(2, math.ceil(len(pages) * min_fraction))
    return {norm for norm, page_set in seen.items() if len(page_set) >= threshold}


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