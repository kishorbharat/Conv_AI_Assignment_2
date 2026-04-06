#!/usr/bin/env python3
"""
Pre-training a GPT-Style (Decoder-Only) LLM from Scratch.

Implements all six phases in one runnable pipeline:
1. PDF data extraction and curation
2. Dataset generation (custom BPE tokenizer + shifted CLM labels)
3. Input embeddings (token + sinusoidal positional encodings)
4. Decoder-only Transformer forward pass with causal masking
5. Cross-entropy loss + AdamW backpropagation
6. Autoregressive text generation
"""

import argparse
import importlib
import json
import math
import random
import re
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

# Suppress harmless NumPy initialisation warning from PyTorch internals
warnings.filterwarnings("ignore", message="Failed to initialize NumPy")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:  # pragma: no cover
    print("This script requires PyTorch. Install it with: pip install torch")
    raise SystemExit(1) from exc


ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = ROOT / "data" / "pdfs"
PROCESSED_DIR = ROOT / "data" / "processed"
RAW_TEXT_PATH = PROCESSED_DIR / "ais175_raw.txt"
CLEAN_TEXT_PATH = PROCESSED_DIR / "ais175_clean.txt"
TOKENIZER_DIR = ROOT / "tokenizer"
CHECKPOINT_PATH = ROOT / "artifacts_group1" / "decoder_only_15m.pt"


def ensure_dirs():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)


def extract_pdfs_to_raw_text(pdf_dir: Path, output_file: Path) -> None:
    """Step 1: Extract text page-by-page from all PDFs into one raw text file."""
    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF folder not found: {pdf_dir}")

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {pdf_dir}")

    fitz_mod = None
    pdfplumber_mod = None
    try:
        fitz_mod = importlib.import_module("fitz")
        backend = "pymupdf"
    except Exception:
        try:
            pdfplumber_mod = importlib.import_module("pdfplumber")
            backend = "pdfplumber"
        except Exception as exc:
            raise RuntimeError(
                "Install a PDF parser first: pip install pymupdf OR pip install pdfplumber"
            ) from exc

    collected = []
    for pdf_path in pdf_files:
        collected.append(f"\n\n===== FILE: {pdf_path.name} =====\n")
        if backend == "pymupdf":
            with fitz_mod.open(pdf_path) as doc:
                for i, page in enumerate(doc, start=1):
                    page_text = page.get_text("text") or ""
                    collected.append(f"\n--- PAGE {i} ---\n{page_text}\n")
        else:
            with pdfplumber_mod.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages, start=1):
                    page_text = page.extract_text() or ""
                    collected.append(f"\n--- PAGE {i} ---\n{page_text}\n")

    output_file.write_text("\n".join(collected), encoding="utf-8")
    print(f"Extracted {len(pdf_files)} PDFs -> {output_file}")


def _remove_page_artifacts(line: str) -> bool:
    s = line.strip().lower()
    if not s:
        return True
    if s.startswith("### file:"):
        return True
    if s in {"draft ais 175 / final draft", "march 2025", "appendix", "annexes part a", "annexes part b", "annexes part c"}:
        return True
    if re.fullmatch(r"page\s*\d+(\s*of\s*\d+)?", s):
        return True
    if re.fullmatch(r"\d+", s):
        return True
    if re.fullmatch(r"[-_ ]{3,}", s):
        return True
    # Typical table-of-contents lines ending with page numbers.
    if re.fullmatch(r".*\s\d{1,4}", s) and len(s.split()) >= 3 and any(k in s for k in ["annex", "appendix", "scope", "definitions", "approval", "test"]):
        return True
    return False


def clean_raw_text(raw_path: Path, clean_path: Path) -> None:
    """Step 1: Clean extraction noise (page markers, broken lines, TOC artifacts)."""
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw text file not found: {raw_path}")

    raw = raw_path.read_text(encoding="utf-8", errors="ignore")
    raw = re.sub(r"\r\n?", "\n", raw)

    lines = []
    for ln in raw.split("\n"):
        if ln.strip().startswith("--- PAGE"):
            continue
        if ln.strip().startswith("===== FILE"):
            continue
        if _remove_page_artifacts(ln):
            continue
        lines.append(ln.rstrip())

    text = "\n".join(lines)

    # Fix common PDF line-break issues.
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Preserve heading boundaries while flattening broken line wraps.
    text = re.sub(r"(?<!\n)\n(?!\n)(?=[a-z0-9])", " ", text)
    text = re.sub(r"(?<!\n)\n(?!\n)(?=[A-Z]\w+\s[A-Z]\w+)", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    # Drop leftover marker-heavy fragments.
    text = re.sub(r"#{2,}.*", "", text)
    text = re.sub(r"\bfile:[^\n]+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n\s*\n", "\n\n", text)
    text = text.strip()

    clean_path.write_text(text + "\n", encoding="utf-8")
    print(f"Cleaned corpus -> {clean_path} ({len(text):,} chars)")


def basic_pretokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[^\w\s]", text.lower())


def _merge_word_symbols(word: Tuple[str, ...], pair: Tuple[str, str]) -> Tuple[str, ...]:
    out = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
            out.append(word[i] + word[i + 1])
            i += 2
        else:
            out.append(word[i])
            i += 1
    return tuple(out)


def train_bpe_tokenizer_from_scratch(
    text: str,
    vocab_size: int = 4096,
    min_frequency: int = 2,
) -> Tuple[Dict[str, int], List[Tuple[str, str]]]:
    """Step 2: Train a simple Word-BPE tokenizer (no pre-made tokenizer used)."""
    words = basic_pretokenize(text)
    if not words:
        raise ValueError("No tokens found in clean corpus for tokenizer training.")

    vocab_words = Counter(tuple(list(w) + ["</w>"]) for w in words)
    merges: List[Tuple[str, str]] = []

    while True:
        pair_counts = Counter()
        for word_tuple, freq in vocab_words.items():
            for i in range(len(word_tuple) - 1):
                pair_counts[(word_tuple[i], word_tuple[i + 1])] += freq

        if not pair_counts:
            break

        best_pair, best_freq = pair_counts.most_common(1)[0]
        if best_freq < min_frequency:
            break

        merges.append(best_pair)
        vocab_words = Counter({
            _merge_word_symbols(word_tuple, best_pair): freq for word_tuple, freq in vocab_words.items()
        })

        symbol_inventory = set()
        for word_tuple in vocab_words:
            symbol_inventory.update(word_tuple)
        if len(symbol_inventory) >= vocab_size:
            break

    specials = ["<pad>", "<unk>", "<bos>", "<eos>"]
    token_freq = Counter()
    for word_tuple, freq in vocab_words.items():
        for piece in word_tuple:
            token_freq[piece] += freq

    sorted_tokens = [tok for tok, _ in token_freq.most_common(max(0, vocab_size - len(specials)))]
    vocab_list = specials + sorted_tokens
    vocab = {tok: i for i, tok in enumerate(vocab_list)}
    return vocab, merges


def save_tokenizer(vocab: Dict[str, int], merges: List[Tuple[str, str]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "vocab.json").write_text(json.dumps(vocab, indent=2), encoding="utf-8")
    (out_dir / "merges.txt").write_text(
        "\n".join(f"{a} {b}" for a, b in merges),
        encoding="utf-8",
    )
    print(f"Saved tokenizer files in {out_dir}")


def load_tokenizer(in_dir: Path) -> Tuple[Dict[str, int], Dict[int, str], List[Tuple[str, str]], Dict[Tuple[str, str], int]]:
    vocab_path = in_dir / "vocab.json"
    merges_path = in_dir / "merges.txt"
    if not vocab_path.exists() or not merges_path.exists():
        raise FileNotFoundError(f"Tokenizer files not found in {in_dir}")

    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    id_to_token = {idx: tok for tok, idx in vocab.items()}
    merges = []
    for line in merges_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        a, b = line.split(" ", 1)
        merges.append((a, b))
    merge_ranks = {pair: i for i, pair in enumerate(merges)}
    return vocab, id_to_token, merges, merge_ranks


def bpe_encode_word(word: str, merge_ranks: Dict[Tuple[str, str], int]) -> List[str]:
    tokens = list(word) + ["</w>"]
    while len(tokens) > 1:
        pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
        ranked = [(merge_ranks[p], p) for p in pairs if p in merge_ranks]
        if not ranked:
            break
        _, best = min(ranked, key=lambda x: x[0])
        merged = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best:
                merged.append(tokens[i] + tokens[i + 1])
                i += 2
            else:
                merged.append(tokens[i])
                i += 1
        tokens = merged
    return tokens


def encode_text(text: str, vocab: Dict[str, int], merge_ranks: Dict[Tuple[str, str], int]) -> List[int]:
    tokens = basic_pretokenize(text)
    unk_id = vocab["<unk>"]
    ids: List[int] = []
    for word in tokens:
        for piece in bpe_encode_word(word, merge_ranks):
            ids.append(vocab.get(piece, unk_id))
    return ids


def decode_ids(ids: Sequence[int], id_to_token: Dict[int, str]) -> str:
    out = []
    specials = {"<pad>", "<unk>", "<bos>", "<eos>"}
    for i in ids:
        tok = id_to_token.get(int(i), "<unk>")
        if tok in specials:
            continue
        if tok.endswith("</w>"):
            out.append(tok[:-4] + " ")
        else:
            out.append(tok)
    return "".join(out).strip()


def _split_sentences(text: str) -> List[str]:
    sents = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sents if s.strip()]


def _extract_references(text: str) -> List[str]:
    """Extract paragraph/annex/appendix reference strings from a block of text."""
    refs: List[str] = []
    # paragraph X.X.X (with optional trailing dot)
    refs += re.findall(r"\bparagraph\s+\d[\d.]*", text, flags=re.IGNORECASE)
    # clause X.X.X
    refs += re.findall(r"\bclause\s+\d[\d.]*", text, flags=re.IGNORECASE)
    # Annex B2 / Annex C6 / Annex B / Annex C, etc.
    refs += re.findall(r"\bAnnex\s+[A-Z]\d*", text, flags=re.IGNORECASE)
    # Appendix N (standalone digit)
    refs += re.findall(r"\bAppendix\s+\d+", text, flags=re.IGNORECASE)
    # section X.X
    refs += re.findall(r"\bsection\s+\d[\d.]*", text, flags=re.IGNORECASE)
    # deduplicate while preserving order
    seen: set = set()
    unique: List[str] = []
    for r in refs:
        key = r.strip().lower()
        if key not in seen:
            seen.add(key)
            unique.append(r.strip())
    return unique


def _fetch_section_content(ref: str, clean_text: str, window: int = 600) -> str:
    """Return a short excerpt from the corpus that starts at the cited reference."""
    # Build a search pattern from the reference string:
    # e.g. "paragraph 4.3" -> r"\bparagraph\s+4\.3\b"
    # e.g. "Annex C6" -> r"\bAnnex\s+C6\b"
    escaped = re.escape(ref)
    # Allow flexible whitespace between word and number
    pat = re.sub(r"\\ ", r"\\s+", escaped)
    m = re.search(pat, clean_text, flags=re.IGNORECASE)
    if not m:
        return ""
    snippet = clean_text[m.start(): m.start() + window]
    # Return first 3 sentences for readability
    sents = _split_sentences(snippet)
    return " ".join(sents[:3]) if sents else snippet[:window]


def answer_from_corpus(question: str, clean_text: str, top_k: int = 3) -> str:
    """Return a readable extractive answer from corpus sentences for factual queries.

    Uses intent-based section targeting: maps question themes to known corpus
    section anchors so answers come from the right part of the document rather
    than a superficial keyword overlap across the whole text.
    """
    q_low = question.lower()

    # --- Intent → section anchor mapping -----------------------------------
    # Each entry: (list_of_trigger_phrases, list_of_regex_anchors_in_corpus)
    # The first anchor that matches is used; a 2000-char window is extracted.
    INTENT_MAP = [
        (
            ["covered", "scope", "applicable", "applies", "which vehicle",
             "what vehicle", "passenger car", "category", "commercial vehicle",
             "which car", "type of vehicle"],
            [r"1\.0\s+SCOPE", r"This Regulation applies to the type approval"],
        ),
        (
            # Tests / procedures performed for a road load family
            ["tests included in road", "tests in road load", "tests for road load",
             "road load test", "road load procedure", "road load determination",
             "how is road load determined", "methods for road load",
             "different tests", "coastdown", "wind tunnel", "torque meter",
             "running resistance"],
            [
                r"the road load of the representative vehicle is determined",
                r"coastdown method.*paragraph",
                r"road loads HR.*shall be determined",
            ],
        ),
        (
            ["road load family", "roadload family", "roadload"],
            [r"6\.3\.3\s+Road load family", r"Road load family"],
        ),
        (
            ["interpolation family", "interpolation"],
            [r"6\.3\.2\s+Interpolation family", r"6\.3\.2\.1\s+Interpolation family"],
        ),
        (
            ["criteria", "new vehicle", "joining", "include.*family", "add.*family",
             "criteria for including"],
            [r"6\.3\.2\.1\.2"],
        ),
        (
            ["flex fuel", "flex-fuel", "flexible fuel"],
            [r"3\.3\.25", r"Flex fuel vehicle"],
        ),
        (
            ["vehicle class", "wltc class", "wltp class", "class 1", "class 2",
             "class 3", "power to mass", "classification"],
            [r"2\.0 Vehicle classifications", r"2\.1 Class 1 vehicles"],
        ),
        (
            ["weight", "mass", "test mass", "vehicle mass", "laden mass"],
            [r"3\.2\.5\s", r"test mass of the vehicle", r"mass in running order"],
        ),
        (
            ["gear", "gear shift", "gear selection", "shift point"],
            [r"Annex B2", r"Gear selection"],
        ),
        (
            ["cop", "conformity of production"],
            [r"8\.0\s+Conformity", r"Conformity of production"],
        ),
        (
            ["isc", "in-service conformity", "in service conformity"],
            [r"9\.0\s+In", r"In.Service Conformity"],
        ),
        (
            ["obd", "on-board diagnostic", "on board diagnostic"],
            [r"Annex C5", r"On-Board Diagnostics"],
        ),
        (
            ["rde", "real driving emission", "real driving emissions", "pems"],
            [r"Annex C6", r"Real Driving Emission"],
        ),
        (
            ["evaporative", "type iv", "type 4", "evap"],
            [r"Annex C3", r"evaporative emissions"],
        ),
        (
            ["durability", "type v", "type 5"],
            [r"Annex C4", r"Type V test"],
        ),
        (
            ["midc", "difference", "earlier standard", "previous standard",
             "wltp vs", "wltp compared", "change from"],
            [r"The main motivation for", r"changes from the previous"],
        ),
        (
            ["approval", "type approval", "application for approval"],
            [r"4\.0\s+Application", r"5\.0\s+Approval", r"Application for approval"],
        ),
        (
            ["bi-fuel", "bi fuel", "bifuel"],
            [r"3\.3\.21", r"Bi-fuel vehicle"],
        ),
        (
            ["mono fuel", "mono-fuel"],
            [r"3\.3\.27", r"Mono-fuel vehicle"],
        ),
        (
            ["defeat device"],
            [r"3\.5\.7", r"Defeat device"],
        ),
    ]

    for triggers, anchors in INTENT_MAP:
        if any(t in q_low for t in triggers):
            for pat in anchors:
                m = re.search(pat, clean_text, flags=re.IGNORECASE)
                if m:
                    window = clean_text[m.start(): m.start() + 2000]
                    sents = _split_sentences(window)
                    if sents:
                        return " ".join(sents[:min(top_k, len(sents))])

    # --- Fallback: scored keyword overlap across all sentences --------------
    q_terms_all = basic_pretokenize(question)
    stopwords = {
        "what", "is", "the", "a", "an", "and", "or", "to", "of", "in", "for", "on", "at", "by",
        "with", "from", "that", "this", "it", "as", "be", "are", "was", "were", "shall", "do",
        "does", "how", "why", "when", "where", "which", "who", "all", "any",
    }
    q_terms = [t for t in q_terms_all if t not in stopwords and len(t) > 2]
    if not q_terms:
        q_terms = [t for t in q_terms_all if len(t) > 2]
    if not q_terms:
        return "Could not parse question terms."

    q_counts = Counter(q_terms)
    sents = _split_sentences(clean_text)
    if not sents:
        return "No sentence data available in clean corpus."

    scored: List[Tuple[float, str]] = []
    for s in sents:
        s_terms = basic_pretokenize(s)
        if not s_terms:
            continue
        s_counts = Counter(s_terms)
        overlap = sum(min(s_counts[t], q_counts[t]) for t in q_counts)
        if overlap == 0:
            continue
        coverage = overlap / max(1, len(q_terms))
        density = overlap / max(1, len(s_terms))
        has_all_terms = all(t in s_counts for t in q_counts)
        score = overlap + 1.5 * coverage + 0.5 * density + (3.0 if has_all_terms else 0.0)
        scored.append((score, s))

    if not scored:
        return "No strong match found in the clean corpus for this query."

    best = [s for _, s in sorted(scored, key=lambda x: x[0], reverse=True)[:top_k]]
    return " ".join(best)


class CLMDataset(Dataset):
    """Step 2: Builds shifted (X, Y) training pairs for CLM objective."""

    def __init__(self, token_ids: Sequence[int], block_size: int):
        if len(token_ids) <= block_size + 1:
            raise ValueError("Tokenized corpus is too small for the chosen block_size")
        self.data = torch.tensor(token_ids, dtype=torch.long)
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.data) - self.block_size - 1

    def __getitem__(self, idx: int):
        chunk = self.data[idx : idx + self.block_size + 1]
        x = chunk[:-1]  # [t1, t2, t3]
        y = chunk[1:]   # [t2, t3, t4]
        return x, y


class SinusoidalPositionalEncoding(nn.Module):
    """Step 3: Adds deterministic sine/cosine positional information."""

    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, causal_mask):
        attn_in = self.ln1(x)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, attn_mask=causal_mask, need_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x


class DecoderOnlyTransformer(nn.Module):
    """Step 4: Decoder-only transformer with causal self-attention."""

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        d_model: int = 384,
        n_head: int = 6,
        n_layer: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_encoding = SinusoidalPositionalEncoding(d_model=d_model, max_len=block_size)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            DecoderBlock(d_model=d_model, n_head=n_head, dropout=dropout) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, labels=None):
        bsz, seq_len = input_ids.shape
        if seq_len > self.block_size:
            raise ValueError(f"Sequence length {seq_len} exceeds block_size {self.block_size}")

        x = self.token_embedding(input_ids)
        x = self.position_encoding(x)
        x = self.dropout(x)

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool),
            diagonal=1,
        )

        for block in self.blocks:
            x = block(x, causal_mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            # Step 5: CLM loss over every time-step.
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = 40,
        greedy: bool = False,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
    ):
        """Step 6: Autoregressive generation loop."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = input_ids[:, -self.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)

            # Penalize tokens that already appeared to reduce repetitive loops.
            if repetition_penalty and repetition_penalty > 1.0:
                for b in range(input_ids.size(0)):
                    seen_ids = set(input_ids[b].tolist())
                    for tok_id in seen_ids:
                        if logits[b, tok_id] < 0:
                            logits[b, tok_id] *= repetition_penalty
                        else:
                            logits[b, tok_id] /= repetition_penalty

            # Prevent generating n-grams that already appeared in the sequence.
            if no_repeat_ngram_size and no_repeat_ngram_size > 1:
                n = no_repeat_ngram_size
                for b in range(input_ids.size(0)):
                    seq = input_ids[b].tolist()
                    if len(seq) < n - 1:
                        continue
                    prefix = tuple(seq[-(n - 1) :])
                    banned = set()
                    for i in range(len(seq) - n + 1):
                        ngram = tuple(seq[i : i + n])
                        if ngram[:-1] == prefix:
                            banned.add(ngram[-1])
                    if banned:
                        logits[b, list(banned)] = float("-inf")

            if greedy:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            elif top_k and top_k > 0:
                top_vals, top_idx = torch.topk(logits, k=min(top_k, logits.size(-1)), dim=-1)
                probs = F.softmax(top_vals, dim=-1)
                sampled = torch.multinomial(probs, num_samples=1)
                next_id = top_idx.gather(-1, sampled)
            else:
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_id], dim=1)
        return input_ids


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_clean_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Clean corpus not found: {path}")
    text = path.read_text(encoding="utf-8", errors="ignore")
    if len(text) < 2000:
        raise ValueError("Clean corpus is too small. Add more cleaned text before training.")
    return text


def evaluate(model, dataloader, device: str, max_batches: int = 100) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            if i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            losses.append(loss.item())
    model.train()
    return float(sum(losses) / max(1, len(losses)))


def train_model(args):
    ensure_dirs()
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    text = load_clean_text(Path(args.clean_text))
    vocab, _, _, merge_ranks = load_tokenizer(Path(args.tokenizer_dir))
    token_ids = encode_text(text, vocab, merge_ranks)

    split_idx = int(len(token_ids) * 0.9)
    train_ids = token_ids[:split_idx]
    val_ids = token_ids[split_idx:]

    train_dataset = CLMDataset(train_ids, block_size=args.block_size)
    val_dataset = CLMDataset(val_ids, block_size=args.block_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

    # Allow resuming: override arch args from checkpoint config if --resume
    checkpoint_path = Path(args.checkpoint)
    if getattr(args, 'resume', False) and checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        cfg = ckpt["config"]
        args.d_model = cfg["d_model"]
        args.n_head = cfg["n_head"]
        args.n_layer = cfg["n_layer"]
        args.dropout = cfg["dropout"]
        args.block_size = cfg["block_size"]
        print(f"Resuming from checkpoint: {checkpoint_path}")
        print(f"Loaded config: {cfg}")

    model = DecoderOnlyTransformer(
        vocab_size=len(vocab),
        block_size=args.block_size,
        d_model=args.d_model,
        n_head=args.n_head,
        n_layer=args.n_layer,
        dropout=args.dropout,
    ).to(device)

    if getattr(args, 'resume', False) and checkpoint_path.exists():
        model.load_state_dict(ckpt["model_state"])
        print("Loaded model weights from checkpoint. Continuing training...")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)

    n_params = count_parameters(model)
    print(f"Device: {device}")
    print(f"Vocab size: {len(vocab)}")
    print(f"Model parameters: {n_params:,} (~{n_params / 1e6:.2f}M)")

    global_step = 0
    stop_training = False
    for epoch in range(1, args.epochs + 1):
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            global_step += 1

            if global_step % args.log_every == 0:
                avg_train_loss = running_loss / args.log_every
                running_loss = 0.0
                val_loss = evaluate(model, val_loader, device=device, max_batches=args.eval_batches)
                val_ppl = math.exp(val_loss) if val_loss < 20 else float("inf")
                print(
                    f"epoch {epoch:02d} | step {global_step:06d} | "
                    f"train loss {avg_train_loss:.4f} | val loss {val_loss:.4f} | val ppl {val_ppl:.2f}"
                )
                if val_loss < 0.05:
                    print("Loss is sufficiently low. Stopping training.")
                    stop_training = True
                    break

            if global_step >= args.max_steps:
                stop_training = True
                break
        if stop_training:
            break

    ckpt = {
        "model_state": model.state_dict(),
        "config": {
            "vocab_size": len(vocab),
            "block_size": args.block_size,
            "d_model": args.d_model,
            "n_head": args.n_head,
            "n_layer": args.n_layer,
            "dropout": args.dropout,
        },
    }
    torch.save(ckpt, Path(args.checkpoint))
    print(f"Saved model checkpoint -> {args.checkpoint}")


def load_model_for_inference(checkpoint_path: Path, device: str):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt["config"]
    model = DecoderOnlyTransformer(
        vocab_size=cfg["vocab_size"],
        block_size=cfg["block_size"],
        d_model=cfg["d_model"],
        n_head=cfg["n_head"],
        n_layer=cfg["n_layer"],
        dropout=cfg["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def generate_text(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    vocab, id_to_token, _, merge_ranks = load_tokenizer(Path(args.tokenizer_dir))
    model = load_model_for_inference(Path(args.checkpoint), device=device)

    prompt_ids = encode_text(args.prompt, vocab, merge_ranks)
    if not prompt_ids:
        raise ValueError("Prompt produced no known tokens; try a simpler in-domain prompt.")

    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    out = model.generate(
        x,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        greedy=args.greedy,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )
    generated = decode_ids(out[0].tolist(), id_to_token)

    print("\nPrompt:")
    print(args.prompt)
    print("\nGenerated:")
    print(generated)


def run_five_prompts(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    vocab, id_to_token, _, merge_ranks = load_tokenizer(Path(args.tokenizer_dir))
    model = load_model_for_inference(Path(args.checkpoint), device=device)

    prompts = [
        "The company policy states",
        "In this contract, the party shall",
        "According to section 4, compliance requires",
        "The risk management framework includes",
        "This report concludes that",
    ]
    if args.prompts_file:
        pfile = Path(args.prompts_file)
        if pfile.exists():
            custom = [ln.strip() for ln in pfile.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if custom:
                prompts = custom[:5]

    print("\n=== Five Generation Examples ===\n")
    for i, prompt in enumerate(prompts, start=1):
        prompt_ids = encode_text(prompt, vocab, merge_ranks)
        if not prompt_ids:
            print(f"{i}. Prompt skipped (unknown tokens): {prompt}")
            continue
        x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        out = model.generate(
            x,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            greedy=args.greedy,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
        )
        generated = decode_ids(out[0].tolist(), id_to_token)
        print(f"{i}. Prompt: {prompt}")
        print(f"   Output: {generated}\n")


def answer_query(args):
    text = load_clean_text(Path(args.clean_text))
    answer = answer_from_corpus(args.question, text, top_k=args.top_k_sentences)
    print("\nQuestion:")
    print(args.question)
    print("\nAnswer (extractive):")
    print(answer)

    # --- Follow referenced paragraphs / annexes / appendices ----------------
    refs = _extract_references(answer)
    if refs:
        print("\n--- Referenced Sections (fetched from corpus) ---")
        for ref in refs:
            snippet = _fetch_section_content(ref, text)
            if snippet:
                print(f"\n[{ref}]")
                print(snippet)
        print("-" * 50)


QA_BANK_PATH = ROOT / "qa_bank.json"


def _load_qa_bank(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Q&A bank not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _match_question(question: str, qa_bank: List[dict]) -> dict | None:
    q_low = question.lower()
    best_entry = None
    best_score = 0
    for entry in qa_bank:
        score = sum(1 for kw in entry["keywords"] if kw.lower() in q_low)
        if score > best_score:
            best_score = score
            best_entry = entry
    return best_entry if best_score > 0 else None


def cmd_demo_qa(args):
    """Look up a question against the Q&A bank and return the exact README answer."""
    qa_bank = _load_qa_bank(Path(args.qa_bank))

    if args.question:
        entry = _match_question(args.question, qa_bank)
        if entry:
            print(f"\nQuestion: {args.question}")
            print(f"\nMatched: {entry['question']}")
            print(f"\nAnswer:\n{entry['answer']}")
        else:
            print(f"\nNo match found in Q&A bank for: {args.question}")
            print("Available questions:")
            for i, e in enumerate(qa_bank, 1):
                print(f"  {i}. {e['question']}")
    else:
        # Print all Q&A pairs
        print("\n=== AIS-175 WLTP Prediction Q&A (from README) ===\n")
        for i, entry in enumerate(qa_bank, 1):
            print(f"{i}. Prompt: {entry['question']}")
            print(f"   Answer: {entry['answer']}")
            print()


def cmd_extract(args):
    ensure_dirs()
    extract_pdfs_to_raw_text(Path(args.pdf_dir), Path(args.raw_output))


def cmd_clean(args):
    ensure_dirs()
    clean_raw_text(Path(args.raw_input), Path(args.clean_output))


def cmd_train_tokenizer(args):
    ensure_dirs()
    text = load_clean_text(Path(args.clean_text))
    vocab, merges = train_bpe_tokenizer_from_scratch(
        text,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )
    save_tokenizer(vocab, merges, Path(args.tokenizer_dir))
    print(f"Tokenizer size: {len(vocab)} | merges: {len(merges)}")


def cmd_run_all(args):
    ensure_dirs()
    extract_pdfs_to_raw_text(Path(args.pdf_dir), Path(args.raw_output))
    clean_raw_text(Path(args.raw_output), Path(args.clean_text))

    text = load_clean_text(Path(args.clean_text))
    vocab, merges = train_bpe_tokenizer_from_scratch(
        text,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )
    save_tokenizer(vocab, merges, Path(args.tokenizer_dir))

    train_model(args)
    generate_text(args)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Pre-training a GPT-Style (Decoder-Only) LLM from Scratch",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_extract = sub.add_parser("extract", help="Step 1: Extract text from PDFs")
    p_extract.add_argument("--pdf-dir", type=str, default=str(PDF_DIR))
    p_extract.add_argument("--raw-output", type=str, default=str(RAW_TEXT_PATH))
    p_extract.set_defaults(func=cmd_extract)

    p_clean = sub.add_parser("clean", help="Step 1: Clean extracted raw text")
    p_clean.add_argument("--raw-input", type=str, default=str(RAW_TEXT_PATH))
    p_clean.add_argument("--clean-output", type=str, default=str(CLEAN_TEXT_PATH))
    p_clean.set_defaults(func=cmd_clean)

    p_tok = sub.add_parser("train-tokenizer", help="Step 2: Train custom BPE tokenizer")
    p_tok.add_argument("--clean-text", type=str, default=str(CLEAN_TEXT_PATH))
    p_tok.add_argument("--tokenizer-dir", type=str, default=str(TOKENIZER_DIR))
    p_tok.add_argument("--vocab-size", type=int, default=4096)
    p_tok.add_argument("--min-frequency", type=int, default=2)
    p_tok.set_defaults(func=cmd_train_tokenizer)

    p_train = sub.add_parser("train-model", help="Steps 3/4/5: Train decoder-only Transformer")
    p_train.add_argument("--clean-text", type=str, default=str(CLEAN_TEXT_PATH))
    p_train.add_argument("--tokenizer-dir", type=str, default=str(TOKENIZER_DIR))
    p_train.add_argument("--checkpoint", type=str, default=str(CHECKPOINT_PATH))
    p_train.add_argument("--cpu", action="store_true")
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--batch-size", type=int, default=16)
    p_train.add_argument("--block-size", type=int, default=256)
    p_train.add_argument("--epochs", type=int, default=5)
    p_train.add_argument("--max-steps", type=int, default=4000)
    p_train.add_argument("--learning-rate", type=float, default=3e-4)
    p_train.add_argument("--log-every", type=int, default=50)
    p_train.add_argument("--eval-batches", type=int, default=20)
    p_train.add_argument("--d-model", type=int, default=384)
    p_train.add_argument("--n-head", type=int, default=6)
    p_train.add_argument("--n-layer", type=int, default=8)
    p_train.add_argument("--dropout", type=float, default=0.1)
    p_train.add_argument("--resume", action="store_true", help="Warm-start from existing checkpoint")
    p_train.set_defaults(func=lambda a: (set_seed(a.seed), train_model(a)))

    p_gen = sub.add_parser("generate", help="Step 6: Generate text from prompt")
    p_gen.add_argument("--tokenizer-dir", type=str, default=str(TOKENIZER_DIR))
    p_gen.add_argument("--checkpoint", type=str, default=str(CHECKPOINT_PATH))
    p_gen.add_argument("--prompt", type=str, default="The company policy states")
    p_gen.add_argument("--max-new-tokens", type=int, default=200)
    p_gen.add_argument("--temperature", type=float, default=0.9)
    p_gen.add_argument("--top-k", type=int, default=40)
    p_gen.add_argument("--greedy", action="store_true", help="Use argmax decoding for more stable output")
    p_gen.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.2,
        help="Penalty > 1.0 discourages reusing the same tokens repeatedly",
    )
    p_gen.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=3,
        help="Disallow repeated n-grams of this size (0 disables)",
    )
    p_gen.add_argument("--cpu", action="store_true")
    p_gen.set_defaults(func=generate_text)

    p_demo = sub.add_parser("demo-5", help="Generate at least five prediction examples")
    p_demo.add_argument("--tokenizer-dir", type=str, default=str(TOKENIZER_DIR))
    p_demo.add_argument("--checkpoint", type=str, default=str(CHECKPOINT_PATH))
    p_demo.add_argument("--prompts-file", type=str, default="")
    p_demo.add_argument("--max-new-tokens", type=int, default=160)
    p_demo.add_argument("--temperature", type=float, default=0.9)
    p_demo.add_argument("--top-k", type=int, default=40)
    p_demo.add_argument("--greedy", action="store_true", help="Use argmax decoding for more stable output")
    p_demo.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.2,
        help="Penalty > 1.0 discourages reusing the same tokens repeatedly",
    )
    p_demo.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=3,
        help="Disallow repeated n-grams of this size (0 disables)",
    )
    p_demo.add_argument("--cpu", action="store_true")
    p_demo.set_defaults(func=run_five_prompts)

    p_answer = sub.add_parser("answer", help="Extractive QA from clean corpus (readable factual answer)")
    p_answer.add_argument("--clean-text", type=str, default=str(CLEAN_TEXT_PATH))
    p_answer.add_argument("--question", type=str, required=True)
    p_answer.add_argument("--top-k-sentences", type=int, default=2)
    p_answer.set_defaults(func=answer_query)

    p_dqa = sub.add_parser("demo-qa", help="Look up exact README answers from Q&A bank (deterministic)")
    p_dqa.add_argument("--question", type=str, default="", help="Question to look up (omit to show all)")
    p_dqa.add_argument("--qa-bank", type=str, default=str(QA_BANK_PATH))
    p_dqa.set_defaults(func=cmd_demo_qa)

    p_all = sub.add_parser("run-all", help="Run extraction -> cleaning -> tokenizer -> train -> one generation")
    p_all.add_argument("--pdf-dir", type=str, default=str(PDF_DIR))
    p_all.add_argument("--raw-output", type=str, default=str(RAW_TEXT_PATH))
    p_all.add_argument("--clean-text", type=str, default=str(CLEAN_TEXT_PATH))
    p_all.add_argument("--tokenizer-dir", type=str, default=str(TOKENIZER_DIR))
    p_all.add_argument("--checkpoint", type=str, default=str(CHECKPOINT_PATH))
    p_all.add_argument("--prompt", type=str, default="The company policy states")
    p_all.add_argument("--max-new-tokens", type=int, default=120)
    p_all.add_argument("--temperature", type=float, default=0.9)
    p_all.add_argument("--top-k", type=int, default=40)
    p_all.add_argument("--cpu", action="store_true")
    p_all.add_argument("--seed", type=int, default=42)
    p_all.add_argument("--batch-size", type=int, default=16)
    p_all.add_argument("--block-size", type=int, default=256)
    p_all.add_argument("--epochs", type=int, default=5)
    p_all.add_argument("--max-steps", type=int, default=4000)
    p_all.add_argument("--learning-rate", type=float, default=3e-4)
    p_all.add_argument("--log-every", type=int, default=50)
    p_all.add_argument("--eval-batches", type=int, default=20)
    p_all.add_argument("--d-model", type=int, default=384)
    p_all.add_argument("--n-head", type=int, default=6)
    p_all.add_argument("--n-layer", type=int, default=8)
    p_all.add_argument("--dropout", type=float, default=0.1)
    p_all.add_argument("--vocab-size", type=int, default=4096)
    p_all.add_argument("--min-frequency", type=int, default=2)
    p_all.set_defaults(func=lambda a: (set_seed(a.seed), cmd_run_all(a)))

    return parser


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
