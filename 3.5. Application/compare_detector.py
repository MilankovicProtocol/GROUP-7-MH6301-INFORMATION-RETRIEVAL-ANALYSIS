"""
compare_detector.py
Detect sentences in Yelp reviews that contain *comparisons* between the
reviewed business and other businesses the reviewer has visited.

Per the assignment specification (MH6301GroupProject.pdf, §3.5):

    "Define and develop a simple NLP/IR application based on the dataset.
     An example application is to detect the sentences containing comparison
     in reviews. In a sentence containing 'comparison' the reviewer compares
     the current business being reviewed with other businesses that he/she
     has visited."

The detector combines three orthogonal signal layers:

  1. Comparative POS tags. NLTK's averaged_perceptron_tagger labels English
     comparative forms as JJR (e.g. "better", "cheaper") and RBR (e.g.
     "more", "less"). Both are strong comparison signals.

  2. A curated lexicon of multi-word cues. Some comparisons are phrased
     without an explicit comparative form, e.g. "compared to", "as good
     as", "rather than", "instead of", "unlike". These are matched as
     bigrams or trigrams over the lower-cased token stream.

  3. The "than" construction. A sentence with "than" preceded (anywhere in
     the same sentence) by a JJR / RBR / "more" / "less" is a near-certain
     comparison. Plain "than" without a comparative is downweighted because
     it is sometimes used in temporal sense ("rather than wait"); see the
     lexical fallback in step 2.

Each detected sentence is scored by the number of independent cue hits.
A simple post-hoc target extractor scans forward from each cue to capture
the next noun phrase, which is reported as the (probable) comparison
target.

CLI:
    python compare_detector.py --top 20
    python compare_detector.py --top 50 --output ../sample_output/sample_top50.txt
    python compare_detector.py --business-id <id> --top 20
    python compare_detector.py --scan-cap 50000 --top 20      # fast mode
"""

from __future__ import annotations

import argparse
import heapq
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import nltk

# Module-relative import; works both as `python compare_detector.py` and as
# `python -m SourceCode.compare_detector` from one level up.
try:
    from data_loader import iter_philly_reviews, philly_business_ids
except ImportError:  # pragma: no cover - only triggers in unusual invocation
    from .data_loader import iter_philly_reviews, philly_business_ids  # type: ignore


# ---------- NLTK data bootstrap ----------

def ensure_nltk_data() -> None:
    """Download the small NLTK data packages we depend on if missing.

    Tolerant of a corrupted local cache: any failure from `nltk.data.find`
    (LookupError, BadZipFile, OSError, etc.) triggers a re-download.
    """
    required = [
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ]
    for resource_path, pkg in required:
        ok = False
        try:
            nltk.data.find(resource_path)
            ok = True
        except Exception:
            ok = False
        if not ok:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass


# ---------- comparison cue lexicon ----------

# Multi-word phrases. Each is a tuple of lowercased tokens. We test the
# sentence's lowercased token list for a contiguous match.
MULTIWORD_CUES: tuple[tuple[str, ...], ...] = (
    ("compared", "to"),
    ("compared", "with"),
    ("in", "comparison", "to"),
    ("in", "comparison", "with"),
    ("as", "good", "as"),
    ("as", "bad", "as"),
    ("not", "as", "good", "as"),
    ("not", "as", "bad", "as"),
    ("as", "much", "as"),
    ("rather", "than"),
    ("instead", "of"),
    ("similar", "to"),
    ("different", "from"),
    ("better", "than"),
    ("worse", "than"),
    ("nothing", "like"),
)

# Single-token cues that on their own are weak but valid signals.
SINGLE_TOKEN_CUES: frozenset[str] = frozenset({
    "unlike",
    "versus",
    "vs",
    "vs.",
    "outperforms",
    "outshines",
    "beats",
})

# Verbs whose lemma signals comparison. We match on prefix to catch
# inflections without setting up a stemmer just for these.
COMPARE_VERB_PREFIXES: tuple[str, ...] = (
    "prefer",
    "compar",
    "outperform",
    "outshin",
)

# POS tags for comparative forms (JJR = comparative adj, RBR = comparative adv).
COMPARATIVE_TAGS: frozenset[str] = frozenset({"JJR", "RBR"})

# A "than" inside a sentence with a JJR / RBR / "more" / "less" anywhere is a
# strong comparison. By itself, "than" can be ambiguous (e.g. "rather than").
THAN_TOKENS: frozenset[str] = frozenset({"than"})
COMPARATIVE_LEMMAS: frozenset[str] = frozenset({"more", "less"})

# Noun-phrase chunk pattern (very small grammar) for post-hoc target extraction.
NP_TAGS: frozenset[str] = frozenset({
    "NN", "NNS", "NNP", "NNPS",  # nouns
    "DT", "PRP$",                # determiners / possessives
    "JJ",                        # plain adjective
    "CD",                        # cardinal numbers ("two")
})


# ---------- detection ----------

@dataclass
class CueHit:
    kind: str          # one of: pos_jjr, pos_rbr, multiword, single, verb, than_with_comparative
    span: tuple[int, int]   # (start_token_idx, end_token_idx_exclusive)
    surface: str       # the matched surface text (joined tokens)


@dataclass
class Match:
    sentence: str
    tokens: list[str]
    pos: list[tuple[str, str]]
    hits: list[CueHit] = field(default_factory=list)
    target_np: str | None = None

    @property
    def score(self) -> int:
        # Each distinct cue contributes 1 to the score; duplicate categories
        # are downweighted so a sentence repeating the same cue twelve times
        # does not crowd out genuinely richer comparisons.
        kinds = {h.kind for h in self.hits}
        return len(kinds) + min(len(self.hits) - len(kinds), 2)


def _multiword_hits(tokens_lower: list[str]) -> list[CueHit]:
    hits: list[CueHit] = []
    for phrase in MULTIWORD_CUES:
        L = len(phrase)
        for i in range(len(tokens_lower) - L + 1):
            if tuple(tokens_lower[i : i + L]) == phrase:
                hits.append(CueHit(
                    kind="multiword",
                    span=(i, i + L),
                    surface=" ".join(phrase),
                ))
    return hits


def _pos_hits(pos: list[tuple[str, str]]) -> list[CueHit]:
    hits: list[CueHit] = []
    for i, (tok, tag) in enumerate(pos):
        if tag == "JJR":
            hits.append(CueHit("pos_jjr", (i, i + 1), tok))
        elif tag == "RBR":
            hits.append(CueHit("pos_rbr", (i, i + 1), tok))
    return hits


def _single_token_hits(tokens_lower: list[str]) -> list[CueHit]:
    hits: list[CueHit] = []
    for i, t in enumerate(tokens_lower):
        if t in SINGLE_TOKEN_CUES:
            hits.append(CueHit("single", (i, i + 1), t))
        elif any(t.startswith(p) for p in COMPARE_VERB_PREFIXES):
            hits.append(CueHit("verb", (i, i + 1), t))
    return hits


def _than_construction_hit(
    tokens_lower: list[str], pos: list[tuple[str, str]]
) -> CueHit | None:
    """Fire if 'than' appears in a sentence that also contains a comparative
    form (JJR/RBR or the words 'more' / 'less')."""
    if not any(t in THAN_TOKENS for t in tokens_lower):
        return None
    has_comparative = any(tag in COMPARATIVE_TAGS for _, tag in pos)
    if not has_comparative:
        has_comparative = any(t in COMPARATIVE_LEMMAS for t in tokens_lower)
    if not has_comparative:
        return None
    idx = tokens_lower.index("than")
    return CueHit("than_with_comparative", (idx, idx + 1), "than")


def _target_np(pos: list[tuple[str, str]], cue_end_idx: int, max_len: int = 6) -> str | None:
    """Greedy forward scan from index `cue_end_idx`, collecting tokens whose
    POS is in NP_TAGS until we hit something else or `max_len` tokens."""
    out: list[str] = []
    for i in range(cue_end_idx, min(cue_end_idx + max_len, len(pos))):
        tok, tag = pos[i]
        if tag in NP_TAGS:
            out.append(tok)
        elif not out:
            # Skip leading "an", "the" if not picked up. Already in NP_TAGS,
            # so this branch handles function words like "to" between cue
            # and NP head, e.g. "compared to the other place".
            continue
        else:
            break
    if not out:
        return None
    return " ".join(out).strip()


def detect_in_sentence(sentence: str) -> Match | None:
    """Run the full detector on a single sentence.

    Returns a `Match` if at least one cue fires, else `None`.
    """
    tokens = nltk.word_tokenize(sentence)
    if not tokens:
        return None
    tokens_lower = [t.lower() for t in tokens]
    pos = nltk.pos_tag(tokens)

    hits: list[CueHit] = []
    hits.extend(_pos_hits(pos))
    hits.extend(_multiword_hits(tokens_lower))
    hits.extend(_single_token_hits(tokens_lower))
    than_hit = _than_construction_hit(tokens_lower, pos)
    if than_hit is not None:
        hits.append(than_hit)

    if not hits:
        return None

    # Greedy target NP from the first cue end.
    first_cue = min(hits, key=lambda h: h.span[0])
    target = _target_np(pos, first_cue.span[1])

    return Match(
        sentence=sentence.strip(),
        tokens=tokens,
        pos=pos,
        hits=hits,
        target_np=target,
    )


# ---------- streaming top-N over the review corpus ----------

@dataclass(order=True)
class _HeapEntry:
    score: int
    # We need a stable secondary key for ties; review id text length acts as
    # a cheap stand-in. The actual payload is excluded from comparison.
    sort_key: int
    business_id: str = field(compare=False)
    stars: float = field(compare=False)
    sentence: str = field(compare=False)
    target_np: str | None = field(compare=False)
    cue_kinds: tuple[str, ...] = field(compare=False)


def _split_sentences(text: str) -> list[str]:
    """Robustly split a review into sentences."""
    if not text:
        return []
    # nltk.sent_tokenize is reasonable; we strip and drop empties.
    return [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]


def scan_corpus(
    top_n: int = 20,
    business_id_filter: str | None = None,
    scan_cap: int | None = None,
    review_cap: int | None = None,
    verbose: bool = True,
) -> tuple[list[_HeapEntry], dict[str, int]]:
    """Stream the Philadelphia review subset and return the top-N comparison
    sentences plus a small statistics dict.
    """
    heap: list[_HeapEntry] = []
    stats = {
        "reviews_scanned": 0,
        "sentences_scanned": 0,
        "sentences_with_comparison": 0,
        "hits_total": 0,
    }

    if business_id_filter is not None:
        business_ids = {business_id_filter}
    else:
        business_ids = philly_business_ids()

    for review in iter_philly_reviews(business_ids=business_ids, limit=review_cap):
        stats["reviews_scanned"] += 1
        for sent in _split_sentences(review.get("text", "")):
            if scan_cap is not None and stats["sentences_scanned"] >= scan_cap:
                break
            stats["sentences_scanned"] += 1
            m = detect_in_sentence(sent)
            if m is None:
                continue
            stats["sentences_with_comparison"] += 1
            stats["hits_total"] += len(m.hits)

            entry = _HeapEntry(
                score=m.score,
                sort_key=len(sent),
                business_id=review.get("business_id", ""),
                stars=float(review.get("stars", 0)),
                sentence=m.sentence,
                target_np=m.target_np,
                cue_kinds=tuple(sorted({h.kind for h in m.hits})),
            )
            if len(heap) < top_n:
                heapq.heappush(heap, entry)
            else:
                heapq.heappushpop(heap, entry)

        if verbose and stats["reviews_scanned"] % 50_000 == 0:
            print(f"  scanned {stats['reviews_scanned']:,} reviews, "
                  f"{stats['sentences_with_comparison']:,} comparison sents found",
                  file=sys.stderr)
        if scan_cap is not None and stats["sentences_scanned"] >= scan_cap:
            break

    heap.sort(key=lambda e: (-e.score, e.sort_key))
    return heap, stats


# ---------- output ----------

def format_entry(rank: int, entry: _HeapEntry) -> str:
    target = f"target=\"{entry.target_np}\"" if entry.target_np else "target=<none>"
    cues = ",".join(entry.cue_kinds)
    return (
        f"[{rank}] score={entry.score} stars={entry.stars:g} "
        f"business_id={entry.business_id} cues={cues} {target}\n"
        f"    \"{entry.sentence}\"\n"
    )


def format_run(entries: list[_HeapEntry], stats: dict[str, int]) -> str:
    lines: list[str] = []
    lines.append(
        f"# §3.5 Application — Comparison-Sentence Detector\n"
        f"# Dataset: Yelp Open Dataset, Philadelphia subset.\n"
        f"# Reviews scanned: {stats['reviews_scanned']:,}\n"
        f"# Sentences scanned: {stats['sentences_scanned']:,}\n"
        f"# Sentences detected as comparison: {stats['sentences_with_comparison']:,}\n"
        f"# Total cue hits (any category): {stats['hits_total']:,}\n"
        f"# Returned: top-{len(entries)} by cue-diversity score\n"
        f"\n"
    )
    for i, e in enumerate(entries, 1):
        lines.append(format_entry(i, e))
    return "".join(lines)


# ---------- CLI ----------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--top", type=int, default=20,
                        help="number of top-scoring comparison sentences to return")
    parser.add_argument("--business-id", default=None,
                        help="restrict scan to a single business_id")
    parser.add_argument("--scan-cap", type=int, default=None,
                        help="stop after this many sentences scanned (smoke / dev)")
    parser.add_argument("--review-cap", type=int, default=None,
                        help="stop after this many reviews scanned (smoke / dev)")
    parser.add_argument("--output", default=None,
                        help="write to this file as well as stdout")
    parser.add_argument("--quiet", action="store_true",
                        help="suppress progress messages on stderr")
    args = parser.parse_args(argv)

    ensure_nltk_data()

    entries, stats = scan_corpus(
        top_n=args.top,
        business_id_filter=args.business_id,
        scan_cap=args.scan_cap,
        review_cap=args.review_cap,
        verbose=not args.quiet,
    )

    out_text = format_run(entries, stats)
    sys.stdout.write(out_text)

    if args.output is not None:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(out_text, encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
