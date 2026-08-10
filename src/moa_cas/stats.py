"""
moa_cas/stats.py
-------------------
Corpus statistics report, mirroring Table 1 of Biswas et al. 2025
("Adapting Whisper for low-resource Hindi-English Code-Mix speech..."):
duration, per-language word counts, and code-switch bigram counts.
"""


def summarize(rows: list, label: str = "") -> dict:
    duration_h = sum(r.duration for r in rows) / 3600 if rows else 0.0
    n_words_l1 = sum(r.n_words_l1 for r in rows)
    n_words_en = sum(r.n_words_en for r in rows)
    n_cs = sum(r.n_cs_points for r in rows)
    n_code_mixed = sum(1 for r in rows if r.is_code_mixed)

    summary = {
        "utterances": len(rows),
        "duration_hours": round(duration_h, 2),
        "words_l1": n_words_l1,
        "words_en": n_words_en,
        "total_words": n_words_l1 + n_words_en,
        "cs_points": n_cs,
        "code_mixed_utterances": n_code_mixed,
        "code_mixed_pct": round(100 * n_code_mixed / len(rows), 1) if rows else 0.0,
    }

    print(f"\n{'─' * 52}")
    print(f"  {label or 'Corpus'} — {len(rows)} utterances")
    print(f"{'─' * 52}")
    print(f"  Duration        : {summary['duration_hours']:.2f} h")
    print(f"  Words (L1)      : {summary['words_l1']:,}")
    print(f"  Words (English) : {summary['words_en']:,}")
    print(f"  CS bigrams      : {summary['cs_points']:,}")
    print(f"  Code-mixed utts : {summary['code_mixed_utterances']:,} ({summary['code_mixed_pct']}%)")
    print(f"{'─' * 52}\n")

    return summary
