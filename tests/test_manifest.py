"""Round-trip test for the ManifestRow JSONL schema (pipeline/manifest.py)."""

from moa_cas.pipeline.manifest import ManifestRow, write_jsonl, read_jsonl


def test_manifest_roundtrip(tmp_path):
    rows = [
        ManifestRow(
            utt_id="u1", audio_filepath="a.wav", text="धार्मिक स्थान है",
            duration=2.5, dataset="mucs", lang_pair="hi-en", split="test",
            n_words=3, n_words_l1=3, n_words_en=0, n_cs_points=0, is_code_mixed=False,
        ),
        ManifestRow(
            utt_id="u2", audio_filepath="b.wav", text="यह हमारा main function है",
            duration=3.1, dataset="mucs", lang_pair="hi-en", split="test",
            n_words=5, n_words_l1=3, n_words_en=2, n_cs_points=2, is_code_mixed=True,
            speaker_id="spk01",
        ),
    ]

    out_path = tmp_path / "manifest.jsonl"
    write_jsonl(rows, out_path)

    loaded = read_jsonl(out_path)
    assert len(loaded) == 2
    assert loaded[0]["utt_id"] == "u1"
    assert loaded[0]["text"] == "धार्मिक स्थान है"
    assert loaded[0]["is_code_mixed"] is False
    assert loaded[1]["is_code_mixed"] is True
    assert loaded[1]["speaker_id"] == "spk01"
