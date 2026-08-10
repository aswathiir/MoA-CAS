"""
moa_cas.preprocessing
------------------------
No eager submodule imports: `text` is pure Python (no deps beyond stdlib),
while `audio` needs soundfile/scipy for Kaldi segment extraction. Importing
`moa_cas.preprocessing.text` shouldn't require an audio stack to be
installed, so callers import each submodule directly:

    from moa_cas.preprocessing.text import tag_word, tag_utterance
    from moa_cas.preprocessing.audio import extract_kaldi_split
"""
