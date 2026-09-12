# tei-line-ends

**OCR correction at the line break in TEI** — the two error classes that exist
only because a line ended there, and a method for correcting them that needs no
dictionary and never lets a machine write to the transcription unsupervised.

Extracted from the [Wiener Salonblatt](https://anno.onb.ac.at/) corpus project:
787 issues of a Viennese society newspaper, 1919–1938, transcribed in
Transkribus. The tools are corpus-independent; the measurements quoted
throughout come from that corpus.

---

## The problem

A line break damages the two characters next to it, and nothing else.

**Hyphenation.** A line ends in `-`. Is that a hyphen the word actually has
(`Erz-herzog`), or only the mark that the line ran out (`Vor-trag` → `Vortrag`)?
Get it wrong and every downstream step — search, NER, the edition — inherits a
word that does not exist.

**The line-final character.** The last letterform on a line is clipped by the
margin or sits next to the break mark, and the OCR misreads it far more often
than a character anywhere else. The signature case is a final `n` read as `r`:

```
Kronen -> Kroner        Wien -> Wier        Grafen -> Grafer
Damen  -> Damer         Sohn -> Sohr        Namen  -> Namer
```

## The method

Four ideas, and they are the transferable part:

1. **The corpus is its own dictionary.** The reference lexicon is built from a
   position the error cannot reach — the *interior* tokens of a line, neither
   first nor last, which no line break can have damaged. A line-final spelling
   that never occurs inside a line, while a one-character variant occurs 3,000
   times, is a misreading. No external word list, no language model, no
   assumption about spelling: historical orthography is whatever this corpus
   does, and the corpus is what is counted.

2. **Statistics propose; a local LLM judges; a human outranks both.** Every
   machine verdict lands in a plain CSV decision store with a `source` column
   ranked `manual > llm > rule`. Re-running a detector cannot undo a decision a
   person made. **No detector ever writes to the XML** — exactly one script
   applies the store, and it is not the one that found anything.

3. **Position is half the warrant.** `Wier` is a misreading at a line end and a
   perfectly good token in the middle of a line. Corrections are therefore
   keyed by *where* they apply — at any line end, before one specific word, or
   at one specific line — never as a global search-and-replace.

4. **The base rate decides the architecture.** When 99.5% of line ends are
   already correct, a classifier that is right 89% of the time breaks eighteen
   words for every one it fixes. Measured here: plain argmax over the context
   model **breaks 4,705 correct tokens to fix 258**. That is why the
   per-occurrence stage measures and gates rather than corrects — a negative
   result that shapes the whole design.

## The two chains

Both run over TEI exported from Transkribus, and both are documented at length
in the scripts' own docstrings.

### Line-break hyphens

```
validate_xml_hyphens.py     rule verdicts per word pair, from the interior lexicon
      >>> resolve_break_context.py   contested pairs, one occurrence at a time
      >>> resolve_hyphens_llm.py     a local model reads the sentence
            >>> data/csv/hyphen_decisions.csv  +  hyphen_occurrences.csv
                  >>> correct_xml_hyphens.py   applies them
                  >>> mark_xml_word_breaks.py  records the verdict as @wsb:break
```

`kaiser|in` is "Kaiserin Zita" in one column and "der Kaiser in Wien" in the
next — so a pair store is not enough, and `resolve_break_context.py` moves that
decision down to the single occurrence.

### Line-final characters

```
validate_line_end_chars.py  rule verdicts per token: OK / MISREAD / REVIEW
      >>> resolve_line_end_llm.py    a local model confirms or overturns
extract_ocr_corrections.py  a second detector, from the hyphen run's notes
resolve_line_end_context.py the class statistics cannot see (der/den/dem)
            >>> data/csv/ocr_corrections.csv
                  >>> correct_xml_ocr.py       applies them
```

Full walkthrough with flow charts: **[docs/line-end-correction.md](docs/line-end-correction.md)**.

### The same two parts, wired opposite ways

The most interesting result in the repo, and the reason the two chains are here
together:

| | hyphens | line-final characters |
|---|---|---|
| population | 2,842 contested breaks | ~258,000 ambiguous line ends |
| who decides | **the LLM decides every case**; the context score is evidence in the prompt | **the score decides what gets asked**; the LLM decides the answer |

Same components, opposite roles, and the reason is only the size of the
population. Affordability is a design input, not an implementation detail.

## Install and run

Python ≥ 3.11, three dependencies.

```bash
git clone https://github.com/USER/tei-line-ends
cd tei-line-ends
uv sync                      # or: pip install lxml requests tqdm

python config.py             # print the resolved paths

# point it at your own TEI
python lineends/validate_xml_hyphens.py     --xml-folder /path/to/tei
python lineends/validate_line_end_chars.py  --xml-folder /path/to/tei
python lineends/resolve_line_end_context.py --xml-folder /path/to/tei --score
```

Every script takes `--xml-folder` and `--max-files`; nothing writes to your TEI
until you run `correct_xml_ocr.py` or `correct_xml_hyphens.py`, and both of
those take `--dry-run`. Reports land in `output/`, dated.

The review stages need a local model — [Ollama](https://ollama.com) or LM
Studio, no API key, nothing leaves the machine. Pick it per task in
`config.toml`.

## Adapting it to your corpus

The tools assume TEI with `<p>`/`<lg>` containing `<l>` elements, a
`<facsimile>` with zones, and region `@subtype`s — i.e. a Transkribus export.
Everything corpus-specific is in `config.toml`:

| setting | what to change it to |
|---|---|
| `[corpus] project_namespace` | a domain you control; it carries the word-break attribute |
| `[corpus] marker_prefix` | the `<application ident='...'>` prefix recording that a step ran |
| `[llm.*]` | your local model per task |
| `[transkribus]` | your collection — see [docs/transkribus-links.md](docs/transkribus-links.md) |

Two things are genuinely German and are worth knowing before you start: the
alphabet `validate_line_end_chars.py` substitutes over includes `äöüß`, and the
prompts in `data/prompts/` are written in German about German orthography. The
lexicon method itself assumes nothing about language.

Region subtypes are a CLI flag: `--skip-subtypes ad,ad-frame` keeps
advertisements out of the lexicon, because their layout is not prose. Yours will
be named differently.

## Status

| | |
|---|---|
| hyphen chain | complete, run over 787 issues |
| line-final characters, rule + LLM | complete, run over 787 issues |
| `resolve_line_end_context.py` | **scores and measures; does not correct** |

The per-occurrence line-end stage is deliberately unfinished: the store and the
resolver are not built, because the hand-checked sample has to say what the gate
is worth first. What it measured so far — including the three failure
populations no margin can separate, and a window-size experiment that changed
**0 margins and 0 proposals** — is in
[docs/line-end-context-correction.md](docs/line-end-context-correction.md).

No sample corpus ships with the repo: the transcriptions are not ours to
republish. Point `--xml-folder` at your own export.

## Documentation

| | |
|---|---|
| [docs/line-end-correction.md](docs/line-end-correction.md) | the line-final character chain, end to end, with flow charts |
| [docs/line-end-context-correction.md](docs/line-end-context-correction.md) | the per-occurrence stage: findings, measurements, why it does not correct |
| [docs/hyphen-occurrence-disambiguation.md](docs/hyphen-occurrence-disambiguation.md) | the same move for hyphens, which is where it worked first |
| [docs/ocr-word-correction.md](docs/ocr-word-correction.md) | truncated tokens — the class a substitution cannot repair |
| [docs/transkribus-links.md](docs/transkribus-links.md) | wiring findings back to the scan |

The scripts' docstrings are the reference documentation and are written to be
read. Each one says what it refuses to do and why.

## Citation

See [CITATION.cff](CITATION.cff). MIT licensed.
