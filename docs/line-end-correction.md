# The line-end character correction, end to end

Companion to `lineends/README.md` §2 ("The review loops"). Written to be
read aloud: it explains *why* this chain has one more stage than the hyphen
chain, and what that stage actually does.

---

## 0. The one-paragraph version

Transkribus misreads the **last character of a line** far more often than a
character anywhere else — the letterform is clipped by the margin or sits next
to the break hyphen. `Wien` comes out as `Wier`, `Kronen` as `Kroner`. The
correction has three stages: corpus **statistics** propose, a local **LLM**
confirms or overturns, and a separate script **applies** the surviving verdicts
to the XML. A fourth stage — `resolve_line_end_context.py` — exists for the
cases the statistics are structurally blind to, and it works at a different
granularity from everything else: not "this word is always wrong", but "this
word is wrong *on this one line*".

---

## 1. First, the distinction that causes the confusion

There are **two OCR corrections in `lineends`**, and they answer different
questions. They must never be merged.

| | store | claim | applied where |
|---|---|---|---|
| **global** | `data/csv/replacement.csv` | "this string is always wrong, *wherever* it occurs" — `Selte` → `Seite` | every paragraph, corpus-wide |
| **positional** | `data/csv/ocr_corrections.csv` | "this word is wrong *here*, at a line end" — `Wier` → `Wien` | only at a line break |

`Wier` is not wrong in the middle of a line; the letters were never clipped
there. Putting a positional error into `replacement.csv` would corrupt every
innocent occurrence. **The position is half the warrant** — which is the whole
reason this second store exists.

And inside the positional store there are again two resolutions:

    wildcard row      Wier , * , Wien       "this token, at ANY line end"
    pair row          Antra , der , Antrag  "this token, before THAT word"

Everything below is about how those rows get made — and about a third,
finer resolution (one specific line) that is designed but not yet built.

---

## 2. The master flow

```
                      data/xml/  (787 issues, Transkribus TEI)
                          |
      +-------------------+------------------------------------+
      |                   |                                    |
      v                   v                                    v
 [A] STATISTICS      [C] CONTEXT MODEL                   [B] SIDE CHANNEL
 validate_line_      resolve_line_end_context.py         hyphen_decisions.csv
 end_chars.py        (the "ambiguous" class)             (notes from the
      |                   |                               hyphen LLM run)
      |                   |                                    |
      v                   v                                    v
 OK / MISREAD /      margin score per OCCURRENCE          extract_ocr_
 REVIEW per TOKEN    (writes NOTHING)                     corrections.py
      |                   |                                    |
      |                   v                                    |
      |              output/*_line_end_sample.csv              |
      |              >>> a human, phase 1                      |
      |                   :                                    |
      |                   : (phases 4-6 unbuilt)               |
      v                   :                                    v
 >>> [LLM] resolve_line_end_llm.py  <<<-------------------  (pair rows)
      |    prompt: data/prompts/prompt_ocr_correction.txt
      |
      v
 data/csv/ocr_corrections.csv        <-- the positional store
      |    (manual > llm > rule; confidence high/medium/low)
      v
 >>> correct_xml_ocr.py  +  data/csv/replacement.csv
      |
      v
 data/xml/  (corrected in place)  +  output/*_ocr_corrections_applied.csv
```

Read the three entry points as **three detectors feeding one store**, one
gatekeeper (the LLM), and one applier.

---

## 3. Stage A — the statistics (`validate_line_end_chars.py`)

**The trick: a lexicon the error cannot reach.** No dictionary is used. The
reference is built from the corpus itself, from a position a line break cannot
damage: the **interior** tokens of a line, neither first nor last. Same trick as
`validate_xml_hyphens.py`, and literally the same `build_lexicon` call, so both
scripts key their tables identically.

For every line-final token, try every one-character substitution of its final
letter and look both spellings up in that interior lexicon.

```
INPUT   data/xml/*.xml
        |
        +-- interior lexicon (~4M tokens, via validate_xml_hyphens.build_lexicon)
        +-- every line-final token
        v
        for each: observed count  vs  count of each 1-char-substituted variant
        v
VERDICT  OK        the transcribed spelling is a word this corpus uses
         MISREAD   exactly one candidate dominates (>= 20x) and the observed
                   form is essentially absent (<= 1/50 of it)
         REVIEW    two candidates compete, or the observed form is attested
                   after all  ->  only the sentence can decide
        v
OUTPUT  output/<date>_line_end_chars.csv          (always; Transkribus link/row)
        --export-corrections >>> data/csv/ocr_corrections.csv
                                 as wildcard rows, source=rule, confidence=high
        --export-review      >>> output/<date>_linechar_review.jsonl
                                 (+ --review-misread: the confident ones too)
```

Measured on the corpus (880,806 judged line ends):

    OK       877,587  99.63%
    MISREAD    2,350   0.27%   (1,371 distinct tokens)
    REVIEW       869   0.10%   (  318 distinct tokens)

**Why a confident MISREAD is still not applied.** Two failure modes, neither
fixable by moving a threshold:

    dorf -> dort      `-dorf` is a place-name ending; Reiss, Mariae the same.
    Reiss -> Reise    This newspaper is largely surnames and place names, and
                      a rare real name looks exactly like a misread word.

    Scht -> Schw      The token is TRUNCATED, not substituted. No single-
    Geno -> Genf      character substitution can repair it, and proposing one
                      makes the text worse than leaving it alone.

So the confident verdicts are written as `source=rule`, the lowest rank in the
store — a proposal, not a verdict. **This is the design principle of the whole
folder: the detector never writes into `data/xml/`.**

---

## 4. Stage B — the LLM gate (`resolve_line_end_llm.py`)

```
INPUT   output/<date>_linechar_review.jsonl   (REVIEW, and MISREAD if asked)
        data/prompts/prompt_ocr_correction.txt
        via llm_backend.py  ->  ollama / lmstudio, local
        v
ASK     "Which spelling does this sentence want?"  — one question, two jobs:
          REVIEW   the corpus could not settle it. "später" outnumbers "späten"
                   inside a line, yet "am späten Abend" is simply correct German.
          MISREAD  the corpus settled it and the model is CHECKING the answer —
                   which catches exactly the Reiss/Scht failures above.
        v
OUTPUT  data/csv/ocr_corrections.csv,  wildcard rows,  source=llm
```

Three design choices worth a slide:

1. **The model chooses, it never spells.** It is shown the candidate list and
   told to pick — and the transcribed spelling is always offered **first**, so
   "leave it alone" is as easy to choose as any correction. Candidates come from
   one character substitution on an *actually observed* token. This is what
   stops the model quietly modernising historical orthography (`Cassa`, `Theil`,
   `Bureau`), the one failure that would rewrite the source. An answer outside
   the candidate set is recorded as "leave it alone".
2. **`llm` outranks `rule`.** A confirmation leaves the store as it was; a
   rejection overturns the statistics. Nothing can override a `manual` row.
3. **The prompt is a file, not a string literal** — revising the rules should
   show up as a diff of the *text*, not of the code that sends it.

Current state of the store: **1,320 rows, all `source=llm`, all wildcard, 695 of
them "confirmed as-is"**. Every `rule` row the statistics wrote has been passed
through the model and replaced by its verdict. Over half the machine's confident
proposals were talked out of it.

---

## 5. Stage B′ — the side channel (`extract_ocr_corrections.py`)

The one that produces **pair** rows rather than wildcard rows, and the only
stage whose input is another stage's exhaust.

```
INPUT   data/csv/hyphen_decisions.csv  ->  the `note` column
        While resolve_hyphens_llm.py was deciding how a broken word should be
        written, the model kept remarking on something it was never asked:
           "...bei dem die OCR-Erkennung Buchstaben (u, O) ausgelassen hat."
        Free German prose, far too varied for a regex.
        v
ASK     Not "is this word right?" but "what does this NOTE claim about these
        two words?" — the model reads the note, not the newspaper, and may not
        invent a correction the note does not make.
        v
OUTPUT  data/csv/ocr_corrections.csv,  PAIR rows (left+right), source=llm,
        each carrying its own confidence
```

109 of 1,458 settled pairs carry such a note — a minutes-long run. The yield
grows every time the hyphen review queue is drained further. The point for the
talk: **the notes were dead text, and this makes them a second detector for
free.**

---

## 6. Stage C — the per-occurrence stage (`resolve_line_end_context.py`)

This is the part that does not look like the rest, and here is why.

### What it is for

`validate_line_end_chars.py` names its own blind spot in its docstring:

> *"When the misread spelling is itself a common word, no frequency table can
> detect it. 'den' and 'dem' misread as 'der' is exactly this: `der` is one of
> the commonest words in German, so a line ending in `der` looks perfectly
> normal however it got there. Only the syntax of the sentence settles it."*

Two defaults hide that class from stage A:

    AMBIGUOUS_AT = 20   observed spelling is itself a word  ->  verdict OK
    --min-length 4      der / den / dem are too short to judge  ->  verdict OK

Turning the first on alone takes the review queue from 594 to 82,836. The second
removes 71% of the class *including the docstring's own example*. Together:

    ambiguous line ends       257,715
        long  (len >= 4)       82,286   31.9%
        short (len <  4)      175,429   68.1%   <- where the model is STRONGEST

The exclusion is sound on its own terms — a function word's one-letter variants
are other real words — and it is exactly the objection a context model answers:
a function word's case is fixed by its neighbours and by nothing else.

### How it judges

```
INPUT   data/xml/*.xml   (reads the corpus itself; NOT stage A's jsonl)
        |
        +-- interior lexicon + bigrams, via validate_xml_hyphens.build_lexicon
        +-- ContextModel (context_model.py), shared with the hyphen stage
        v
SCORE   for each ambiguous line end, score every candidate reading against its
        left and right neighbour; report the MARGIN between best and observed
        v
OUTPUT  --score          margin distribution, what each gate would flag
        --evaluate       "free gold standard": held-out line-INTERIOR
                         occurrences, where the answer is known for nothing
        --calibrate      fits the noisy-channel term (r->n 508 vs 36 reverse)
        --export-sample  output/<date>_line_end_sample.csv  >>> a human
```

**It writes no correction. At all.** That is deliberate, and it is the single
most quotable number in the whole pipeline:

> Being right 89% of the time is nowhere near good enough when 99.5% of the
> input is already correct. Simulated at a 0.5% error rate, plain argmax over
> this model **breaks 4,705 correct tokens to fix 258** — 5% precision.

That is a base-rate effect. No better model fixes it; only a gate does, and a
gate trades away most of the recall.

### What the hand check found (271 rows, 2026-09-04)

45 yes / 226 no — and the errors are three populations the margin cannot
separate, which is what the script's `gate_reason` now holds back:

* **paragraph-final: 102 checked, 0 right.** Not "weaker" — worthless. The
  clearest class is `ab -> am`: *"stiegen im »Hotel Panhans« ab"* is a separable
  verb whose stem is three to six tokens back, usually on an earlier line.
* **no attested bigram on the right: 127 checked, 1 right (0.8%).** The
  substitution is on the token's *final* letter, and in German the final letter
  is what agrees with what *follows* — determiner to noun, adjective to noun.
  The left neighbour cannot discriminate an inflection.
* **capitalised on both sides: 26 wrong against 4 right.** "Frederik" for
  "Frederic", "Marchese" for "Marchesa". Whether the scan says one or the other
  is a question about the scan; no amount of sentence settles it.

Gating those out: 264 usable rows → 111, precision 17.0% → 36.0%, and 40 of the
45 confirmed corrections survive.

### And a negative result worth a slide

The obvious reading is "the window is too narrow". Right diagnosis, wrong fix:
`ContextModel` is a **bigram** model, so both readings occupy the same slot and
every term beyond the adjacent token cancels out of the margin. Re-scoring all
264 checked rows at windows of 2, 3 and 5 changes **0 margins and 0 proposals**.
More context means a higher-order model or the LLM — never a window parameter.

### Status: honest version

Phases 2–3 (`--score`, `--evaluate`, `--calibrate`) are built and have been run
over the full corpus. Phases 4–6 — the occurrence store
`data/csv/line_end_occurrences.csv`, `--export-review`, `--resolve`, and
consumption by `correct_xml_ocr.py` — are **deliberately unbuilt** until the
hand-checked sample says what the gate is worth. So today this stage is a
**measuring instrument and a detector**, not a corrector.

---

## 7. The point of the talk: the roles are swapped

This is the answer to "it's similar to the hyphen correction, but the LLM part
is different."

|  | **hyphens** | **line-end characters** |
|---|---|---|
| rule stage | `validate_xml_hyphens.py` | `validate_line_end_chars.py` |
| per-occurrence stage | `resolve_break_context.py` | `resolve_line_end_context.py` |
| LLM | `resolve_hyphens_llm.py` | `resolve_line_end_llm.py` |
| store(s) | `hyphen_decisions.csv` (pair) + `hyphen_occurrences.csv` (occurrence) | `ocr_corrections.csv` (token) + *`line_end_occurrences.csv` (planned)* |
| applied by | `correct_xml_hyphens.py` | `correct_xml_ocr.py` |
| **population** | **2,842 contested breaks** | **283,805 ambiguous line ends** |
| **who decides** | **the LLM decides every case; the score is evidence in the prompt** | **the score decides what gets ASKED; the LLM decides the answer** |

    hyphens     2,842 cases  ->  affordable to ask the model about all of them,
                                 so the context score is just evidence.

    line ends   283,805 cases -> not a queue anyone can afford. The score has to
                                 become the FILTER, which makes it load-bearing
                                 in a way it deliberately was not in the hyphen
                                 stage — so it needs its own precision measured,
                                 its own channel term, and its own gate.

Same two components, opposite wiring, and the reason is purely the size of the
population.

The second axis of difference, worth naming separately:

    resolution    hyphen pair store    per PAIR       "kaiser|in" corpus-wide
                  break occurrences    per BREAK      this line, this column
                  ocr_corrections      per TOKEN      "Wier at any line end"
                  line_end_occurrences per LINE END   (planned)

Each per-occurrence stage exists because the stage above it was forced to give
**one answer for the whole corpus** to a question whose answer changes from
column to column: *"kaiser|in"* is "Kaiserin Zita" in one column and "der Kaiser
in Wien" in the next.

---

## 8. Stage D — applying it (`correct_xml_ocr.py`)

The only script in the chain that touches `data/xml/`. Six operations, in order:

```
1  delete text lines wrongly detected inside image regions
2  replacement.csv          global string fixes   (utils/ocr_text_corrections.py)
3  OCR digit errors         1->r 0->o 2->z 5->s in years, days, Nummer/Seite
4  missing dot in times     "730 Uhr"  -> "7.30 Uhr"
5  missing slash in ranges  "191415"   -> "1914/15"
6  ocr_corrections.csv      the positional rows from stages A / B / B'
```

Two things to say about it:

**Corrections are found on the JOINED paragraph, not line by line.** Operations
2–5 used to run per `<l>` element, which missed two whole classes: a word split
across the break (`Vor¬` + `trage` — neither line contains the string a rule
matches), and a space-padded rule at a line edge. Measured over 120 files: 450
corrections in those classes against 53 a line-by-line pass finds. So the script
joins the paragraph exactly as the import will, corrects it through the *same*
`correct_text()` the database step uses, then **diffs** the result to find where
each correction landed and writes it back into the one `<l>` it belongs to. (A
diff, rather than locating each rule's match, because `replacement.csv` restores
diacritics in stages — `Lancut -> Lańcut -> Łańcut` — so the rules must run over
each other's output and there is no set of independent match positions.)

**Application is gated, unlike the hyphen store.** A row does not apply just by
existing:

    applies if   approved=yes                      (a human said so), OR
                 confidence >= --min-confidence    (default: high)
                 AND NOT approved=no               (a human retired it)

    wildcard rows apply only to a line that really ends there — never to a
    hyphenated line, where the token is a fragment rather than the word it names.
    An exact pair row is the more specific statement and wins over a wildcard.

`approved=no` retires a wrong suggestion for good without deleting the row, so a
later re-run does not propose it again.

---

## 9. Script → input → output, one table

| script | reads | writes | role |
|---|---|---|---|
| `validate_line_end_chars.py` | `data/xml/` | `output/<date>_line_end_chars.csv`; `ocr_corrections.csv` (`rule`); `output/<date>_linechar_review.jsonl` | detector, corpus statistics, per token |
| `resolve_line_end_llm.py` | that JSONL + `data/prompts/prompt_ocr_correction.txt` | `ocr_corrections.csv` (`llm`, wildcard) | judge |
| `extract_ocr_corrections.py` | `hyphen_decisions.csv` notes | `ocr_corrections.csv` (`llm`, pair) | second detector, from exhaust |
| `resolve_line_end_context.py` | `data/xml/` | reports; `output/<date>_line_end_sample.csv` | detector for the class stats cannot see — **writes no correction** |
| `ocr_corrections.py` | — | — | the store: schema, `manual > llm > rule`, `merge()` |
| `correct_xml_ocr.py` | `data/xml/`, `replacement.csv`, `ocr_corrections.csv` | `data/xml/`, `output/<date>_ocr_corrections_applied.csv` | the only writer to the XML |
| `context_model.py`, `llm_backend.py` | — | — | shared by every review loop in `lineends` |

---

## 10. The three sentences to say out loud

1. **Statistics find what is rare, an LLM finds what is wrong.** The corpus can
   tell you `Wier` never occurs inside a line; only the sentence can tell you
   that `Reiss` is somebody's name.
2. **A detector may never write into the source.** Every proposal lands in a CSV
   with a source rank, and a human outranks the model, which outranks the
   corpus. The text is corrected by one script, from the store, and by nothing
   else.
3. **The base rate decides the architecture.** With 99.5% of line ends already
   correct, a model that is right 89% of the time breaks eighteen words for
   every one it fixes. That is why the third stage measures instead of
   correcting — and why the LLM decides everything in the hyphen chain and only
   the *shortlisted* cases here.
