# Plan: Per-Occurrence Line-End Character Correction

> **Status: phases 2-3 implemented 2026-08-30** as
> `lineends/resolve_line_end_context.py`; the phase-1 hand-check sample is
> generated and waiting on a person. Phases 4-6 (the store, `--export-review`,
> `--resolve`) are not built, deliberately: they should not be until the sample
> says what the gate is worth. Measured figures from the shipped code are in
> "What the implementation measured" at the end and supersede the estimates
> below where they differ.
>
> Originally proposed 2026-08-29, revised the same day after a critical pass. Every number below was measured on the current corpus
> (787 files, 750,692 judged line ends, `ad`/`ad-frame` excluded), reusing the
> `ContextModel` and the free-gold-standard method built for
> `docs/hyphen-occurrence-disambiguation.md`.
>
> The first draft of this plan overstated the model (0.931 against a candidate
> set narrower than the one the pipeline actually uses, on a class-balanced
> sample) and understated the population by 3.5x. Both are corrected below;
> what changed and why is recorded in "Corrections to the first draft" at the
> end, because the *kind* of error is instructive and the same trap is still
> open elsewhere.

## Background

The hyphen stage moved a decision from the *pair* to the *occurrence* and it
worked. The obvious question is whether the same move helps the other line-end
error class — `validate_line_end_chars.py`'s misread final character, "Wien"
transcribed as "Wier".

It does, but not in the same shape, and the differences matter more than the
similarity. **The roles of the context model and the LLM have to swap.**

There are two separate problems here, and only the second is large.

### Problem A — wildcard rows are global by token

`ocr_corrections.py` keys a line-final fix by the token alone, `right = *`,
meaning "this token wherever it ends a line". That is the same corpus-wide
resolution the hyphen pair store had, with the same consequence: `dorf` → `dort`
fires on every line-final `dorf`, including the place names. The README already
names the two failure modes — surnames and place names, and truncated rather
than substituted tokens.

Size: **1,050 tokens / 1,647 occurrences** (`MISREAD`), plus 271 tokens / 594
occurrences already routed to `REVIEW`. Small, and the existing
`rule` → `llm` → `manual` ranking plus `approved=no` already gives a human a way
out. Worth an occurrence store, but it is not where the value is.

### Problem B — the class statistics cannot see at all

`validate_line_end_chars.py` says so itself:

> *"When the misread spelling is itself a common word, no frequency table can
> detect it. 'den' and 'dem' misread as 'der' is exactly this: `der` is one of
> the commonest words in German, so a line ending in `der` looks perfectly
> normal however it got there. Only the syntax of the sentence settles it."*

Those are switched off by default (`AMBIGUOUS_AT = 20`, `--ambiguous`), and
correctly so — turning them on turns 594 review items into 82,836:

    default        OK 748,451 (99.70%)   MISREAD 1,647   REVIEW    594  (271 tokens)
    --ambiguous    OK 666,209 (88.75%)   MISREAD 1,647   REVIEW 82,836  (1,882 tokens)

**But `--ambiguous` is only one of two switches, and it is the smaller one.**
See Finding 2: a second default excludes 71% of the class before `--ambiguous`
is ever consulted, and the docstring's own headline example is inside the
excluded part. The real population is **283,805 line ends**, not 82,242.

---

## Findings

### Finding 1 — the context model transfers, and beats the frequency baseline

The `ContextModel` built for the hyphen stage, unchanged, scoring
`(previous, candidate, next)` for each candidate and taking the best.
Evaluated on the free gold standard — line-interior occurrences, where the true
token is known and no line break can have damaged it — **at the candidate space
the pipeline actually uses** (`--confusions all`: every attested one-character
substitution of the final letter, up to 22 candidates) and at the corpus's
**natural class frequencies**:

| n (held out) | model | pick-most-frequent-candidate baseline |
|---|---|---|
| 40,000 | **0.891** | 0.711 |

Per confusion set, on a class-balanced sample so the per-class recall is
visible (this is a diagnostic, not the headline — see the corrections section):

```
  1,200   0.796   dem/den/der          800   0.961   am/an
  1,200   0.852   vom/von/vor          800   0.939   zum/zur
  1,200   0.869   diesem/diesen/dieser 800   0.921   im/in
  1,200   0.948   großem/großen/großer
```

German case marking is exactly what a bigram captures: "in dem Hause" against
"in den Häusern" is settled by the following noun, and that noun is the next
token. This is the model's best case, not a stretch of it.

There is also no length asymmetry to correct for. Both readings are one token,
so the smoothing trap that broke the hyphen model's first version
(`BIGRAM_WEIGHT`, balanced accuracy 0.557) cannot arise in the same form.

### Finding 2 — a second default hides 71% of the class, including the headline case

`der`, `den` and `dem` — the example the script's own docstring leads with —
**do not appear in its report at all**, even with `--ambiguous`. They are
removed a step earlier:

```python
if len(ev.word) < args.min_length:          # default 4
    ev.verdict, ev.reason = 'OK', 'too-short'
```

with a reason that is sound on its own terms:

> *"Short words are mostly function words whose one-letter variants are other
> real words; judging them by frequency invites false positives."*

That is true, and it is precisely the objection a context model answers. The
short band is where the model is **strongest** — `am`/`an` 0.961, `zum`/`zur`
0.939, `im`/`in` 0.921 — because a function word's case is fixed by its
neighbours and by nothing else. Frequency cannot see it; context barely has to
try.

Splitting the ambiguous population at that boundary:

| band | line ends | share |
|---|---|---|
| long — `len >= 4`, judged today | 81,098 | 28.6 % |
| **short — `len < 4`, excluded outright** | **202,707** | **71.4 %** |
| total | 283,805 | |

So there are two switches, not one, and this is the bigger of them. **This
finding is also what corrects the first draft's population figure** (82,242):
that number counts only the long band, and only the part of it the validator's
own report lists.

Whether the short band should actually be opened is a phase-1 question, not a
foregone conclusion — what it flags is visibly more mixed (Finding 4).

### Finding 3 — and it must not be allowed to correct anything

Being right 93% of the time is nowhere near good enough, because 99.5% of the
input is already correct. Simulating a 0.5% error rate over 60,000 line ends
drawn from the ambiguous class, injecting substitutions within each confusion
set, and applying the model by plain argmax:

```
 margin   changed   fixed   broken  precision  recall
      0     4,980     258    4,705       5.2%   88.4%
      2     1,154     236      910      20.5%   80.8%
      4       500     215      279      43.0%   73.6%
      6       330     193      134      58.5%   66.1%
      8       191     143       46      74.9%   49.0%
     10        92      84        8      91.3%   28.8%
     12        62      59        3      95.2%   20.2%
```

**Ungated, the model breaks 4,705 correct tokens to fix 258.** This is the
whole difference from the hyphen stage and it is a base-rate effect, not a
model defect: there, a contested pair's two readings were both genuinely live,
so argmax was a fair question to ask. Here the transcription is right almost
always, and the only reason to overturn it is evidence strong enough to beat
that prior.

Note what this does to the plan's own thresholds: `broken` barely moves with the
assumed error rate while `fixed` scales with it, so precision is sensitive to a
number nobody has measured (see Risks).

### Finding 4 — as a *detector* it is affordable and finds real errors

Scoring every real line end in the ambiguous class — all 283,805, at the full
candidate space — and counting what each gate would flag, split at the
`--min-length` boundary of Finding 2:

| margin ≥ | long band | short band | total | share of 283,805 |
|---|---|---|---|---|
| 6 | 1,557 | 2,725 | 4,282 | 1.51 % |
| 8 | 933 | 980 | 1,913 | 0.67 % |
| 10 | 449 | 356 | 805 | 0.28 % |
| 12 | 235 | 131 | 366 | 0.13 % |
| 16 | 91 | 44 | 135 | 0.05 % |

That is the number that makes this stage possible. **283,805 candidates is not
an LLM queue; 4,282 is, and 805 comfortably is.**

What the **long band** flags at margin ≥ 10, read straight off the corpus:

```
              des [älterer  -> älteren]  Sohnes            +16.4   ok
               im [erster   -> ersten]   Stockwerke        +17.8   ok
         Toilette [weißer   -> weißem]   Silberbrokat      +12.6   ok
              von [schwarzer-> schwarzen] Chantillyspitzen +17.7   ok
        technisch [vollendeter -> vollendeten] Wiedergabe  +24.5   ok
            eines [Geheimer -> Geheimen] Rates             +25.8   ok
          Kreisen [beider   -> beiden]   Reichshälften     +15.2   NO - correct genitive
           Freiin [Magdalena-> Magdalene] Sardagna         +12.8   NO - a name
         Borghese [Duchessa -> Duchesse] Terranova         +14.7   NO - a title
```

Six of nine are real OCR errors nothing in the pipeline currently sees. Two of
the three failures are **proper names** — the same failure mode the statistical
detector has and the prompts already warn about.

What the **short band** flags at margin ≥ 12 is more mixed, and more
interesting:

```
          Dentice [der -> dei] Principi        +13.4   ok   Italian nobiliary
         Salviati [der -> dei] Duchi           +13.5   ok   the same
            Silva [v   -> y  ] Carvajal        +21.6   ok   Spanish "y"
          Bischof [vor -> von] Steinamanger    +18.4   ok
         Pongratz [den -> der] LinienschLt     +12.7   ok
           Bruder [den -> der] dreizehn        +13.1   ?
          gewesen [is  -> in ] die             +13.0   NO - truncated "ist"
             Cary [T   -> v  ] Grayson         +12.3   NO - a middle initial
```

Roughly five of eight, and it catches things nothing else would: the Italian
`dei` and the Spanish `y` in noble names are invisible to a German frequency
lexicon. It also exposes **a failure mode this plan had not identified: single
letters.** "Cary T. Grayson" has an initial at the line end, and every letter of
the alphabet is an attested candidate for it. A minimum length of 2, and
probably excluding lone capitals outright, is needed before the short band is
opened at all.

Two-thirds is a fine hit rate for an LLM queue. It is a catastrophic one for
automatic correction, which is the same conclusion as Finding 3 from the other
direction.

### Finding 5 — the noisy channel is real, asymmetric, and already measured

A flat margin treats every substitution alike. The corpus does not:

```
   r->n:   508 occurrences      n->r:    36      ratio  14.1 : 1
   s->e:   100                  e->s:    13             7.7 : 1
   s->n:    57                  n->s:    12             4.8 : 1
   e->a:    39                  a->e:    15             2.6 : 1
   m->n:    39                  n->m:    21             1.9 : 1
```

(Read `r->n` as: the OCR produced `r` where `n` was meant.) A final `n` misread
as `r` is fourteen times likelier than the reverse, which is the physical fact
the whole detector was built on — the letterform clipped by the margin.

So the right scoring rule is the classic noisy channel,
`P(context | candidate) × P(observed | candidate)`, not context alone. The
second term is estimable **from data that already exists**: the 1,050 confident
`MISREAD` tokens are a labelled sample of what the OCR does to a final
character. A per-substitution prior would let `r→n` clear a lower bar than
`n→r`, which a single global margin cannot express.

### Finding 6 — do not pre-filter the candidates by frequency

The ambiguous class is concentrated — within the long band, 66 tokens cover
half of its 82,242 reported occurrences — but the head of that list is not where the errors are:

```
     Wien -> Wied   3,608 line ends      Graf -> Graz   3,108
    Franz -> Frank  1,351               Karl -> Kard   1,139
   seiner -> seinen 1,495              einem -> einer  1,142
```

`Wien` and `Graf` are flagged only because `Wied` and `Graz` are themselves
attested words (a Fürst zu Wied, the city of Graz). They are candidates, never
corrections: the context model gives them a hugely negative margin and the gate
never fires. The inflectional pairs below them — `seiner`/`seinen`,
`einem`/`einer`, `ihrem`/`ihrer` — are where the real ambiguity is.

The lesson from the hyphen plan applies unchanged: **let the margin do the
filtering, not a frequency gate on the candidate.** Scoring all 283,805 costs
seconds of arithmetic; only the LLM queue needs to be small, and the margin is
what makes it small. A frequency pre-filter would have to be tuned to exclude
`Wien` while keeping `seiner`, and there is no such threshold — that is the
same mistake as the ratio gate in the hyphen plan.

### Finding 7 — the model leans on the least reliable half of its context

For a line-final token `w`, the two conditioning tokens are not equally
trustworthy:

    previous   the token before `w` on the same line — line-interior, the
               position the whole lexicon is built from precisely because a
               line break cannot damage it
    next       the first token of the *next* line — a line-initial position,
               which `build_lexicon` excludes from the reference for the same
               reason it excludes line-final ones

Ablating them on the gold standard:

| context used | accuracy |
|---|---|
| previous + next | 0.891 |
| next only | 0.833 |
| previous only | 0.818 |

**`next` carries more of the signal than `previous` does**, and `next` is the
half drawn from a suspect position. The gold standard cannot show this cost,
because its `next` is always an undamaged interior token — so 0.891 is an
optimistic estimate of live performance by an unknown margin, and the phase-1
sample is the only thing that can price it.

The hyphen stage does not have this problem, and it is worth being explicit
about why rather than assuming the two stages are alike: there, `after[0]` is
the *second* token of the next line (`tail[1:]`), because the first one is the
right half of the pair being judged. Its context is interior on both sides.

A related consequence: **the last line of a paragraph has no next token at
all** — 18,760 of the ambiguous class, and 145,898 line ends in the corpus
overall. They fall back to `previous` only, at 0.818, and a run must report them
as their own band rather than silently scoring them on half the evidence.

That band is not permanent; see "The paragraph-final gap" below.

---

## Design

### Roles swapped, relative to the hyphen stage

    hyphens     LLM decides every contested break; the score is evidence in
                the prompt. Affordable because there are only 2,842 of them.

    line ends   The score decides *what gets asked*; the LLM decides the
                answer. Not affordable otherwise: 283,805 against 2,842.

The score is therefore load-bearing here in a way it deliberately was not
there, and everything below follows from that: it needs the channel term
(Finding 5), it needs its precision measured rather than assumed (Finding 4),
and it must still never write a correction by itself (Finding 3).

### `data/csv/line_end_occurrences.csv` — a new store

Mirrors `break_occurrences.py` exactly, including the reasoning: keyed by
`(file, line_facs)`, because the token in question is the last one on that
line, and the `facs` id is the only stable name for it. Same
`manual > llm > rule` ranking, same stale-key check (the stored `observed` must
still be the token at that line, or the row is reported and skipped).

```
file, line_facs, observed, corrected, score, channel, source, model, decided_on, note
```

`corrected == observed` is a meaningful row and not a no-op: it records that
this occurrence was looked at and left alone, which is what stops it being
asked again on the next run.

Relationship to `ocr_corrections.csv`: unchanged, and consulted first. A
wildcard row still says what a token usually becomes at a line end; an
occurrence row overrules it for one line, exactly as `hyphen_occurrences.csv`
overrules the pair plan. The two stores answer different questions and are not
merged, for the reason `ocr_corrections.py`'s own docstring gives about
`replacement.csv`.

### `lineends/resolve_line_end_context.py`

```
--score            score every ambiguous line end; report the margin
                   distribution and what each gate would flag (writes nothing)
--evaluate         the free gold standard: held-out line-interior occurrences,
                   balanced accuracy and per-class recall, per confusion set
--calibrate        fit the channel term from the confident MISREAD rows and
                   report what it changes
--export-review    the flagged line ends as JSONL with sentence context
--resolve          the local model over that queue, into the occurrence store
```

`ContextModel`, `read_paragraph_lines`, the JSONL conventions, the sharding and
flush behaviour and `llm_backend.py` all exist. The genuinely new pieces are the
channel term and the store.

### Phases

1. **Measure the real error rate.** Hand-check a stratified sample of ~300
   flagged line ends, stratified over **three axes at once**, because each one
   is a place the plan is currently guessing: margin band (6/8/10/12+), token
   band (short against long, Finding 2), and whether a next token exists
   (Finding 7). Every number in Findings 3 and 4 rests on an assumed 0.5% error
   rate and an eyeballed two-thirds; neither is evidence. **This is the phase
   that decides whether the rest is worth building**, and it needs no code
   beyond `--score` and an export.

   It also settles the two questions this plan deliberately leaves open: what
   `--min-length` should become, and whether the short band is worth opening at
   all.

2. **`--score` and `--evaluate`.** The diagnostic pair, built before anything
   that writes — the same order that caught the broken smoothing in the hyphen
   stage, where the evaluator earned its cost immediately.

3. **`--calibrate`.** The channel term from the `MISREAD` rows, scored against
   the phase-1 sample. Keep it only if it beats the flat margin on that sample;
   an untested refinement is a liability, not an improvement.

4. **The store and `--export-review`.** Gate at whatever margin phase 1 shows
   to be worth a person's or a model's time — the table in Finding 3 is the
   menu, not the answer.

5. **`--resolve`**, with a prompt derived from `prompt_ocr_correction.txt`,
   which already forbids modernising historical orthography and already warns
   about proper names and truncations. Both warnings matter more here than
   there.

6. **Consumption.** `correct_xml_ocr.py` reads the occurrence store the way
   `correct_xml_hyphens.py` now reads `hyphen_occurrences.csv`. Note this lands
   in the *joined-text diff* path, which is more delicate than the hyphen
   correction's single-character rewrite — see the `changes-break` and
   `join-artifact` outcomes in that script.

Phase 1 is the whole decision. Phases 2–3 are cheap and reusable whatever it
says.

---

## Risks and open questions

- **The error rate in this class is unknown, and every precision figure depends
  on it.** It cannot be measured by the method that found the others, since the
  defining property of the class is that frequency cannot see it. `broken`
  is roughly independent of the rate while `fixed` scales with it, so at 0.1%
  margin-12 precision falls to about 80% and at 2% it rises past 99%. Phase 1
  exists solely to replace this guess.

- **The simulation and the corpus disagree**: 91% predicted at margin ≥ 10
  against roughly two-thirds observed by eye. The simulation assumed every
  error lands inside the confusion set, which is exactly the assumption the
  truncation class ("Scht", "Geno") violates. Trust the corpus.

- **Proper names are the standing failure mode** — `Magdalena` → `Magdalene`,
  `Duchessa` → `Duchesse`. This corpus is largely names, the statistical
  detector already fails this way, and a context model has no special defence:
  a name sits in a grammatical slot like any noun. Worth testing whether
  excluding capitalised tokens whose lowercase form is unattested removes more
  noise than signal.

- **The free gold standard measures only half the model.** Line-interior
  occurrences carry no OCR error, so they can score `P(context | candidate)`
  and say nothing about `P(observed | candidate)`. The channel term can only be
  evaluated against phase 1's hand-checked sample, which is another reason that
  sample has to exist before anything is built on it.

- **Trigrams are the obvious next lever and were not tested.** For the hyphen
  stage a wider window was worth exactly nothing, because in a bigram model the
  extra terms cancel out of the margin. **That reasoning does not carry over**:
  here the two readings differ in the *identity* of one token rather than in
  the number of tokens, so a trigram spanning it would contribute genuinely
  different terms to each reading. Finding 7's ablation is direct evidence for
  this: each neighbour alone is worth 0.818 and 0.833, and together 0.891, so
  context is genuinely additive here in a way it provably was not there. This
  is the one place where more context should actually pay, and it should be
  measured on the free gold standard before any model is adopted.

- **`--ambiguous` and `--min-length` stay at their defaults.** Nothing in this
  plan changes what a default run of `validate_line_end_chars.py` reports; the
  new script asks for the wider class explicitly. Lowering `--min-length` in
  the validator itself would flood its report with function words it has no way
  to judge, which is the behaviour the default is there to prevent.

- **Region scope is now one shared decision**, not a per-script default:
  `CORRECTED_SUBTYPES` / `SKIPPED_SUBTYPES` in `validate_xml_hyphens.py`.
  Corrected: `paragraph`, `heading`, `heading-sub`, `image-caption`,
  `image-credit`, `ad-content`, `imprint`, `volume`, `footnote`. Skipped: `ad`,
  `ad-frame` (unusable OCR), `image`, `mode` (no text — what the line model
  reports there is stray marks read as characters), and the three `separator*`
  classes. **Every measurement in this plan predates that alignment** and used
  the older `ad,ad-frame` scope, so `image`, `mode` and the separators are
  inside the figures. They contribute ~1,500 lines between them, so nothing
  here turns on it — but `image` and `mode` are pure single-character noise
  (`K`, `2`, `8`) and single characters are exactly the failure mode named
  below, so a rerun under the corrected scope should come before phase 1's
  sample is drawn.

- **Single-letter tokens must be excluded before the short band is opened.**
  A middle initial at a line end ("Cary T. | Grayson") has all 29 letters as
  attested candidates, and the model will confidently propose one. A minimum
  length of 2, and probably a rule against lone capitals, is a prerequisite and
  not a refinement (Finding 4).

## Deliberately not done

- No automatic correction from the score, at any margin. Finding 2 is the whole
  reason this stage exists in the shape it does.
- No merge of `line_end_occurrences.csv` into `ocr_corrections.csv`. Positional
  and per-occurrence are different claims, and the store that already keeps
  positional apart from global should not be the one to blur it.
- No attempt on the truncation class ("Scht", "Geno"). Several characters are
  missing, no single substitution repairs them, and a candidate set built from
  one-character variants cannot contain the answer. Still a different detector,
  as `docs/ocr-word-correction.md` says.


---

## Corrections to the first draft

Recorded because the *kind* of mistake recurs, and two of these three traps are
still open elsewhere in this repository.

**The model was evaluated on an easier problem than the one it will face.**
The first draft reported 0.931. That was measured against a hand-written
confusion table (`r->nm`, `n->rm`, …) while the pipeline's own default is
`--confusions all`, every letter for every other — a candidate set of up to 22,
not 3. It was also measured on a class-balanced sample (400 per member), which
put the majority baseline at 0.333 for a three-way set and made the model look
better than the corpus warrants. Re-measured at the real candidate space and
the corpus's natural frequencies: **0.891 against a 0.711 baseline.** The
conclusion survives; the margin over baseline is 0.18, not 0.22.

*The trap:* evaluating against a candidate space you chose rather than the one
the system uses. The balanced-sampling half of it is the same mistake the
hyphen plan warns about in the opposite direction — there the danger was raw
accuracy flattering a skewed class, here it was balanced sampling flattering
the model against a deflated baseline.

**The population was understated 3.5x**, and by a mechanism worth naming: the
first draft took its figure from the validator's own report, which is filtered
by `--min-length 4` before anything else. Reading a number off a report without
checking what the report excludes is how `der`/`den`/`dem` — the case the
script's docstring leads with — came to be absent from a plan about exactly
that case. **283,805, not 82,242.**

*The trap:* trusting a tool's output as a measurement of the world rather than
of the tool. This is the same class of error as the `read_paragraphs` regression
recorded in `docs/line-end-correction.md`, where a filter
silently emptied the corpus and the analysis reported zero rather than failing.

**The queue sizes were computed from a truncated candidate set** — two
candidates per token, taken from the report's `suggested` and `runner_up`
columns, rather than all attested substitutions. Corrected in Finding 4: at
margin ≥ 6 the queue is 4,282, not 1,400.

**Two findings the first draft missed entirely**, both now Findings 2 and 7:
the `--min-length` boundary, and the fact that the model leans on `next` — the
half of its context drawn from a damaged position — more heavily than on
`previous`. Neither is visible without ablating the model or reading the
validator's control flow, and neither would have surfaced from re-reading the
plan.


---

## The paragraph-final gap (depends on the XML refactor)

Design only. `merge_xml_factoids.py` does not exist yet — see
an XML-layer refactor in the originating project.

**145,898 line ends** in scope are a paragraph's last line. Unlike the hyphen
stage, where such a break is *undecidable* (there is no second word to pair),
here the token is judged normally by frequency — the existing scan reads
paragraphs with `min_lines=1` deliberately, because a line end needs no
following line. What is missing is only the **context model's `next` token**,
and Finding 7 says that is the more informative half: 0.818 without it against
0.891 with it.

So the gap costs accuracy on 145,898 line ends rather than hiding them
entirely, and it is concentrated precisely where the evidence is thinnest.

`merge_xml_factoids.py` writes `@next`/`@prev` across fragments without moving
any `<p>`, so the fix is the same one-place change as for the hyphens: **when
the following line is not in this `<p>`, follow `@next` to the first line of
the fragment that continues it.** The scoring, the store and the queue are
untouched.

Two consequences specific to this stage:

  * **The free gold standard is unaffected**, because it is built from
    line-interior occurrences that never cross a paragraph boundary. So the
    evaluator can measure the improvement without itself changing — score the
    same held-out set, first with `next` withheld and then with it supplied,
    and the difference is exactly what closing this gap buys. That comparison
    is already available today (0.818 against 0.891) and should be re-run per
    band once the chain exists.
  * **Phase 1's sample must stratify on it** — it is already one of the three
    axes — because a hand-checked precision figure drawn mostly from
    paragraph-final line ends would understate the system, and one drawn mostly
    from mid-paragraph ones would overstate it.

The marker and warning design is shared with the hyphen stage and is written up
there: check `merge_xml_factoids`' `<application>` marker per file via
`xml_io.read_marker_version`, warn rather than fail, fall back to today's
behaviour, and **count the affected line ends in the report** rather than
emitting a warning line that scrolls past.

The ordering question recorded there — whether classification and merging need
corrected text, given that the correctors would now run after them — is settled
and needs no work here: factoid classification is geometric, heading
classification has a manual override layer, and the continuation headings are
hand-checked. The correctors move to the end of `xml_io.STEPS` unchanged.


---

## What the implementation measured

`resolve_line_end_context.py` exists and has been run over the full corpus
under the corrected region scope, with `--min-length 2` and lone initials
excluded (Finding 4's "Cary T. | Grayson"). Where these differ from the
estimates above, these are right.

### Population and queue

    749,027  judgeable line ends
    257,715  ambiguous (observed spelling is itself a word, and at least one
             attested one-character substitution of its final letter is too)
               long   82,286  (31.9%)
               short 175,429  (68.1%)
               no next 16,089 (6.2%, a paragraph's last line)

| margin ≥ | flagged | long | short | no next |
|---|---|---|---|---|
| 4 | 5,716 | 2,124 | 3,592 | 532 |
| 6 | 3,621 | 1,559 | 2,062 | 346 |
| 8 | 1,768 | 934 | 834 | 187 |
| 10 | 750 | 449 | 301 | 56 |
| 12 | 357 | 235 | 122 | 7 |
| 16 | 132 | 91 | 41 | 0 |

The shape of Finding 4 holds: an affordable queue at every gate worth using.
The substitution the model proposes most often at margin ≥ 8 is `r -> n` (408
of them), which is the error the whole detector was built around — the model
found that class from context alone, having been told nothing about it.

### Accuracy, and where it comes from

Over 40,000 held-out line-interior occurrences at natural frequencies:

    context model                  0.931
    pick the commonest candidate   0.761
    weighted balanced accuracy     0.823   against a 0.754 majority baseline

**The two numbers disagree for a reason worth acting on**, and it is the
sharpest thing this run produced. Per candidate set:

```
              candidates       n  balanced  majority
 einem/einen/einer/eines     516     0.833     0.335
 ihrem/ihren/ihrer/ihres     404     0.821     0.369
   seinem/seinen/seiner…     370     0.847     0.505
 dei/del/dem/den/der/des   6,858     0.847     0.512
                   im/in   2,947     0.917     0.719
             und/ung/uns   3,761     0.672     0.984   <- worse than the prior
                 die/dir   1,957     0.750     0.999   <- worse than the prior
                   zu/zw     935     0.500     0.991   <- worse than the prior
               Wied/Wien     446     0.625     0.991   <- worse than the prior
```

The model's discriminative power is concentrated almost entirely in the
**balanced, inflectional sets** — the German case endings, where it beats the
prior by 30-50 points. On **lopsided sets** it adds nothing and scores below
the prior: it cannot reliably recover a true `Wied` from context, because
almost nothing in the sentence distinguishes it from `Wien`.

This does **not** overturn Finding 6. A lopsided set costs nothing in
deployment, because deployment is anchored on the observed token and gated on
the margin, so `Wien` is never overturned without overwhelming evidence and
in practice never is. What the split says is narrower and still useful: **the
recall on real errors will be poor in exactly those sets**, so a hand-checked
precision figure drawn from them would be measuring a different system from one
drawn from the inflectional sets. Phase 1's sample stratifies on margin, token
band and has-next; on this evidence it should stratify on set balance too, or
at least report it.

### The channel, confirmed

`--calibrate` reads the confident `MISREAD` verdicts and finds what Finding 5
predicted, with `g -> s` a new entry near the top:

    r -> n   508   reverse 36    ratio 14.1
    g -> s    17   reverse  1    ratio 17.0
    s -> e   100   reverse 13    ratio  7.7
    e -> d    16   reverse  3    ratio  5.3
    s -> n    57   reverse 12    ratio  4.8

Still unwired, and deliberately: it is phase 3, and it has to be scored against
the phase-1 sample before it is trusted with a gate.

### Two implementation notes worth keeping

**`ContextModel` moved to `context_model.py`.** Two callers now, and lifting it
out exposed a coupling that had been invisible with one: `margin()` read a
module-level `SCORE_WINDOW` from the script it lived in. It is a parameter now.
The hyphen stage's evaluation is byte-identical before and after (0.835), which
is what makes that a refactor rather than a change.

**The lexicon is built by `validate_xml_hyphens.build_lexicon`, not inline.**
The first version counted adjacent interior pairs directly, which is three
lines shorter and quietly different: `build_lexicon` sorts a pair into
`bigrams`, `periods` or `commas` by the punctuation between them, so an inline
count folds "aus. Karl" in with "aus Karl". Two scripts scoring against the
same claimed reference while counting differently is the failure this codebase
has already been bitten by once.

### What phase 1 now needs

`output/20260830_line_end_sample.csv` — 272 rows over 17 strata, `verdict`
blank for `y`/`n`. Everything downstream of it is unbuilt on purpose.
