# Plan: Per-Occurrence Break Disambiguation

> **Status: phases 1-5 implemented 2026-08-29**, phase 6 (`--evaluate`) built
> and run. Six things turned out differently from the plan and are recorded in
> "What changed during implementation" at the end - including one design error
> the free gold standard caught before anything was applied, and two rows the
> plan would have "fixed" into corruptions. What is left is the LLM run itself,
> which needs a local server, and the decision whether to apply the promotion.
>
> Every number below was measured on the current corpus (787 files, 825,525
> in-paragraph line breaks, `ad`/`ad-frame` excluded).

## Background — the ceiling the pair-level store hits

`validate_xml_hyphens.py` asks one question per distinct `(left, right)` pair
and `hyphen_decisions.csv` stores one answer per pair. That is the right shape
for the overwhelming majority of breaks: `Prin|zessin` is `join` wherever it
occurs, `Windisch|Graetz` is `hyphen` wherever it occurs, and settling it once
is what makes a re-run over a growing corpus cheap.

It is the wrong shape for pairs where **both readings are real German** and only
the sentence says which is meant:

    kaiser|in       "Kaiserin Zita"          vs  "der Kaiser in Wien"
    wie|der         "wieder in der Stadt"    vs  "wie der Bruder"
    nach|dem        "nachdem Er verfügt"     vs  "Witwe nach dem Grafen"
    an|deren        "anderen Banken"         vs  "an deren Bord"
    außer|dem, des|gleichen, seit|dem, Botschafter|in …

For these there is no single answer to store, so whichever verdict is written is
applied to every occurrence and the occurrences reading the other way are
corrupted.

---

## Evaluation of the embedding proposal

The colleague's instinct is right: **the decision has to move from the pair to
the occurrence, and what decides it is the surrounding context.** That is the
load-bearing part of the idea and this plan adopts it.

The instrument is wrong, in the form embeddings are usually meant.

**Why cosine similarity of sentence embeddings does not work here.** A sentence
embedding is trained to be *invariant* to exactly the difference being decided.
"Se. Majestät kehrte wieder nach Wien zurück" and "… wie der nach Wien zurück"
describe the same scene with the same content words; an encoder places them
within a hair of each other and the residual distance is noise, not
grammaticality. There is also nothing to compare *against* — cosine needs a
reference vector and the pair supplies none. Similarity is the wrong question.
The question is **which of these two strings is a probable German sentence**,
which is a likelihood question.

**Two instruments that do answer it:**

1. **A language-model score** — score the sentence under each reading and take
   the higher probability. This plan starts with a bigram LM built from the
   corpus's own line-interior tokens, which needs no new dependency and reuses
   the tables `build_lexicon()` already produces (Finding 4).

2. **A supervised classifier over contexts, trained on labels the corpus
   supplies for free.** This is the salvageable and genuinely good
   embedding-shaped idea. For any contested pair the corpus already holds
   *undamaged* examples of both readings — inside lines, where no break can have
   touched them. Every line-interior `sowie` is a labelled `join` example with
   context; every `so wie` a labelled `split`. Same newspaper, same register,
   same historical orthography, no annotation cost (Finding 6). Here a context
   embedding is a perfectly reasonable *feature representation* — feeding a
   classifier with real labels, not a similarity score against nothing.

**Verdict: feasible, worth doing, and far smaller than expected.** The affected
population is 61 pairs / 1,978 occurrences (Finding 1) — small enough that the
whole job can go to the existing local-LLM infrastructure directly, with the
n-gram score serving as evidence and cross-check rather than as a throughput
optimisation.

---

## Findings

### Finding 1 — the gate is the *minority reading's* absolute count, not a ratio

The obvious way to find contested pairs is to compare `join_count` (unigram
`left+right`) against `separation` (`split_count + comma_count + period_count`)
in the line-interior lexicon. **How** they are compared matters more than it
looks, and the codebase's existing idiom is the wrong one here.

A **ratio gate** — the `--ratio 50` idiom `decide_form()` already uses — asks
whether one reading dominates. Applied to this question it removes exactly the
cases that motivate the stage:

| pair | occ | join# | sep# | ratio | contested at 1/20? |
|---|---|---|---|---|---|
| `wie\|der` | 163 | 5,347 | 131 | 40.8 | **no** |
| `an\|deren` | 48 | 834 | 36 | 23.2 | **no** |
| `König\|in` | 2 | 2,399 | 28 | 85.7 | **no** |
| `so\|wie` | 221 | 4,888 | 48 | 101.8 | no |
| `nach\|dem` | 284 | 307 | 1,413 | 4.6 | yes |
| `Botschafter\|in` | 39 | 113 | 388 | 3.4 | yes |

A ratio gate drops `wie|der`, `an|deren` and `König|in` — and the hand probe in
Finding 4 found real "wie der" and "an deren" readings among their occurrences.
**A lopsided prior is the failure mode, not a reason to dismiss the pair:** it
is precisely when one reading is 40× commoner that the rare reading gets
steamrollered by a global verdict.

An **absolute gate on the minority reading** asks the right question — is the
rare reading a live option at all? — and needs only a noise floor:

| minority interior count ≥ | pairs | occurrences | share of breaks | per issue |
|---|---|---|---|---|
| 2 | 318 | 4,828 | 0.58 % | 6.1 |
| 5 | 128 | 3,219 | 0.39 % | 4.1 |
| **10** | **61** | **1,978** | **0.24 %** | **2.5** |
| 20 | 35 | 1,687 | 0.20 % | 2.1 |

At ≥ 2 the set is full of OCR noise — `mit|der` (join# 6 against sep# 3,215,
"mitder" is not a word), `die|am` (2 / 376), `eben|so` (573 / 2), `um|so`
(2 / 145). **≥ 10 is the natural floor**: it admits every genuinely ambiguous
pair named above that exists in the corpus and excludes all of the noise.

> Note for the record: `eben|so`, `außer|dem` and `des|gleichen` were named as
> examples. On *this* corpus `eben|so` is not contested (573 : 2 — "eben so" is
> essentially never the reading); `außer|dem` (161 : 73) and `des|gleichen`
> (13 : 18) are. Which pairs need this treatment is a measurement, not an
> intuition, and the measurement is per corpus.

**Take `--min-minority 10` as the default and 61 pairs / 1,978 occurrences as
the size of the job.**

### Finding 2 — which pairs are covered, and why they are almost all LLM-decided

*This answers the question directly: the stage is scoped by the contested test,
not by who decided the pair. All three sources are eligible. In practice one
source dominates, and the reason is itself the bug.*

The 61 contested pairs, by who settled them today:

| settled by | pairs | occurrences |
|---|---|---|
| store: `llm` | 24 | 1,228 |
| store: `manual` | 22 | 511 |
| undecided (`REVIEW` / no form) | 14 | 177 |
| **statistics alone** | **1** | **62** |

Only one contested pair was ever settled by statistics without review —
`Kunstdruck|Ausgabe` (`hyphen`, verdict `OK`, nothing rewritten). That is not
because statistics-decided pairs are exempt from the problem. It is because
`judge()` already detects contestedness and refuses to decide it:

```python
if ev.form == 'join':
    ...
    if ev.separation >= 2 and not settled:
        ev.reason = 'contested-both-readings'
        return 'REVIEW'
```

```python
if (ev.join_count >= 2 or ev.dash_count >= 2) and not settled:
    ev.reason = 'contested-both-readings'
    return 'REVIEW'
```

So a contested pair is routed to `REVIEW` → `resolve_hyphens_llm.py` → the
store. **And `and not settled` then disables the guard permanently.** The review
queue is the funnel that converts "this pair is contested" into a global answer,
and the moment the LLM replies, the pipeline stops knowing it was ever
contested.

That is the architectural bug in one sentence, and it means:

- **the pipeline already identifies the right population** — the contested test
  exists and works; what is missing is a place to put a per-occurrence answer;
- **the new stage must intercept before the store is written**, not after, or it
  will be re-deriving a fact the pipeline already had and threw away;
- the two guards' thresholds (`separation >= 2`, `join_count >= 2`) are the same
  too-loose absolute floor Finding 1 rejects, and should move to
  `--min-minority` as well.

**A second effect of the same guard:** it only fires when the verdict would
*change* an occurrence. Of the 1,978 contested occurrences, the current pipeline
rewrites **600** and **silently endorses 1,378** — leaving the OCR mark as it
stands and reporting nothing. A `wie¬der` that should read "wie der" is verdict
`OK`, invisible in the report, and joined to "wieder" on import. The stage's
workload is therefore `total`, not `affected`.

### Finding 3 — five wrong rows are in the store today, from `spread` groups

`check_hyphen_consistency.py` wrote 224 of the store's 226 `manual` rows. Five
contradict the corpus evidence by more than 5:1:

```
      so|wie      form=split   occ=221  affected=202   sowie 4,888  / so wie   48
      um|so       form=join    occ= 11  affected= 10   umso      2  / um so   145
   König|in       form=split   occ=  2  affected=  1   Königin 2,399 / König in 28
    sein|er       form=split   occ=  1  affected=  0   seiner 8,063 / sein er   4
Cocktail|parties  form=join    occ=  3  affected=  1
```

`so,wie` had been settled `join` by the LLM with a correct justification
("*'sowie' als Konjunktion im Sinne von 'und auch'*") and was overwritten the
same day with `consistency: so settled as split 4, join 4; this pair was join`.
Applying the store as it stands turns 202 correct `so¬wie` breaks into "so wie".
`König|in` is the `kaiser|in` case, already in the corpus and already set the
wrong way.

**The mechanism is not what it looks like, and the tool is not simply at
fault.** Reconstructing the pre-patch store, all five sit in `spread` groups:

```
kind=spread left='so'    majority=''       {split 4, join 4}
    so|dann join · so|gar join · so|wie join · so|wohl join
    so|bald split · so|lange split · so|viel split · so|weit split
kind=spread left='König' majority='split'  {split 2, hyphen 1}
    König|Sohn hyphen · König|Carols split · König|in split
```

and `write_patch()` in the current code drafts a `spread` row with an **empty**
form on purpose, for exactly this reason:

> *"`spread` has no such identity — the left is an ordinary word fragment and
> every right is a different word — and its majority proposes real damage."*

The `so` group has no majority at all (4–4), and the `König|in` row proposes the
form the pair already had, which `write_patch` skips. **Neither row is what the
current code produces**, so they predate that safeguard or were filled in by
hand from the report. The tool has since been fixed; the rows have not.

Two things follow. First, the `so` group shows why grouping by a short function
word finds nothing: `so|bald`, `so|lange`, `so|viel`, `so|weit` against
`so|dann`, `so|gar`, `so|wie`, `so|wohl` is not one decision made eight ways, it
is eight different words. Second, **`spread` groups over function-word lefts are
largely the contested population** — they should be promoted to per-occurrence
handling and excluded from consistency reporting entirely, rather than argued
about group by group.

### Finding 4 — a corpus bigram model separates most occurrences, and fails predictably

For each contested occurrence, take the token before `left` on the current line
and the token after `right` on the next, and score both readings under an
add-one bigram LM over the line-interior lexicon:

    margin = log P(prev → left+right) P(left+right → next)
           − log P(prev → left) P(left → right) P(right → next)

Over the 1,978 contested occurrences: `|margin| ≥ 2` covers 80.4 %, `≥ 3`
71.5 %, `≥ 4` 63.4 %. It is right where it is confident —

```
JOIN  d= +9.38  … de Chinot de Fromessent so¬ | wie Enkelin weiland …     ✓ sowie
JOIN  d=+12.85  … reiste Dienstag von Wien wie¬ | der nach dem Kriegs…    ✓ wieder
SPLIT d= −7.38  … PD. u. StkrD., Witwe nach | dem Grafen Thaddäus …       ✓ nach dem
```

— and unreliable near zero (`nach | dem Achilleion` at `d=+0.22`, wrong). But
two failure modes are **structural, not tunable, and both survive a margin
gate**:

**The corpus prior swamps a one-token window.** On `wie|der` it picks `join`
162 times of 163, because `wieder` outnumbers `wie der` 40:1. Two of twelve
inspected are real splits it got confidently wrong:

```
JOIN d=+7.75  … Derselben, wie | der vorangegangenen hl. Messe wohnte …   ✗ wie der
JOIN d=+5.68  … Täuschung der eigenen Bevölkerung wie | der Neutralen …   ✗ wie der
```

and `an | deren Bord sich bald darnach …` — a relative construction — was joined
at `d=+7.90`. A window of two or three tokens each side sees "wohnte … bei" and
"Bord sich"; a bigram cannot.

**Syntax the window cannot reach at all.** `Botschafter|in`, 39 occurrences,
*all* of them `split` ("Botschafter in Wien / in Paris / in St. Petersburg").
The model picks `join` 15 times, because with `prev = k.` and `next = St.` there
is no signal whatsoever. **This is the `kaiser|in` class — the one that
motivated the request — and the n-gram tier cannot do it.**

**Consequence for the design: do not gate on the margin.** At 1,978 occurrences
the n-gram tier buys nothing worth its failure modes — sending everything to the
LLM is ~1,978 calls, one evening on the existing infrastructure, against ~700
saved calls and two silent error classes. Keep the score, spend it as *evidence
in the prompt* and as a *disagreement flag for human review*, not as a gate.
(For reference, the margin disagrees with the stored pair verdict on 54 % of
contested occurrences. That is an upper bound on the disputed set, not an error
count — the scorer over-joins — but it is large enough to justify the stage.)

### Finding 5 — the occurrence key already exists, and the correcting script already has it

`correct_xml_hyphens.py` reads with **lxml**, walking `//tei:p` → `.//tei:l` and
looking up `plan.get((left, right))` per line (`correct_xml_hyphens.py:196-221`).
The element is in hand at that point, so `element.get('facs')` yields
`facs_1_tr_1784278140_tl_3` — unique within the file — and **the occurrence hook
is a one-line lookup with no reader change on the correction side.** Note that
its loop index runs over markup-free lines only, so a positional index would be
unreliable; the `facs` id is the correct key and the only correct key.

The *analysis* side does need a small additive change:
`validate_xml_hyphens.read_paragraphs()` discards `LINE_RE.group(1)`. A sibling
reader returning `(region_id, page, subtype, [(line_id, text), …])` keeps
`read_paragraphs` untouched — it is guarded by
`test_and_debug/test_xml_readers.py` and three scripts key their tables through
it.

Risk to record: the `tr_…` component is a Transkribus region id and may not
survive a re-export. The store must degrade gracefully — an unresolvable key is
reported and skipped, never guessed at.

### Finding 6 — the corpus supplies its own labelled data, and its own evaluation set

For the 61 contested pairs, both readings are attested ≥ 10 times in undamaged
line-interior text by construction of the gate:

```
wie|der       wieder 5,347  / wie der  131      nach|dem   nachdem 307 / nach dem 1,413
an|deren      anderen 834   / an deren  36      Botschafter|in  …in 113 / … in     388
außer|dem     außerdem 161  / außer dem 73      so|wie     sowie 4,888 / so wie      48
```

This is training data for the classifier of option 2, and — more immediately —
**a gold standard for the whole stage at zero annotation cost.** Hold out
interior occurrences, hand the scorer only their contexts, measure. The usual
blocker on a stage like this is that nobody wants to label 2,000 breaks by hand;
here nobody has to.

Two methodological cautions, both load-bearing:

- **Report balanced accuracy and per-class recall, never raw accuracy.** These
  classes are 40:1 and worse; a majority-class predictor scores ~95 % on
  `wie|der` while getting every interesting case wrong. Raw accuracy would
  certify precisely the failure mode of Finding 4.
- The minority class is thin for some pairs (`an|deren`: 36), which argues for
  one pooled model with the pair as a feature over 61 per-pair fits.

---

## Design

### Where the stage sits

```
statistics   validate_xml_hyphens.py     → per-pair verdict from frequencies
LLM          resolve_hyphens_llm.py      → per-pair verdict from 3 contexts
manual       hyphen_decisions.csv        → per-pair verdict by hand
NEW          resolve_break_context.py    → per-occurrence verdict from its own context
```

In the run order of the README's hyphen loop, after `validate_xml_hyphens.py`
(it needs the lexicon and the pair table) and before `correct_xml_hyphens.py`
(which consumes its output):

```
validate_xml_hyphens.py --export-review …      # contested pairs → queue
resolve_break_context.py --promote --score \   # NEW
                         --export-review --resolve
resolve_hyphens_llm.py …                       # the rest, still per pair
correct_xml_hyphens.py …
mark_xml_word_breaks.py …
```

Invariant to preserve: **a pair is handled at exactly one level.** A pair
declared contextual has no global form; a pair with a global form is never asked
per occurrence. Otherwise the two stores race and the outcome depends on which
script ran last.

### `hyphen_decisions.csv` gains one form: `context`

```
wie,der,context,manual,,2026-08-29,"both readings live; decided per occurrence"
```

Joins `join | hyphen | period | comma | split | skip` in
`hyphen_decisions.FORMS`. Semantics: *this pair has no global answer; consult
`hyphen_occurrences.csv`.* Deliberately unlike `skip` — `skip` means "leave every
occurrence alone", `context` means "every occurrence gets its own verdict".

Consumers:

- `validate_xml_hyphens.judge()` — a `context` pair contributes nothing to the
  report; the occurrence store owns it. The two `contested-both-readings` guards
  additionally move from their `>= 2` floor to `--min-minority` (Finding 2).
- `correct_xml_hyphens.py` — looks the line up by `(file, facs)` (Finding 5); a
  miss leaves the line untouched and is reported.
- `mark_xml_word_breaks.py` — reads the occurrence verdict for `@break`. On a
  miss it falls back to `hyphen_utils.determine_hyphen_action` with `cert='low'`,
  exactly as it does for a pair the statistics never had to decide.
- `check_hyphen_consistency.py` — **must skip `context` pairs entirely.** They
  are the pairs its `spread` class currently damages (Finding 3).

### `data/csv/hyphen_occurrences.csv` — the new store

Same conventions as its two siblings: plain CSV, spreadsheet-editable,
`manual > llm > rule` source rank, merge never downgrades, delete a row to have
it reconsidered.

```
file, line_facs, left, right, form, score, source, model, decided_on, note
```

- `file` / `line_facs` — the occurrence key; the break is the one ending
  `line_facs`.
- `left` / `right` — redundant with the key and kept anyway: it makes the file
  readable in a spreadsheet and lets a stale key be detected (pair at that line
  no longer matches → report, skip).
- `form` — `join | hyphen | period | comma | split`. No `context`, no `skip`;
  this is where the question stops.
- `score` — the bigram margin, always recorded whatever decided the row, since
  every verdict in this project carries its evidence.

### `lineends/resolve_break_context.py`

```
--promote            find contested pairs (--min-minority, default 10) and write
                     `context` rows into hyphen_decisions.csv
--score              write the bigram margin for every contested occurrence
--export-review      the occurrences as JSONL with a ±3-token context window
--resolve            run the local LLM over that queue (llm_backend.py,
                     [llm.break_context] in config.toml, prompt in
                     data/prompts/prompt_break_context.txt)
--evaluate           score against held-out line-interior occurrences,
                     reporting balanced accuracy and per-class recall
```

No new plumbing: `llm_backend.py` supplies the wire protocol, retry policy,
`parse_json_reply` and `load_prompt`; `--shard K/N`, periodic flush and
`utils/notify.py` are the established shape of a long local run.

### Phases

1. **Retire the five wrong rows** (Finding 3): `so,wie` → `join`, `um,so` →
   `split`, `König,in` → `join`, `sein,er` → `join`, and review
   `Cocktail,parties`. Add to `check_hyphen_consistency.py` a guard that refuses
   to draft any row contradicting the frequency evidence by more than `--ratio`.
   Ships alone, independent of everything below, and fixes 202 live corruptions.

2. **`context` as a form.** Add it to `FORMS`, teach the four consumers, move the
   two `contested-both-readings` floors to `--min-minority`. Ships with zero
   `context` rows, so the corpus diff must be empty — which is how it is
   verified.

3. **`--promote`.** Contested detection on the minority gate. Start at
   `--min-minority 20` (35 pairs, 1,687 occurrences) and widen to 10 once the
   loop runs end to end.

4. **Occurrence store + `--score`.** No gate: the margin is recorded as evidence
   on every occurrence, never as a verdict (Finding 4).

5. **`--resolve`.** The prompt asks what `resolve_hyphens_llm.py` asks — which
   reading does *this* sentence want — over one occurrence with a ±3-token
   window, offering only the candidate spellings, carrying the bigram margin as
   evidence rather than as an instruction. German, in `data/prompts/`, arguable
   as text, with the existing prohibition on modernising historical orthography
   carried over verbatim.

6. **`--evaluate`.** Held-out interior occurrences as a free gold standard.
   Balanced accuracy, per class, per pair. Run it **before** phase 5's output is
   applied to anything, and use it to decide whether the classifier of option 2
   is worth building at all.

Phases 1 and 2 are worth doing whatever happens to the rest.

---

## Risks and open questions

- **A wider context window is the cheapest accuracy win and was not measured.**
  The probe used one token each side because that is what the existing bigram
  tables hold. Trigrams over the same interior tokens, and ±3 tokens for the LLM,
  should both be measured in phase 6 before anything more elaborate is built.

- **The `¬` itself is unused as a signal.** It is the OCR's own opinion and is
  right 97.9 % of the time corpus-wide; `decide_form()` already treats it as a
  fallible prior. On contested pairs its reliability is unknown and unmeasurable
  without labels (the free interior labels have no break mark). Worth passing to
  the LLM as evidence; not worth trusting.

- **The store grows per occurrence, not per pair**, which breaks the
  "settled once, never asked again" economics the other two stores rest on. At
  2.5 contested occurrences per issue this is a non-issue in practice — a new
  issue costs ~3 LLM calls. It would stop being a non-issue if `--min-minority`
  were dropped to 2 (6.1 per issue, and mostly noise), which is the reason not
  to.

- **Occurrence keys are Transkribus region ids** and may not survive a re-export
  (Finding 5). Report unresolvable keys; never guess. A `--rekey` pass matching
  on `(pair, context)` is the fallback if it ever happens.

- **Historical orthography.** The bigram tier is immune — its lexicon *is* the
  corpus. The LLM tier is not, and the existing prompts' prohibition has to be
  carried over.

- **Region scope is now one shared decision**, not a per-script default:
  `CORRECTED_SUBTYPES` / `SKIPPED_SUBTYPES` in `validate_xml_hyphens.py`.
  Corrected: `paragraph`, `heading`, `heading-sub`, `image-caption`,
  `image-credit`, `ad-content`, `imprint`, `volume`, `footnote`. Skipped:
  `ad`, `ad-frame` (unusable OCR), `image`, `mode` (no text — stray marks
  read as characters), and the three `separator*` classes (printer's rules).
  Every measurement in this plan predates that alignment and used the older
  `ad,ad-frame` scope, so the figures include `image`, `mode` and separator
  regions and are slightly pessimistic; the populations move by well under a
  percent (those classes contribute ~1,500 lines between them) and no
  conclusion turns on it, but a rerun should be done before any threshold is
  fixed.

## Deliberately not done

- No sentence-embedding similarity scoring. It measures the wrong quantity and
  the two readings of a break are near-identical under it by construction.
- No margin gate (Finding 4). The n-gram score is evidence, not a verdict.
- No external German LM as a dependency in phases 1–6. If the LLM tier proves
  too slow, a masked-LM pseudo-log-likelihood over a German BERT is the next
  thing to try — but it is a new dependency on a model trained on modern German,
  and the free labels of Finding 6 should be spent measuring whether it beats
  the local instruction model before it is adopted.
- No change to `replacement.csv` or `ocr_corrections.csv`. This stage is about
  the break, not about misread words.


---

## What changed during implementation

### 1. Phase 1 was bigger, and two of its five "fixes" were wrong

The plan named five store rows to retire. Building the guard first and letting
it audit the whole store found **39 rows rewriting 547 breaks**, and the five
were not the interesting ones.

The dominant class is 28 `period` verdicts on abbreviated titles. This
magazine sets them closed up — `LegSekr.` occurs **969 times in the corpus and
`Leg. Sekr` zero times**, likewise `GenDir.`, `GenOberst.`, `KommRat.` — and
the model reached `period` by generalising the prompt's own example, saying so
in its note: *"Analog zum Beispiel „Geh. Rat" im Prompt"*. (`Geh. Rat` really
is written with the space; that is why the pair is contested and these are
not.) Those 28 are applied: **461 spurious full stops removed.**

Of the plan's five, three were right (`so|wie`, `um|so`, `Cocktail|parties` —
all three overrode a *correct* LLM verdict, and `um|so` would additionally have
modernised the period's own "um so mehr"). **Two would have caused
corruption:**

- `sein|er` — the plan said `join` on the frequency evidence (`seiner` 8,063
  against 4). The single occurrence is `zufrieden sein; | er war besser
  besucht`: a semicolon. `join` would have added a `¬` and produced "seiner".
- `König|in` — the plan said `join`. Its two occurrences disagree with each
  other: `Dragoner-Regimente König¬ | in Cannstadt` is plausibly the regiment
  "Königin", while `begab sich der König | in Begleitung des Herzogs` is
  certainly two words. It is a `context` pair, not a differently-global one.

Both were caught by reading the occurrences instead of the counts, which is the
same discipline this whole stage is about. The audit prints "a question, not a
verdict" for that reason, and the remaining 11 rows are left for review.

### 2. The context model was broken, and `--evaluate` caught it

The plan specified an add-one-smoothed bigram model. Add-one over a
161k-word vocabulary charges roughly **12 nats per token before any evidence is
considered**, and the separated reading has one token more — a fixed thumb on
the scale worth far more than any sentence. Measured against the free gold
standard:

    add-one            join recall 1.00, split recall 0.00
                       balanced accuracy 0.557
    majority baseline                    0.718

The model was not reading the sentence, it was paying a length penalty, and it
scored *below* answering the commoner reading every time. Interpolating the
bigram against a unigram backoff (`BIGRAM_WEIGHT = 0.99`) charges the extra
token its honest unigram cost instead:

    interpolated                         0.835

This is the phase-6 evaluation doing exactly the job it was put in for —
catching a design error before a single break was rewritten — and it is the
argument for building the evaluator *before* the tier it evaluates, not after.

It also repaired the failure the plan called structural. Measured over the same
2,585 contested breaks, with the same pair store:

    add-one        disagrees with the pair verdict on 1,472 (57%)
    interpolated   disagrees on 592 (23%)

    Botschafter|in   add-one       split 24, join 15
                     interpolated  split 39, join 0   (all 39 correct)

The class the plan named as beyond a one-token window — "with `prev = k.` and
`next = St.` there is no signal at all" — was not beyond it. It was buried
under the smoothing penalty.

### 3. A wider context window cannot help a bigram model at all

Listed in the plan as "probably the cheapest accuracy win". It is not a win of
any size: in a bigram model every term beyond the token adjacent to the break
is identical under both readings and cancels out of the margin. Measured over
the gold standard at windows of 1, 2 and 3 tokens, the balanced accuracy is
**0.835, 0.835, 0.835** — not close, identical. Using more context needs
trigrams, which is a different lexicon and a separate piece of work. The wider
window survives only where it can actually be read, in the LLM prompt.

### 4. `--min-minority` must not replace the `contested-both-readings` floors

The plan said to move `judge()`'s two `>= 2` guards onto `--min-minority`.
That is backwards. Raising the floor from 2 to 10 makes *fewer* pairs go to
`REVIEW`, so the weakly-contested band (minority 2-9) would lose its LLM
adjudication without gaining per-occurrence treatment, which only applies from
10 up. The floors stay at 2 and are now a named constant explaining why they
are deliberately not the promotion threshold.

### 5. The contested population is 95 pairs / 2,842 breaks, not 61 / 1,978

`contested_pairs` weighs `join_count + dash_count` against `separation`, where
the plan's measurement used `join_count` alone. Counting the dashed spelling
admits the compound-name class — `Franz|Josef`, `Marie|Therese`,
`Maria|Theresia`, `New|York` — which is genuinely per-occurrence ambiguous and
is the case `check_hyphen_consistency.py`'s own docstring cites ("Franz Josef"
the emperor's two names against "Franz-Josef" the order named after him). It is
3.6 breaks per issue rather than 2.5, and still an evening's work.

One caveat this creates: for a name pair the real question is `hyphen` against
`split`, and the margin's axis is joined-against-apart, which puts `hyphen` on
the joined side. The score is therefore less informative for that class. It is
evidence and not a verdict, so this is a note rather than a defect, but a
trigram model would want a three-way score.

### 6. `--promote` drafts rather than writes

`hyphen_decisions.merge()` replaces only on strictly higher source rank, so a
`manual` promotion of a pair a human had already settled by hand would have
been **dropped in silence** — and 22 of the contested pairs are exactly that.
`--promote` therefore drafts a patch CSV in the house style and `--apply`
appends it, which wins the tie in `load()`.

Appending also uncovered a live hazard: `hyphen_decisions.csv` had no trailing
newline, so appending to it joined two rows into one unparseable line. Fixed in
the file and guarded in `apply_promotion`.

### 7. Also built, beyond the plan

- `correct_xml_hyphens.py` now consumes the occurrence store, keyed by
  `(file, facs)`, and an occurrence verdict outranks the pair plan for any
  break it covers. Verified end to end: an occurrence row flips a break the
  pair plan had settled the other way.
- **Stale-key detection**, which the plan only listed as a risk. Every row is
  checked against the pair actually at that line, and a mismatch is reported
  and skipped rather than applied. Verified with a deliberately stale row:
  `expected Nicht|vorhanden, found Juana|Marquesa`.
- `check_hyphen_consistency.py --audit [--audit-csv]`, which is what found item
  1 and is worth re-running whenever the store is patched.

### Still open

- The LLM run (`--resolve`) needs a local server; the prompt and queue are
  ready and `--dry-run` is verified.
- The promotion is **drafted, not applied**. Applying it removes the corpus-wide
  verdict from 95 pairs, so until the occurrence store is filled those 2,842
  breaks would be left as the OCR has them. That trade is a decision, not a
  default.
- The 11 audit rows the frequency evidence still objects to, three of which
  (`König|in`, `Gemahl|in`, `so|wie`) are `context` candidates rather than rows
  to re-form.


---

## The paragraph-final gap (depends on the XML refactor)

Design only. `merge_xml_factoids.py` does not exist yet — see
an XML-layer refactor in the originating project — and nothing here is implemented.

### What is missing

Every break this stage judges is a break *between two lines of one `<p>`*. A
paragraph's last line has no following line inside its own element, so the
question "how should this pair be written?" cannot be asked at all:

    paragraphs in scope                 145,898
    in-paragraph breaks judged today    875,799
    paragraph-final line ends           145,898
      ... ending in a break mark          3,324   (2.3% of paragraph ends,
                                                   0.4% of all breaks)

**3,324 hyphens are currently undecidable**, not merely unreviewed. The
continuation is real — it is in the next column or on the next page — but no
script in `lineends` can see it, which is why the paragraph-merge step (downstream, not shipped)
joins them later and `utils/check_remaining_hyphens.py` reports whatever is
left over for manual correction.

### What closes it, and why it is not a second pass

`merge_xml_factoids.py` writes `@part` on every paragraph and `@next`/`@prev`
across fragments. It does not physically join anything — the `<p>` elements stay
where they are and gain a chain — and that is exactly what this stage needs.
**The fix is not "run again over the merged paragraphs". It is: when the
following line is not in this `<p>`, follow `@next` to the first line of the
fragment that continues it.**

That keeps the whole design intact. `break_pair()` still derives the key from
two line strings, the occurrence store is still keyed by the `facs` of the line
the break ends, and `correct_xml_hyphens.py` still rewrites one `<l>`. Only the
lookup of "the next line" changes, in one place, and `tei_annot.py` is planned
to own the chain parsing already.

The same move closes the same gap for `mark_xml_word_breaks.py`, which today
marks a paragraph-final break `unknown` precisely because "what continues the
paragraph lives in the next column or on the next page".

### Ordering (settled)

`xml_io.STEPS` currently runs both correctors **first**, ahead of
`set_xml_readingorder`. Reading `@next` requires them to run **after**
`merge_xml_factoids`, which itself runs after the reading order and after
`classify_xml_factoids`. So the chain becomes:

    correct_xml_ad_regions → set_xml_readingorder → classify_xml_headings
      → classify_xml_factoids → merge_xml_factoids
      → correct_xml_ocr → correct_xml_hyphens → mark_xml_word_breaks

**Settled: the correctors move to the end, and nothing is split.** The
reordering was worth checking rather than assuming, because a corrector running
after the steps that read its output is a dependency pointing backwards. It
turns out not to bite, for three separate reasons:

  * `classify_xml_factoids` classifies by **geometry** — region position,
    lookback, factoid type — not by matching text, so uncorrected text costs it
    nothing.
  * `classify_xml_headings` does match text: exact match after normalisation
    against `chapters.csv`, which is brittle to a character substitution in
    principle. In practice the headings are short, repeating chapter names, and
    `data/csv/manual_heading_classifications.csv` already exists as the layer
    that catches what the exact match misses. That remedy is in place whatever
    order the steps run in.
  * The `Fortsetzung`/`Schluß`/`Nachtrag` continuation headings that merging's
    second pass depends on are **few and already checked by hand** (Christian,
    2026-08-30). They do not need a correction pass in front of them.

So the chain above is the chain, `xml_io.STEPS` is reordered to match, and the
correctors keep their present shape — one `correct_xml_ocr`, one
`correct_xml_hyphens`, both at the end. Splitting the global `replacement.csv`
pass away from the break-dependent passes was the fallback if this had gone the
other way; it is not needed and should not be built.

### The marker, and what a run does without it

The check is the existing one, not a new mechanism:
`xml_io.read_marker_version(tree, ident)` against `merge_xml_factoids`'
`<application>` marker, the same way the reading order is already recorded.

Three properties worth fixing now, because they are cheap and easy to get
wrong later:

  * **Per file, not per run.** A corpus mid-migration is half marked, and a
    single check at startup would either abort a run that could have done 700
    files or bless 87 it should not have.
  * **A warning, never a failure.** Without the marker the stage does what it
    does today: judges the in-paragraph breaks and leaves the paragraph-final
    ones alone. That is the current behaviour, so an unmarked file is not
    broken, only incomplete.
  * **Count it in the report.** "3,324 breaks not judged because 87 files carry
    no merge marker" is a number someone can act on; a warning line that
    scrolls past is not. The absence of the marker should be as visible in the
    output as a verdict is.

### Status: the counting half is in, the rest is blocked

Implemented 2026-08-30, because it does not depend on the refactor and the
omission was invisible until it was:

    3,324 further break marks sit on a paragraph's last line and were
    not judged at all: the word continues in the next column or on the next
    page, which nothing in lineends can reach until `merge_xml_factoids.py`
    writes the `@next` chain.

`validate_xml_hyphens.py` and `correct_xml_hyphens.py` both print it. It is
kept **out of the verdict table and out of its total**, because it counts
breaks nobody looked at rather than breaks that came back clean, and folding
the two together is precisely how an omission comes to read as an absence of
errors.

One subtlety worth recording, since it is the sort of thing that quietly
under-reports: `analyse()` reads paragraphs with `min_lines=2`, so a *one-line*
paragraph ending in a break mark was invisible to the first version of this
count — 103 of the 3,324. The reader is now called with `min_lines=1` and the
one-line paragraphs are filtered out immediately afterwards, which counts them
without letting them into the lexicon, where they would have changed every
verdict. The verdict totals are byte-identical before and after.

Not implemented, and not implementable yet:

  * **following `@next`** — zero paragraphs in the corpus carry `@next`,
    `@prev` or `@part`, and `merge_xml_factoids.py` does not exist. There is
    nothing to follow.
  * **the marker check** — `merge_xml_factoids`' `<application>` ident has not
    been chosen. Writing the check against a guessed name would be inventing
    an interface for somebody else to match, which is worse than not having it.
    (The corpus currently carries no `<application>` markers at all, having
    been re-exported on 2026-08-29.)

### Knock-on

Once the chain is readable here, `utils/check_remaining_hyphens.py` is
reporting a leftover that no longer has to exist, and
the paragraph-merge step (downstream, not shipped) joins text whose breaks have already been
settled. Neither becomes wrong; both get quieter, and the note in
`hyphen_utils.py` about the rule being "the fallback behind the `@break` marks"
extends to the paragraph boundary as well.
