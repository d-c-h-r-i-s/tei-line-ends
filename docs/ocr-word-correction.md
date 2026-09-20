# Plan: Corpus-as-Dictionary OCR Word Correction

> **Status 2026-09-17: detector A implemented** as `lineends/validate_line_end_truncations.py` (in `lineends`, beside the other line-end scripts, rather than as half of a combined `validate_xml_ocr_words.py`). Full corpus: 697 line ends / 561 tokens against the ~715 estimated below; zero pair-level overlap with the hyphen validator's `FALSE_NEGATIVE` rows. Output is per line end rather than per type, `type_count` keeping the per-type view. `gap_px` was measured before use and runs the other way from a clipping region: hits end further from the region border (median 25 px against 17 px over 646,799 line ends), so it is reported and not used. **Check 3, 2026-09-18:** a 60-row sample of those 697 came back 38% truncated, against the 80% bar. Two rules fix it: only one letter missing (2–3 letters: 1 of 20 truncated) and a completion attested beside a neighbour (3% without one). Together 84% (21 of 25), keeping every right answer; the detector now reports 307 line ends / 239 tokens, offering completions of up to three letters but qualifying a token only on a one-letter one. Detector B is not built.

## Background

`0_validate/validate_xml_hyphens.py` settles line-break hyphens without any external word list, and
the question is whether the same trick — the corpus as its own dictionary — can find general OCR
errors. `data/csv/replacement.csv` (2,641 lines, 2,148 with both fields filled, 1,978 distinct
`old;new` pairs) is a growing hand-built list and is certainly incomplete. Two worries were raised
up front: a systematic error that occurs often enough
could overrule a rarer correct spelling, and running a local LLM over everything is too slow.
A third observation was that words at the end of a line sometimes lose a character or two,
suspected to be a Transkribus glitch on non-rectangular (L-shaped) paragraphs.

All three were measured on 150 of the 735 XML files before any design was fixed. What follows is
what the measurements say, then what to build.

### Finding 1 — the L-shape theory does not hold, but truncation is real

Region shape does **not** predict truncation. Restricted to non-final lines of `paragraph`/`ad`
regions, the rate of a truncated line-final word is:

| region shape | rate |
|---|---|
| rectangular | 0.737 % |
| polygon, line in the full-width part | 0.224 % |
| polygon, line inside the notch | 0.406 % |

The notch is no worse than a plain rectangle — if anything better. What *does* separate the classes,
by a factor of 16, is the **region subtype**:

| subtype | rate | est. corpus-wide |
|---|---|---|
| `ad` / `ad-frame` | 3.17 % / 3.44 % | ~2,300 occurrences |
| `paragraph` | 0.199 % | ~780 occurrences |
| `image-caption` | 0 % | — |

So the phenomenon exists and is worth catching; the L-shape is a red herring. Independent
confirmation from token position: the OOV rate is 4.95 % for line-interior tokens but 13.9 % for
line-final ones.

**`ad` and `ad-frame` are excluded from everything below** — from the lexicon as well as from the
scan. Their OCR is a mess, they are not needed downstream, and they drag every metric down: they
are 21.9 % of all lines but carry three quarters of detector A's raw hits. Dropping them lifts a
hand-judged precision sample from ~45 % to ~75–80 %, even with a lexicon built from only 150 files.

### Finding 2 — the hyphen trick generalises only where an unbiased channel exists

What makes `validate_xml_hyphens.py` safe is not the frequency counting, it is that the lexicon is
built from a position *the error cannot reach* — line-interior tokens cannot have been damaged by a
line break. Truncation has exactly that property (same error position, same immune reference), so
detector **A** below is a genuine sibling of the hyphen script. Arbitrary OCR errors have no such
immune position, which is precisely the worry that was raised, and no amount of counting fixes it.

### Finding 3 — but the manual list itself supplies a safe constraint

Aligning each distinct `old → new` pair gives 384 single-edit confusions of every kind (including
insertions and deletions); restricting to those where both sides are alphabetic leaves 232, of
which **106 occur in at least two distinct rules**. Restricting proposals to those edits, in the
direction the rules were written, is what makes detector **B** safe:

- Only rare types (count ≤ 2) are ever candidates for change; a frequent form is only ever a
  *target*. A systematic error occurring hundreds of times is therefore missed, never propagated.
- The mined direction is error → correct, and the confusions are dominated by diacritic restoration
  (`s→š` ×63, `r→ř` ×47, `l→ł` ×36, `a→á` ×34, `y→ý` ×29, `C→Č` ×23 …). Applying only that
  direction means a diacritic can only ever be *added*. This structurally prevents the feared
  failure: a hand-corrected `Podhajský` (rare) can never be reverted to `Podhajsky` (more
  frequent), because `ý→y` is not in the set.

Measured on 150 files with ads excluded: 375 candidate types / 450 occurrences (~2,200
corpus-wide), and the head of the list is almost all true — `wurdé→wurde`, `Kari→Karl`,
`Prinzessir→Prinzessin`, `Sohr→Sohn`, `Freifran→Freifrau`, `eincr→einer`, `Maiestät→Majestät`,
`wicder→wieder`. Generic Levenshtein-1 yields 13× more candidates but proposes `Gräf→Graf` and
`Graß→Graf` — destroying surnames, in a corpus that is mostly surnames. Generic edit distance is
deliberately **not** used.

Note on the counts: `replacement.csv` contains duplicate rows (2,148 rows → 1,978 distinct
`old;new` pairs), because entries were sometimes added twice by hand. A repeat count is **not** a
measure of importance, so the confusion set must be mined from *deduplicated* pairs — after which
"seen ≥ 2 times" means two *distinct words* exhibit the confusion, which is real evidence. In
practice this barely moves the set (109 confusions raw vs 106 deduplicated; only `ä→Ä`, `ê→č` and
`ó→ô` lose their second occurrence), but the threshold only means something after dedup.

### Finding 4 — the corpus is already corrected in place

`correct_xml_ocr.py` has been applied to `data/xml/`: `Grafer`, `Selte`, `Alsgre`, `Adorjan`,
`Sulkowska` return zero hits, their corrected forms 735/727/40/52. So a run measures the
*remaining* error rate, and every round of corrections makes the lexicon a better dictionary for
the next. It also means precision cannot be scored against `replacement.csv` on the current XML —
of 1,351 single-token rules, 1,301 are simply gone. The 50 that remain are instructive anyway:
25 have a correct form that is **never** attested in the corpus (`Hadik→Hádik`, `Agnes→Agnès`,
`Akos→Ákos`). Frequency cannot possibly find those; only a gazetteer such as
`adelsverzeichnis.sqlite` could, which is out of scope here.

### Finding 5 — the list is applied twice, and that changes where detector B must look

`replacement.csv` is applied at two stages: `0_validate/correct_xml_ocr.py` runs it per `<l>`
element, and `prepare_factoids.py` (downstream, not shipped) runs it again on the joined factoid text, because
line breaks prevent some replacements from firing the first time. Three consequences:

1. **Detector B must count over hyphen-joined text, not raw lines.** 37.3 % of the rare-type pool
   (16,541 of 44,328) never appears line-interior at all — those are hyphen halves and truncations,
   not lexical errors. Joining with `hyphen_utils.join_lines_with_hyphen_cleanup(lines)`
   before tokenising removes 1,661 rare types and drops detector B from 388 to 335 candidate types:
   **13.7 % of the raw-line proposals were fragments.** It also matches the text
   `prepare_factoids.py` actually sees, and prevents the two detectors from proposing contradictory
   fixes for the same truncated token.
2. **Space padding is safe precisely because of the second stage.** A padded rule `" Fran "` cannot
   match a token at the start or end of an `<l>` element, so it does nothing in stage 1 — but after
   the lines are joined it fires in stage 2. This is worth stating so nobody later "fixes" the
   padding by removing it.
3. **Every generated rule must be idempotent**, i.e. `old not in new`. Applying the list twice per
   pipeline run (and re-running `correct_xml_ocr.py` over already-corrected XML) means a rule whose
   output contains its own input compounds every time. This is not hypothetical — see below.

### Finding 6 — four existing rules are non-idempotent and have already corrupted the corpus

Four rows in `replacement.csv` have `old` as a substring of `new`:

| rule | effect |
|---|---|
| `Hofmeister` → `OHofmeister` | compounds |
| `Hofmarschall` → `OHofmarschall` | compounds |
| ` Liti` → ` Litić` | compounds |
| `Centralbade` → `»Centralbade«` | compounds |

The first two have already done damage across the whole corpus. Counting the exact `O`-run in
`data/xml/`:

```
139  OOOHofmeister      185  OOHofmarschall
 33  OOHofmeister         2  OOOHofmarschall
  0  OHofmeister          0  OHofmarschall     <- the intended form
```

**Not one correct occurrence of either title survives.** This is a live data bug independent of
this plan, and fixing it is a separate small task: repair the XML (`OO+Hofmeister` → `OHofmeister`,
same for `Hofmarschall`), then anchor the four rules so they cannot re-fire — pad them
(`" Hofmeister "` → `" OHofmeister "` still compounds, so padding alone is not enough; the rule has
to be dropped in favour of one whose output no longer contains the input, or `apply_replacements`
has to gain a guard). A cheap guard worth adding to `utils/ocr_text_corrections.load_replacement_dict`
regardless: warn on any rule where `old in new`.

While checking this, two more `replacement.csv` hygiene problems surfaced:

* **15 keys carry conflicting values.** `load_replacement_dict` builds `replacements[row[0]] =
  row[1]`, so the last row silently wins: `' ro. '` maps to `' 19. '` on line 36 and to `' 10. '`
  on line 1591; `Agypten` to `Ägypten` (line 105) and `Aegypten` (line 1570); `Adelaide`,
  `Chotélic`, `Kosir`, `Crme` and ten others likewise. Worth a warning in the loader.
* Any rule detector B generates must therefore also be checked against the existing keys, so a
  proposal never introduces a sixteenth conflict.

## What to build

One new validator, `scripts/0_validate/validate_xml_ocr_words.py`, with two independent detectors
over one shared lexicon. Reports and a review queue only — nothing writes to `data/xml/`.

It follows the conventions of `validate_xml_hyphens.py` throughout: date-prefixed CSV in `output/`,
`--max-files`, `--min-count`, `--ratio`, `--quiet`, a Transkribus deep link per row via
`utils/get_transkribus_link.py`, and `in_output()`-style placement of JSONL exports.

`--skip-subtypes` defaults to `ad,ad-frame` (the flag name is borrowed from
`correct_xml_hyphens.py`, where it is opt-in; here it is the default). The exclusion applies to the
lexicon and the scan alike, so ad vocabulary cannot mask an error elsewhere either.

### Reuse rather than reimplement

Import directly from `0_validate/validate_xml_hyphens.py`:

- `strip_token`, `WORD_RE`, `HYPHEN_CHARS`, `STRIP_CHARS`, `INNER_DASH_RE` — keeps the lookup keys
  identical to the hyphen tables, so Antiqua and Fraktur land under one key.
- `build_lexicon` — returns `(unigrams, bigrams, dashed, periods, commas)` from line-interior
  tokens. Detector A needs exactly this; do not build a second one. It takes an iterable of
  `(_, _, lines)`, so excluding ads is just a matter of filtering the paragraph list before passing
  it in — no change to the hyphen script is needed.
- `in_output`, and the `question_record` / `export_review_jsonl` shape.

New parsing is needed only for geometry and region subtype, which `read_paragraphs` drops. Add a
local reader that keeps `<l facs=...>` ids and joins them to the `<zone rendition='Line'>` and
`<zone rendition='TextRegion' subtype=...>` polygons in `<facsimile>`.
`validate_xml_split_lines.py` already parses those `points` polygons — follow its approach.

### Detector A — truncated line-final words

For every line that is **not** the last of its paragraph, take the raw final token and flag it when
all of the following hold:

1. The line does not end in `¬`/`-`/`=` — a marked continuation is the hyphen script's business.
2. The raw token ends in a letter. Punctuation is the single strongest filter in the whole
   detector: a cut-off word cannot end in `.`/`,`/`»`. It removes most false alarms (`Frank,`,
   `überreicht.`, `Tir.`, `geg.`).
3. The stripped token is alphabetic and ≥ 3 characters, with unigram count ≤ 1 — it must be
   essentially unknown to the interior lexicon.
4. Some extension of it by 1–3 letters is a frequent word (≥ 10). Build this as one prefix index
   over the lexicon, not per-token generation.
5. **Not** already explained by the next line, i.e. it is not a dropped hyphen. Test
   `unigrams[left + right]` for the next line's first token *and* for its leading alphabetic run —
   the plain version leaks `Hard | egg-Gudenus` and `Öster | reich-Este` into the report.

Keep `region_subtype` as a CSV column — it costs nothing and makes the exclusion auditable.

Emit the **top three completions with their frequencies**, not a single answer: detection is often
right where the completion is not (`ers` → `ersten`/`erster`/`erst`; `Augus` →
`August`/`Auguste`/`Augustin`). The review record carries the alternatives and lets the adjudicator
pick.

Expected scale and precision, ads excluded: 146 occurrences / 131 distinct types in 150 files, i.e.
**~715 occurrences corpus-wide**, 0.184 % of eligible line ends. On a 20-row random sample ~15–16
were genuine (~75–80 %). The two residual failure modes both shrink once the lexicon is built from
all 735 files rather than 150: real inflected forms that merely look OOV in a small sample
(`Künstlerhaus`, `geleistete`), and compound/hyphen leakage (`Filial | Reservespitale`), which is
what rule 5 is for — the strict version costs only 4 of 150 hits.

### Detector B — confusion-constrained word fixes

1. **Deduplicate** `replacement.csv` to distinct `old;new` pairs, then mine the confusion set: align
   each pair with `difflib.SequenceMatcher`, keep pairs whose alignment is a *single* edit of ≤ 2
   characters on each side, and keep alphabetic confusions seen ≥ 2 times (106 of them). After
   dedup the threshold means "two distinct words show this confusion", which is the property worth
   filtering on. Cache it; it is cheap.
2. Build the frequency table from **hyphen-joined paragraph text** via
   `hyphen_utils.join_lines_with_hyphen_cleanup`, not from raw `<l>` text (Finding 5).
   Count all positions — the error is not positional here — but only over non-ad regions.
3. For every word type of length ≥ 4 with count ≤ 2, generate variants by applying each confusion
   at every position.
4. Keep a variant when it is a corpus word with count ≥ 10 **and** ≥ 10× the rare form's count.
   Rank by target frequency. Ads excluded and text joined, this leaves 335 types / 404 occurrences
   in 150 files (~2,000 corpus-wide) out of 42,667 rare types.

Four guards belong in the code, not just in the report. The last three all follow from the list
being applied twice per run:

- Never propose a change to a token that appears as the *target* (right-hand column) of any
  `replacement.csv` row. That makes every existing manual correction immune by construction.
- **Reject any proposal where `old in new`** — it would compound on every application, which is
  exactly how `OHofmeister` became `OOOHofmeister` (Finding 6).
- **Reject any proposal whose `old` already exists as a key** in `replacement.csv`, so no new
  conflicting duplicate is created.
- Emit the pair only once even if the same confusion is found on several types.

The `find`/`replace` columns must be written **with surrounding spaces** (`" Fran "` → `" Frau "`).
`utils/ocr_text_corrections.apply_replacements` is a plain substring `str.replace`, so an unbounded
rule for a short token would corrupt longer words (`Fran` inside `Franz`). This is why the existing
file has entries like `" Frein ; Freiin "`, and it costs nothing at the line-edge blind spot
because stage 2 catches those once the lines are joined. Rows accepted after review can then be
appended to `replacement.csv` and are picked up by both existing stages — no new correction script
is needed for this class.

Detector A's output, by contrast, is **not** appendable to `replacement.csv`: a truncation is
position-specific (`Augus` is only wrong at that one line end), so a global substring rule is the
wrong instrument. Its fixes belong at the line level in the XML, which is out of scope here.

### Outputs

- `output/YYYYMMDD_ocr_truncations.csv` — `region_subtype`, `left`, `completions`, `affected`,
  `left_freq`, `best_freq`, `gap_px`, `example_context`, `next_line`, `example_file`,
  `example_page`, `transkribus_link`.
- `output/YYYYMMDD_ocr_confusions.csv` — `find`, `replace`, `rare_form`, `rare_freq`,
  `target_form`, `target_freq`, `confusion`, `affected`, `example_context`, `example_file`,
  `example_page`, `transkribus_link`.
- `--export-review NAME` — one JSONL record per distinct type, in the same shape as the hyphen
  queue (`candidates`, `evidence`, `contexts`), so `resolve_hyphens_llm.py` can later be pointed at
  it with a new prompt and a second decision store. Deduplication by type is what keeps an LLM pass
  affordable — a few hundred questions rather than 735 files, exactly as the hyphen loop already
  works.

### README

Add both detectors to the `0_validate` table in `README.md`, in the style of the existing
`validate_xml_hyphens.py` entry, and record the L-shape result as a negative finding — the same way
the line-geometry note is recorded there — so it does not get re-investigated.

## Verification

```bash
cd /Users/christian/Salonblatt-Codebase/salonblatt-workflow

# fast smoke test
uv run scripts/0_validate/validate_xml_ocr_words.py --max-files 20

# full run + review queue
uv run scripts/0_validate/validate_xml_ocr_words.py --export-review ocr_review.jsonl
```

Checks that decide whether this is trustworthy:

1. **Nothing is written.** `git status data/xml/` stays clean.
2. **No ad rows.** `region_subtype` is `ad`/`ad-frame` on zero rows of either CSV.
3. **Detector A precision.** Take 30 random rows and judge them by hand. It must beat the ~75–80 %
   measured on a 20-row sample from 150 files; the full-corpus lexicon should push it higher, and
   > 80 % is the bar. If it does not clear it, rule 5 (hyphen/compound leakage) is the first thing
   to tighten.
4. **Detector A scale** lands near ~715 occurrences / ~640 types. An order of magnitude more means
   the OOV threshold or the ≥ 10 extension threshold is mistuned.
5. **The L-shape result still holds** on the full corpus: split detector A's rows by `rect` vs
   polygon region and confirm the rates do not separate. If they suddenly *do*, the 150-file sample
   was unrepresentative and the original idea deserves another look.
6. **Detector B safety.** Assert in a test that no proposed `replace` value equals a rare form whose
   frequent counterpart differs only by a *removed* diacritic, and that no proposal targets a token
   appearing in the right-hand column of `replacement.csv`. Spot-check that `Gräf`, `Graß`, `Bien`,
   `Sien` are absent from the output — they are what generic edit distance would have produced.
7. **Detector B idempotence and collisions.** Assert `old not in new` for every generated rule, and
   that no generated `old` already exists as a key in `replacement.csv`. Then apply the whole
   augmented list twice to a sample of text and confirm the second pass changes nothing — the
   regression test the `OHofmeister` bug should have had.
8. **Detector B runs on joined text.** Confirm no proposal has a hyphen half as its `old`; the
   count should land near 335 types rather than the 388 raw-line version.
9. **Cross-check against the hyphen report.** Run `validate_xml_hyphens.py` on the same corpus and
   confirm detector A's rows do not overlap its `FALSE_NEGATIVE` rows; any overlap is rule-5
   leakage.
10. **Confusion mining.** `Sohr → Sohn`, `Prinzessir → Prinzessin`, `Freifran → Freifrau`,
    `Maiestät → Majestät` should all appear, each `find`/`replace` space-padded, and the mined set
    must be identical whether or not the duplicate rows are present.

## Explicitly out of scope

- **Repairing the `OOOHofmeister` / `OOHofmarschall` corruption and hardening `replacement.csv`**
  (Finding 6). Independent of this plan, and worth doing first — it is a handful of `sed`-scale
  edits in `data/xml/` plus a warning in `load_replacement_dict` for `old in new` and for duplicate
  keys with conflicting values. Listed here so it is not forgotten, not because it is optional.
- Writing corrections into `data/xml/` — reports and a review queue only.
- The LLM resolver and a decision store for these verdicts. The JSONL export is shaped so that
  `resolve_hyphens_llm.py` + `hyphen_decisions.py` can be generalised later without rework.
- Diacritic restoration on proper names against `adelsverzeichnis.sqlite` (4,494 families). This is
  the right tool for the 25 rules whose correct form is never attested, and
  `data_analysis/check_noble_families.py` already does the Aho-Corasick matching with umlaut and
  ASCII folding — but it is a gazetteer problem, not a corpus-statistics one.
- A geometric whole-phrase text-loss report (mid-paragraph lines stopping far short of their
  region's right edge, ~2,600 corpus-wide). Measured and real, but a separate concern from
  word-level correction.
- `ad` and `ad-frame` regions, entirely. If they are ever wanted, `--skip-subtypes ''` reopens them
  — but expect roughly three times the rows at roughly half the precision.
