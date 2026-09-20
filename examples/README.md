# `examples/` — five issues, raw, and the decisions made about the full corpus

Everything here is from the **Wiener Salonblatt** corpus project and is released
under **CC BY 4.0** (see [LICENSE](LICENSE)). The code in this repository is MIT;
this directory is data, and it has its own licence.

Nothing here is a default. The scripts read `data/csv/`, which holds two-row
examples; point them at this directory explicitly, as every command below does.
Salonblatt's verdicts are about Salonblatt's text, and applying them to another
corpus would corrupt it.

---

## `xml/` — five issues, straight out of Transkribus

```
ONB_wsb_19140926.xml   ONB_wsb_19190726.xml   ONB_wsb_19260905.xml
ONB_wsb_19160304.xml   ONB_wsb_19230113.xml
```

**Completely unprocessed.** No OCR correction, no hyphen resolution, no word
break marks, no reading order — the TEI as Transkribus exported it, with one
exception: the `<bibl>` creator statement in each `<teiHeader>` read
`TRP document creator: <email address>`, which Transkribus fills in from the
account, and now names the author instead. Nothing below the header was
touched. That is what
makes them useful: you can run the whole chain and watch it change the text,
which you cannot do against a corpus that has already been corrected.

Chosen to be small and to span the range. The corpus splits in two by file size:
issues up to about 1926 store region polygons as a few corner points (~0.3 MB an
issue), while later exports store them at per-pixel granularity (~10 MB an
issue, almost all of it coordinates). The TEI is otherwise identical, and none
of these scripts read `@points`, so the early issues make the same demonstration
at a thirtieth of the size.

## `csv/` — the decision stores for all 787 issues

| file | rows | what it is |
|---|---|---|
| `hyphen_occurrences.csv` | 2,842 breaks over **95 pairs** | one verdict per *occurrence*, for the pairs no corpus-wide answer fits — about 30 judgements per word |
| `hyphen_decisions.csv` | 1,450 pairs | one verdict per word pair; 58 of them `form=context`, the store declining to answer and pointing at the file above |
| `ocr_corrections.csv` | 1,320 tokens | line-final character verdicts; **695 are "leave it alone"**, the model declining to change what the statistics proposed |
| `replacement.csv` | 2,897 rules | hand-maintained global substitutions. The one file here that is not machine output, and the one most likely to be useful to another German-language newspaper project |
| `transkribus_filenames.csv` | 5 | document ids for the five issues above, so the deep links work |

## `ground-truth/` — what the gates were measured on

| file | rows | |
|---|---|---|
| `20260830_line_end_sample.csv` | 271 | stratified over margin, token band and has-next |
| `20260904_line_end_sample_gated.csv` | 300 | drawn after the gates, to measure them rather than trust them |

Both are **fully labelled by hand**, verdict in the first column. Every
precision figure quoted in [docs/](../docs/) traces to these two files — the
three failure populations, the 17.0% → 36.0% the gates buy, the channel's
0.644 → 0.791 AUC. They are here so the claims can be re-measured rather than
taken on trust.

---

## Run the chain

```bash
# what the corpus says about itself - no model, no corrections written
python lineends/validate_xml_hyphens.py       --xml-folder examples/xml
python lineends/validate_line_end_chars.py    --xml-folder examples/xml
python lineends/validate_line_end_truncations.py --xml-folder examples/xml
python lineends/resolve_line_end_context.py   --xml-folder examples/xml --score

# apply the corpus's own decisions (copy the XML first - these rewrite in place)
cp -r examples/xml /tmp/wsb && python lineends/correct_xml_ocr.py \
    --xml-folder /tmp/wsb \
    --csv examples/csv/replacement.csv \
    --corrections examples/csv/ocr_corrections.csv --dry-run

python lineends/correct_xml_hyphens.py \
    --xml-folder /tmp/wsb \
    --decisions examples/csv/hyphen_decisions.csv \
    --occurrences examples/csv/hyphen_occurrences.csv --dry-run
```

Measured on these five issues, September 2026:

```
correct_xml_ocr.py        5/5 files changed, 138 <l> elements corrected,
                          10 text lines removed from inside image regions
correct_xml_hyphens.py    5/5 files touched, 18 lines corrected (9 join, 9 split)

validate_xml_hyphens.py   3,992 breaks: 7 FALSE_POSITIVE, 6 FALSE_NEGATIVE, 6 REVIEW
validate_line_end_chars.py  3,716 line ends: 4 MISREAD
resolve_line_end_context.py 678 ambiguous line ends, 34 proposals at margin >= 6
```

## Why the detectors find less here than the docs report

**The method builds its dictionary from the corpus**, out of tokens in the
interior of a line where no line break can have damaged them. Five issues give
about 22,000 interior tokens; the full corpus gives roughly four million. So
`--min-count 20` — a candidate spelling has to be attested twenty times before
it counts as a reading at all — is met far less often here, and the detectors
are correspondingly quiet. `validate_line_end_truncations.py` finds nothing at
all on five issues.

That is the method working as designed, not a defect, and it is the honest
limitation to know before pointing these tools at a small collection: **they
need a corpus to have an opinion.** The stores in `csv/` are what the same code
produced over all 787 issues.
