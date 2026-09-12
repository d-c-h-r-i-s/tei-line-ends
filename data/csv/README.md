# `data/csv/` — the decision stores

Plain CSVs, on purpose. Every one of them can be opened in a spreadsheet,
sorted, hand-edited and committed, and that is the point: a machine proposal
and a human verdict live in the same file, distinguished by a column rather
than by which tool wrote them.

The files here are **examples with a couple of rows each**, so the format is
visible and the scripts run on a fresh clone. They are not the Wiener
Salonblatt project's own stores — those are decisions about one specific
corpus, and applying them to yours would corrupt it. Yours grow as you run the
detectors.

| file | one row is | written by | applied by |
|---|---|---|---|
| `replacement.csv` | a string that is wrong **wherever** it occurs | by hand | `correct_xml_ocr.py` |
| `ocr_corrections.csv` | a word that is wrong **at a line break** | `validate_line_end_chars.py`, `resolve_line_end_llm.py`, `extract_ocr_corrections.py` | `correct_xml_ocr.py` |
| `hyphen_decisions.csv` | how a word **pair** at a break should be written | `resolve_hyphens_llm.py` | `correct_xml_hyphens.py` |
| `hyphen_occurrences.csv` | how **one specific break** should be written | `resolve_break_context.py` | `correct_xml_hyphens.py` |
| `transkribus_filenames.csv` | issue → Transkribus document id | by hand — see [../../docs/transkribus-links.md](../../docs/transkribus-links.md) | every validator, for links |

A store the scripts create themselves does not have to exist beforehand; a
missing one simply means no decisions yet.

## `replacement.csv`

Semicolon-separated, no header, no quoting — every field is a literal string,
and `"` is one of the characters being corrected.

```csv
 Selte ; Seite 
osterr;österr
```

**The surrounding spaces are part of the rule** and are what keeps `Selte` from
matching inside a longer word. Rules are applied in file order, each over the
previous one's output, which is what lets diacritics be restored in stages
(`Lancut` → `Lańcut` → `Łańcut`).

Never put a *positional* error here. `Wier` is a misreading of `Wien` at a line
end and a fine token in the middle of a line; a global rule would corrupt every
innocent occurrence. Those belong in `ocr_corrections.csv`, which is keyed by
position.

## Ranking, in every store that a model writes to

```
manual  >  llm  >  rule
```

An incoming row replaces an existing one only if it ranks strictly higher, so
re-running a detector cannot undo a decision a person made. To genuinely
revisit a case, delete its row or set `source` to `manual`. In
`ocr_corrections.csv` a row additionally has to clear a confidence bar to
apply at all, and `approved=no` retires a wrong suggestion for good without
deleting it, so the next run does not propose it again.
