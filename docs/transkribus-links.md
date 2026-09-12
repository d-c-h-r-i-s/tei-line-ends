# Transkribus deep links

Every validator in this repo writes a CSV where one row is one finding. A
finding you cannot look at is nearly useless — the question is almost always
"what does the scan actually say there?" — so each row carries a link that opens
the page in Transkribus at the right page number.

This is optional. With nothing configured the link column is empty and
everything else works exactly as before.

## What you need

Two things, because a Transkribus URL is built from two pieces:

```
https://beta.transkribus.org/collection/XXXXXX/doc/XXXXXXX/edit?pageNr=11
                                        ^^^^^^     ^^^^^^^           ^^
                                        collection document          page
```

1. **A collection id** — one number for your whole collection.
2. **A document id per issue** — one number per file, which is why it needs a
   lookup table.

## 1. The collection id

Open the collection in the Transkribus web app and read the number out of the
URL. It identifies *your* collection, so it does not belong in a committed
file:

```bash
cp .secret.env.example .secret.env
# then edit:
export TRANSKRIBUS_COLLECTION_ID=XXXXXX
```

`.secret.env` is gitignored. It is read from the environment first and parsed
from the file otherwise, so `uv run ...` in a shell that never sourced it still
works.

If your collection is public and the id is not worth hiding, put it in
`config.toml` instead:

```toml
[transkribus]
collection_id = "XXXXXX"
```

## 2. `data/csv/transkribus_filenames.csv`

One row per issue, semicolon-separated, with a header:

```csv
ISSUE_DATE;TRANSKRIBUS_ID;
19190501;XXXXXXX;
19190508;XXXXXXX;
```

| column | what it is |
|---|---|
| `ISSUE_DATE` | the key. Whatever the filename pattern below pulls out of an XML filename — a date in the original project, but any stable per-file id works |
| `TRANSKRIBUS_ID` | the document id, the second number in the URL |

The trailing semicolon is the original file's and is ignored; a third column
would be too, so you can keep notes there.

**Getting the ids out of Transkribus.** The web app shows one per document; for
a collection of any size, ask the API instead:

```bash
curl -u "USER:PASSWORD" \
  "https://transkribus.eu/TrpServer/rest/collections/XXXXXX/list" \
  | python3 -c "import json,sys; [print(f\"{d['title']};{d['docId']};\") for d in json.load(sys.stdin)]"
```

Then map each title to the date your filenames use.

## 3. The filename pattern

The scripts know an XML file, not an issue date, so one regex says how to get
from one to the other — `[transkribus] filename_pattern` in `config.toml`, with
exactly one capture group:

```toml
filename_pattern = '(\d{8})'      # ONB_wsb_19190501.xml -> 19190501
```

Anything a `re` pattern can express works. If your files are named by document
id already, capture that and make it the first column of the CSV; nothing in
the lookup requires the key to be a date.

## When a file is missing from the table

It warns once per issue, on stderr, and leaves the link empty — a gap in the
table, not a reason for a run over 787 files to stop. `links_available()` says
up front whether a collection id and a table are both present, for a caller
that would rather leave the column out than collect warnings.
