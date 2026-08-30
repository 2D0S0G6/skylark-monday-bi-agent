# Skylark Drones — Monday.com Business Intelligence Agent

A conversational BI agent that answers founder-level business questions from two
**live Monday.com boards** — *Deals* and *Work Orders*.

The supplied Excel files are seed data used only to set the boards up. At runtime
the application queries Monday.com over GraphQL; change a value in Monday and the
next answer reflects it.

**Contents** — [Approach](#approach) · [Architecture](#architecture) ·
[Features](#features) · [Setup](#setup) · [Monday.com setup](#mondaycom-setup) ·
[Deployment](#deployment-streamlit-community-cloud) ·
[Example questions](#example-questions) · [Testing](#testing) ·
[Assumptions](#assumptions) · [Trade-offs](#trade-offs) ·
[AI tools used](#ai-tools-used) · [Challenges faced](#challenges-faced) ·
[Design decisions](#design-decisions) · [Limitations](#limitations) ·
[Potential improvements](#potential-improvements)

---

## Overview

Ask a question in plain English — *"How's our pipeline looking this quarter?"*,
*"Which deals are at risk?"*, *"Compare pipeline vs operational workload"* — and the
agent will:

1. use **Groq** to turn the question into a structured query plan,
2. fetch the **current** board data from the Monday.com GraphQL API,
3. **normalise** messy values (sector casing, `N/A` placeholders, `₹2.5 Cr`, five
   date formats, duplicate rows, header rows pasted into the data),
4. run **deterministic pandas analytics** — every number comes from Python,
5. **validate** the computed figures for internal consistency,
6. use **Groq** to explain what the numbers mean in executive language,
7. attach only the **data-quality caveats that matter** to that question.

The split is deliberate: the LLM does language, Python does arithmetic. The model
is never asked to add, average or percentage anything.

---

## Approach

The brief is a data problem wearing a chatbot's clothes. The hard part is not
producing fluent text — it is producing a number a founder can act on, from data
that is roughly half-empty and inconsistently formatted, and being honest about
the half that is missing.

So the guiding principle was a hard split of responsibilities:

| The LLM does | Python does |
|---|---|
| understand the question | fetch from Monday.com |
| classify intent, extract sector/period/filters | normalise messy values |
| explain what the numbers mean | filter, join, aggregate |
| write the executive summary | every sum, count, percentage, share, rank |
| | validate the results |
| | detect and classify data-quality issues |

An LLM asked to total 18 deal values will sometimes get it wrong, and — worse —
will do so with no error signal. Every figure in this application is therefore
computed in pandas. The model receives a JSON block of **finished** facts, with
currency pre-formatted (`{"amount": 124000000, "display": "₹12.40 Cr"}`) and a
prompt instruction to quote the `display` string verbatim, removing any
opportunity to re-derive or mis-round it.

I worked in this order, which is also the order of the pipeline:

1. **Profiled the real seed files first** (`tools/inspect_seed_data.py`) before
   writing a line of parsing logic. That surfaced the header rows pasted into the
   data, the 52 % missing deal values, the categorical probability field and the
   second file's offset header — none of which a guessed schema would have caught.
2. **Built the parsers bottom-up** (`utils/`) with tests, so currency and date
   handling was proven before anything depended on it.
3. **Built the Monday client** against a mocked GraphQL endpoint that reproduces
   the real board's messiness, so the whole stack was testable offline.
4. **Built normalisation and analytics**, keeping data quality as a first-class
   output rather than an afterthought.
5. **Added the LLM layer last**, so the application was already correct and
   useful before any model was involved — which is why it still works without one.
6. **Ran it against the real boards**, which surfaced four genuine defects that no
   amount of mocking would have (see *Challenges faced*).

---

## Architecture

```
                    ┌──────────────────────────────────────────────┐
  Founder ────────► │  Streamlit chat UI  (app.py)                 │
                    │  history · examples · metrics · details      │
                    └───────────────┬──────────────────────────────┘
                                    │ question + recent turns
                    ┌───────────────▼──────────────────────────────┐
                    │  Groq query planner  (agent/planner.py)      │
                    │  NL → QueryPlan JSON, validated by Pydantic  │
                    │  keyword fallback if Groq is down            │
                    └───────────────┬──────────────────────────────┘
                                    │ intent · boards · sector · period · filters
                    ┌───────────────▼──────────────────────────────┐
                    │  Monday.com GraphQL  (monday/client.py)      │
                    │  column discovery · cursor pagination        │
                    │  retries · typed errors · TTL cache          │
                    └───────────────┬──────────────────────────────┘
                                    │ raw board rows
                    ┌───────────────▼──────────────────────────────┐
                    │  Normalisation  (analytics/normalization.py) │
                    │  text · dates · currency · missingness       │
                    │  → clean DataFrame + DataQualityReport       │
                    └───────────────┬──────────────────────────────┘
                                    │
                    ┌───────────────▼──────────────────────────────┐
                    │  Deterministic analytics (pandas)            │
                    │  deals.py · work_orders.py · cross_board.py  │
                    └───────────────┬──────────────────────────────┘
                                    │ computed facts (JSON)
                    ┌───────────────▼──────────────────────────────┐
                    │  Validation  (orchestrator.validate_facts)   │
                    └───────────────┬──────────────────────────────┘
                                    │
                    ┌───────────────▼──────────────────────────────┐
                    │  Groq narrator  (agent/response.py)          │
                    │  explains the facts · quotes them verbatim   │
                    │  deterministic markdown renderer as fallback │
                    └───────────────┬──────────────────────────────┘
                                    ▼
                              Executive answer
```

### Project structure

```
.
├── app.py                      # Streamlit UI
├── config.py                   # env / secrets configuration
├── requirements.txt
├── requirements-dev.txt
├── .env.example
├── DECISION_LOG.md
│
├── agent/
│   ├── planner.py              # NL → QueryPlan (Groq + keyword fallback)
│   ├── prompts.py              # planner / narrator / leadership prompts
│   ├── response.py             # narration + deterministic renderer
│   ├── llm.py                  # Groq wrapper, JSON extraction, typed errors
│   ├── data_service.py         # fetch + normalise + TTL cache
│   ├── orchestrator.py         # end-to-end flow, routing, validation
│   └── schemas.py              # QueryPlan (Pydantic)
│
├── monday/
│   ├── client.py               # GraphQL client, pagination, error mapping
│   ├── column_map.py           # canonical field → real board column resolver
│   └── schemas.py              # BoardColumn / BoardItem / BoardSnapshot
│
├── analytics/
│   ├── normalization.py        # messy rows → clean typed DataFrames
│   ├── deals.py                # pipeline / revenue / risk metrics
│   ├── work_orders.py          # operational metrics
│   ├── cross_board.py          # sector-level sales vs delivery comparison
│   └── quality.py              # DataQualityReport
│
├── utils/
│   ├── dates.py                # multi-format parsing, fiscal quarters
│   ├── numbers.py              # ₹ / Cr / Lakh / comma parsing, INR formatting
│   ├── text.py                 # sector / status / stage canonicalisation
│   └── logging.py              # logging with secret redaction
│
├── tools/
│   ├── inspect_seed_data.py    # profile the seed spreadsheets (dev only)
│   ├── list_models.py          # list Groq models available to this account
│   └── verify_connection.py    # end-to-end config smoke test
│
├── seed_data/                  # the supplied Excel files (setup input only)
└── tests/                      # 238 tests, fully mocked API
```

---

## Features

**Query understanding**
- Groq converts questions into a validated `QueryPlan` (intent, boards, sector,
  period, status filter, grouping).
- Thirteen intents including `leadership_update`, `cross_board_analysis`,
  `data_quality` and `greeting`.
- Clarifying questions **only** when genuinely ambiguous (*"How are we doing?"*),
  never when the user named a sector, board, metric or period.
- Greetings and *"what can you do?"* get a warm orientation with suggested
  questions — not a refusal — and cost neither an LLM call nor a board fetch.
- Genuinely unrelated questions are declined politely, with a pointer back to what
  the agent *can* answer. The keyword fallback declines them too, so an unrelated
  question never returns a business summary just because Groq was unavailable.
- Conversational memory: *"How is energy doing?"* → *"What about infrastructure?"*
  keeps the previous intent and period.
- A keyword planner takes over transparently if Groq is unavailable.

**Monday.com integration**
- GraphQL only; no scraping.
- Column IDs are **discovered at runtime** and matched to canonical fields by an
  alias/fuzzy/type-aware resolver — a renamed column still maps.
- Cursor pagination over `items_page` / `next_items_page`.
- Retries with backoff; typed errors for auth, missing board, rate limit, outage.
- Short-lived TTL cache with a visible fetch timestamp and a **Refresh data** button.

**Data resilience**
- Missing values (`""`, `None`, `NaN`, `N/A`, `NA`, `-`, `Unknown`, `Not Available`,
  `TBD`) are tracked as missing, **never coerced to zero**.
- Currency parsing: `₹25 Lakhs`, `₹2.5 Cr`, `25 L`, `2,500,000`, `1,50,00,000`,
  `(1,200)`, `$100000`. Non-INR amounts are parsed but **excluded from INR totals**
  and reported — no exchange rate is invented.
- Date parsing across `12/08/2026`, `2026-08-12`, `12-Aug-26`, `Aug 12, 2026`,
  Excel serials and more; invalid dates are counted, never fatal.
- Sector/status/stage canonicalisation with alias tables plus fuzzy matching for
  typos; **unknown sectors are preserved**, not dumped into "Others".
- Header rows pasted into the data are detected and removed.
- Duplicates are flagged and reported but retained (dropping them would understate
  the pipeline).
- Original values are always kept alongside the normalised ones.

**Deal analytics** — total / open / won / lost value, deal counts, pipeline by
sector, stage group, owner and product, largest opportunities with concentration,
late-stage pipeline, expected-close distribution by quarter, probability-weighted
pipeline, win rate by count and value (suppressed when the closed sample is too
small), and deals at risk with explicit per-deal risk signals.

**Work-order analytics** — total / active / completed / blocked / delayed counts,
delay measured only against orders that *can* be assessed, days overdue, work
orders by sector, status, owner and nature of work, order-book value, billing vs
collection, and upcoming completions.

**Cross-board analytics** — sector-level comparison of open pipeline share against
active delivery-workload share, capacity signals (`pipeline_ahead_of_delivery`,
`delivery_ahead_of_pipeline`), sectors present on only one board, and an explicit
join policy that refuses unreliable keys.

**Data quality** — every issue is classified as *excluded from the calculation*,
*included but incomplete*, or *informational*, and only the relevant ones surface.

**Leadership update** — a copy-paste-ready briefing with Executive Summary,
Pipeline, Operations, Key Risks, Opportunities and Data Quality sections.

**Error handling** — Monday outage, invalid token, board not found, empty board,
malformed rows, missing columns, Groq failure, malformed model JSON, analytics
errors and rate limits all degrade gracefully. Users never see a stack trace;
developers get full diagnostics in the logs (with credentials redacted).

---

## Setup

**Requirements:** Python 3.11+

```bash
git clone <your-repo-url>
cd skylark-monday-bi-agent

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt    # add -r requirements-dev.txt for tests

cp .env.example .env               # then fill in the values
```

### Environment variables

| Variable | Required | Default | Purpose |
|---|:--:|---|---|
| `MONDAY_API_TOKEN` | ✅ | — | Monday.com API token |
| `MONDAY_DEALS_BOARD_ID` | ✅ | — | Deals board ID |
| `MONDAY_WORK_ORDERS_BOARD_ID` | ✅ | — | Work Orders board ID |
| `GROQ_API_KEY` | ✅ | — | Groq API key |
| `GROQ_MODEL` | | `openai/gpt-oss-120b` | Any Groq chat model — see note below |
| `CACHE_TTL_SECONDS` | | `300` | How long a board snapshot is reused |
| `FISCAL_YEAR_START_MONTH` | | `4` | `4` = Indian FY (Apr–Mar), `1` = calendar |
| `MONDAY_PAGE_SIZE` | | `250` | Items per GraphQL page |
| `REQUEST_TIMEOUT_SECONDS` | | `45` | Monday.com HTTP timeout |
| `DELAY_GRACE_DAYS` | | `0` | Days past planned end before "delayed" |
| `LOG_LEVEL` | | `INFO` | Logging verbosity |

> **Groq retires models periodically.** If a call fails with `model_not_found`,
> run `python -m tools.list_models` to print the model IDs your account can
> actually use, then set `GROQ_MODEL` accordingly. The app reports this as a clear
> configuration error rather than silently degrading.

Without `GROQ_API_KEY` the app still runs: it uses keyword-based question
understanding and renders the computed figures directly, without LLM narration.

### Run locally

```bash
python -m tools.verify_connection     # confirms boards, columns and a sample answer
python -m tools.list_models           # lists the Groq models this account can use
streamlit run app.py                  # http://localhost:8501
```

---

## Monday.com setup

### 1. Create a workspace

Sign up at <https://monday.com> (the free tier is sufficient) and open or create a
workspace.

### 2. Create the Deals board from the spreadsheet

Monday's importer creates the board *and* its columns in one step, which is the
quickest reliable path:

1. In the workspace, click **+ Add** → **Import data** → **Excel**.
2. Upload `seed_data/Deal funnel Data.xlsx`.
3. Name the board **Deals**.
4. Choose *first row contains headers*.
5. Map the columns as below, then **Import**.

| Spreadsheet column | Monday column type | Notes |
|---|---|---|
| `Deal Name` | Item name (or Text) | |
| `Owner code` | Text | masked owner codes (`OWNER_001`…) |
| `Client Code` | Text | masked client codes (`COMPANY089`…) |
| `Deal Status` | Status | Open / Won / Dead / On Hold |
| `Close Date (A)` | Date | mostly empty — expected |
| `Closure Probability` | Status or Dropdown | High / Medium / Low |
| `Masked Deal value` | Numbers | ~52 % of rows are blank — expected |
| `Tentative Close Date` | Date | the expected-close field |
| `Deal Stage` | Status | `A. Lead Generated` … `O. Not Relevant at all` |
| `Product deal` | Text or Dropdown | |
| `Sector/service` | Text or Status | |
| `Created Date` | Date | |

> **Text columns are fine.** If Monday will not let you set a column to Date or
> Numbers, leave it as Text — the normaliser parses text dates and currency
> strings, and reports anything it cannot parse.

### 3. Create the Work Orders board

`Work_Order_Tracker Data.xlsx` has a **title row above the real header**, so:

* if the importer offers a "header row" selector, choose **row 2**; or
* open the file, delete row 1, save, and import the result.

Name the board **Work Orders** and map:

| Spreadsheet column | Monday column type |
|---|---|
| `Serial #` | Item name (or Text) — unique work-order ID |
| `Deal name masked` | Text |
| `Customer Name Code` | Text |
| `Sector` | Text or Status |
| `BD/KAM Personnel code` | Text |
| `Execution Status` | Status |
| `Nature of Work` | Status or Text |
| `Type of Work` | Text |
| `Date of PO/LOI` | Date |
| `Probable Start Date` | Date |
| `Probable End Date` | Date |
| `Data Delivery Date` | Date |
| `Amount in Rupees (Excl of GST) (Masked)` | Numbers |
| `Billed Value in Rupees (Excl of GST.) (Masked)` | Numbers |
| `Collected Amount in Rupees (Incl of GST.) (Masked)` | Numbers |
| `Amount Receivable (Masked)` | Numbers |
| `WO Status (billed)` | Status |
| `Invoice Status` | Status |

The remaining columns can be imported or skipped — the agent uses what it finds
and reports the rest as unused.

### 4. Copy the board IDs

Open each board; the ID is the number in the URL:

```
https://your-account.monday.com/boards/1234567890
                                       ^^^^^^^^^^
```

Pasting the **whole URL** into `MONDAY_DEALS_BOARD_ID` also works — the config layer
extracts the numeric ID.

### 5. Create an API token

Monday.com → your avatar (bottom-left) → **Developers** → **My access tokens** →
**Show / Copy**. The token needs read access to both boards.

### 6. Configure and run

```bash
cp .env.example .env
# paste MONDAY_API_TOKEN, both board IDs and GROQ_API_KEY
python -m tools.verify_connection
streamlit run app.py
```

`verify_connection` prints the resolved column mapping, so you can see exactly
which Monday column the agent chose for each field, and which fields it could not
find.

### Column names do not have to match exactly

The agent resolves canonical fields against whatever columns your board actually
has, using alias tables, token overlap, fuzzy matching and column-type hints.
`Deal Value`, `Value`, `Amount` and `Masked Deal value` all resolve to the deal
value; `Industry`, `Vertical` and `Sector/service` all resolve to sector.

If auto-detection picks the wrong column, the mapping layer accepts explicit
overrides (`ColumnMapping` `overrides` argument in `DataService`), and every
resolution is shown in the UI's **Data sources** panel.

---

## Deployment (Streamlit Community Cloud)

1. Push the repository to GitHub. `.env` is git-ignored — verify no secret is
   committed (`git grep -iE "gsk_|eyJ"` should return nothing).
2. Go to <https://share.streamlit.io> → **New app**, pick the repo/branch and set
   the main file to `app.py`.
3. Open **Advanced settings → Secrets** and paste (see
   `.streamlit/secrets.toml.example`):

   ```toml
   MONDAY_API_TOKEN = "..."
   MONDAY_DEALS_BOARD_ID = "1234567890"
   MONDAY_WORK_ORDERS_BOARD_ID = "0987654321"
   GROQ_API_KEY = "..."
   GROQ_MODEL = "openai/gpt-oss-120b"
   ```

4. Deploy. `requirements.txt` is the only build input; there are no local-only
   dependencies, no hard-coded paths and no compiled extensions.

The same repository runs unchanged on Render, Railway, Fly.io or a container —
the start command is always `streamlit run app.py`.

---

## Example questions

1. How's our pipeline looking this quarter?
2. What's the energy sector pipeline?
3. Which sectors have the strongest pipeline?
4. What are our biggest opportunities?
5. Which deals are at risk?
6. What's our expected revenue this quarter?
7. How many active work orders do we have?
8. Which projects are delayed?
9. How is the mining sector performing?
10. Compare our sales pipeline with our operational workload.
11. Which sectors have strong pipeline but low delivery workload?
12. Where might we need delivery capacity next?
13. How is OWNER_002 performing?
14. What data quality issues should I know about?
15. Prepare a leadership update for this quarter.

---

## Testing

```bash
pip install -r requirements-dev.txt
pytest                # 238 tests, ~11s, no network access required
```

The Monday.com API is mocked with `httpx.MockTransport` and Groq with a stub
client, so the suite runs offline. The fake board deliberately reproduces the seed
data's messiness: casing variants, `N/A` values, `₹2.5 Cr` strings, four date
formats, an unparseable amount, a missing status, an unknown sector, a duplicate
row and a header row pasted into the data.

Coverage by area: sector/status/stage normalisation, date parsing, currency
parsing, missing-value handling, malformed and empty datasets, pipeline metrics,
work-order metrics, cross-board aggregation, column mapping, pagination, every
API error path, caching and staleness, planner fallbacks, leadership-update
structure, fact validation, and a headless Streamlit render.

---

## Assumptions

Every assumption below is stated in the app's own output when it affects a figure,
not buried here.

**Business semantics**

| Assumption | Rationale |
|---|---|
| Fiscal year starts in **April** | Indian company; "this quarter" means the fiscal quarter. Configurable via `FISCAL_YEAR_START_MONTH=1` for calendar quarters. The resolved window is always printed ("Q4 FY2025-26 (Jan–Mar 2026)"). |
| `Deal Status` is authoritative; `Deal Stage` fills in only when status is missing | Status is the explicit field; stage is a funnel position. Rows resolved this way are flagged `status_inferred`. |
| `Dead` → `Lost` | Folds "not relevant at all / at the moment" into losses, making the win rate **conservative**. Stated in the win-rate basis. |
| **Late stage** = `E. Proposal/Commercials Sent` + `F. Negotiations` | The last stages before a win; `G`–`K` are already won. Derived from the board's own `A.`–`O.` ordering prefix. |
| **Active work order** = Not Started, In Progress or Blocked | All three still consume delivery capacity. |
| **Delayed** = active **and** past the planned end date | There is no revised-deadline field. `DELAY_GRACE_DAYS` softens the threshold. |
| Work orders with an **unrecognised status** count as neither active nor delayed | Their state is genuinely unknown; guessing would corrupt both counts. |
| Win rate needs **≥ 5 closed deals** | Below that the percentage is noise, so it is suppressed with a reason. |

**Numeric and unit assumptions**

| Assumption | Rationale |
|---|---|
| `1 Lakh = 100,000`, `1 Crore = 10,000,000`; `K`/`M`/`B` western | Standard Indian and western conventions. |
| Base currency is **INR**; non-INR amounts are parsed but **excluded from totals** | Inventing an exchange rate would be fabrication. Excluded rows are reported as a data-quality issue. |
| Closure probability `High`/`Medium`/`Low` → **0.75 / 0.45 / 0.20** | The board stores a band, not a percentage. This is the single largest judgement call in the project: it is published with **every** weighted figure alongside the count of deals that had no probability, and lives in one place (`utils/text.py::PROBABILITY_BANDS`) so it can be tuned. |
| Order value is **excl.** GST while billed/collected are **incl.** GST | As recorded on the board. Billing ratios therefore carry an explicit "indicative only" caveat. |
| Dates are **day-first** for ambiguous `dd/mm/yyyy` | Indian convention; overridden automatically when one component exceeds 12. |

**Data-handling assumptions**

| Assumption | Rationale |
|---|---|
| **Missing ≠ zero**, always | A deal with no recorded value is excluded from totals and counted, never summed as ₹0. This extends to grouped sums, which use `min_count=1` so an all-missing group reads "value missing", not "₹0". |
| **Duplicates are flagged but retained** | Masked deal names repeat legitimately across the dataset; dropping "duplicates" would understate the pipeline. The count is surfaced so the founder can judge. |
| **Unknown sectors are preserved**, not bucketed into "Others" | A new sector must stay visible. Only explicitly-mapped aliases collapse. |
| `Renewables` and `Powerline` stay **distinct** from `Energy` | Merging them would misstate sector totals. Instead, an umbrella lookup means a question about "energy" reaches both. |
| A named sector **not present on the board returns an empty scope** | Silently answering about every sector would be a dangerous fallback. |
| **Sector is the only cross-board join key** | See *Challenges faced*. |

---

## Trade-offs

**Deterministic analytics over LLM computation.** Costs some flexibility — the
agent can only answer questions the analytics layer supports, whereas a
code-generating agent could answer anything. Buys correctness that can be
unit-tested, which for financial figures is the right trade.

**Streamlit over a React + API stack.** Costs fine-grained UI control and rules
out serverless-only hosts such as Vercel (Streamlit needs a persistent WebSocket
server). Buys a working, deployable, data-native UI in a fraction of the time —
and every alternative free host (Streamlit Community Cloud, Render, Railway,
Fly.io, Hugging Face Spaces) runs it unchanged.

**Aggregate cross-board comparison over record-level joins.** Costs the ability to
say "this deal became that work order". Buys honesty: the boards share no reliable
key, and fuzzy-matching them would invent relationships. The join policy documents
the rejected keys explicitly rather than quietly guessing.

**Two LLM calls per question (plan, then narrate) rather than one.** Costs latency
and tokens. Buys a clean separation: the planner runs in JSON mode at temperature
0 and is validated by Pydantic; the narrator never sees raw rows, only finished
facts. Merging them would let question-understanding errors leak into the numbers.

**Retaining duplicates rather than de-duplicating.** Costs some precision if the
board really does contain accidental duplicates. Buys against the worse failure —
silently deleting legitimate repeated-name deals and under-reporting pipeline.
Both the count and the policy are surfaced.

**In-process cache rather than Redis.** Costs consistency across replicas (two
Streamlit Cloud instances may hold snapshots seconds apart). Buys zero
infrastructure. Adequate at this scale.

**Read-only Monday access.** Costs write-back features. Buys the elimination of a
whole class of destructive failure.

**Fixed risk heuristics rather than a learned model.** Costs sophistication. Buys
explainability — every flagged deal lists the concrete signals that triggered it —
and avoids a model trained on no outcome history.

---

## AI tools used

**At runtime (in the product):**

- **Groq** — `openai/gpt-oss-120b` by default, for two narrow jobs: converting a
  question into a structured `QueryPlan` (JSON mode, temperature 0, validated by
  Pydantic), and narrating already-computed facts. It performs **no arithmetic**.
  Configurable via `GROQ_MODEL`; `python -m tools.list_models` lists valid IDs.
- Both jobs have **non-AI fallbacks**: a keyword planner and a deterministic
  markdown renderer, so the application remains correct and usable with no LLM at
  all. This is tested (`test_end_to_end_answer_without_groq`).

**During development:**

- **Claude Code (Claude Opus)** — used as a pair programmer for the whole build:
  profiling the seed spreadsheets, drafting modules, writing the test suite, and
  debugging the live-integration failures. Every design decision (the LLM/Python
  split, the join policy, the probability weights, the missing-≠-zero rule) was
  made deliberately and is documented in `DECISION_LOG.md`; the AI accelerated
  implementation, it did not choose the architecture.
- All generated code was reviewed, executed and tested. The 238-test suite exists
  partly as the verification mechanism for that: nothing was accepted on the basis
  that it looked plausible.

**Not used:** no vector database, no embeddings, no RAG, no agent framework
(LangChain/LlamaIndex). The task needs a structured query plan against two known
boards, not semantic retrieval — adding a framework would have added dependency
weight and indirection without capability.

---

## Challenges faced

**1. The data is much messier than a schema implies.** Profiling the seed files
first was the single highest-value hour of the project. It revealed: two rows in
the deals sheet that are the column headers pasted back into the data; a title row
above the real header in the work-order file (so `header=1`); 12 exact duplicate
rows; and 52 % of deals with no recorded value. Detecting header-echo rows needed a
dedicated heuristic (a row where ≥ 3 cells, and ≥ 50 % of populated cells, equal
their own column title).

**2. "Missing" had to be designed for, not patched around.** The naive path —
`fillna(0)` — would have reported a ₹0 pipeline for a sector whose deals simply
lack recorded values. Preventing that required care at every layer: `NaN` with a
recorded reason at parse time, `min_count=1` on every grouped sum, and a
three-severity `DataQualityReport` so the answer can distinguish *excluded from
the calculation* from *included but incomplete*. One instance of this bug survived
until live testing (see #6).

**3. Closure probability is categorical, not numeric.** A weighted pipeline is a
standard executive metric, but the board stores `High`/`Medium`/`Low`. Options
were: skip the metric, or assume weights. I assumed weights, isolated them in one
constant, and made the app publish the assumption alongside every weighted figure
— visible judgement rather than a hidden one.

**4. The two boards share no reliable join key.** Clients are `COMPANY089` on the
deals board and `WOCOMPANY_002` on the work-order board, with no mapping table.
Deal names are masked aliases (`Sakura`, `Naruto`) that repeat across many rows.
Fuzzy-matching them would have produced impressive-looking, fabricated links. The
resolution was to join on **normalised sector only**, compare *shares* rather than
raw values (unit-free, so it works even when one board's values are missing), and
publish a `JOIN_POLICY` object naming the rejected keys and why.

**5. Monday column IDs bear no relation to column titles.** An imported board gets
`text0`, `date4`, `numbers` — and an evaluator may rename columns anyway. Hard-coding
IDs would have made the app work only on my board. The fix was a resolver that
discovers columns at runtime and scores every (canonical field, column) pair by
alias match, prefix/substring, token overlap, fuzzy similarity and column-type
preference, assigning greedily by descending score so one column is never claimed
twice. On the real boards it resolved **12/12 deal fields and 22/22 work-order
fields exactly**.

**6. Four defects appeared only against the live API.** Worth stating plainly,
because mocked tests alone would have shipped them:
   - **Board IDs pasted as full URLs** — now extracted by the config layer, since
     the evaluator will likely do the same thing.
   - **Groq decommissioned the default model** (`llama-3.3-70b-versatile`) during
     the build. Now a `model_not_found` raises a distinct error telling the
     operator to run `tools.list_models`, instead of silently degrading to keyword
     mode. Token budgets were also raised, because the replacement is a reasoning
     model that spends tokens before answering.
   - **The narrator conflated two counts**, describing "344 deals" as "all open"
     when 344 was the total across all statuses. Fixed by renaming the fields to
     be self-describing (`total_deals_on_board_all_statuses`,
     `open_deals_on_board_all_periods`, `deals_matching_all_filters`) and adding a
     prompt rule against conflating them. A lesson in prompt design: ambiguous
     field names *invite* hallucination.
   - **A month whose only deal had no value reported ₹0** — pandas sums all-`NaN`
     to `0.0`. Exactly the failure the design forbids, hiding in a groupby. Fixed
     here and in `cross_board.py`, with regression tests.

**7. A correct empty answer looks like a broken app.** Asking "how's the pipeline
this quarter?" returns zero on this dataset, because the seed deals close by April
2026. That is right, but indistinguishable from a failed connection. The fix was an
`empty_scope_context` block: when a filter empties the scope, the agent reports
where the open pipeline *actually* sits, month by month, and says explicitly that
the data is present but falls outside the requested window.

**8. Keeping the LLM from being load-bearing.** It would have been easy to let
Groq become a single point of failure. Every LLM path has a tested non-AI
fallback, and `validate_facts()` sanity-checks the computed block before narration
(negative pipeline, a breakdown exceeding its total, overlapping status counts).

---

## Design decisions

Full rationale in [DECISION_LOG.md](DECISION_LOG.md). In brief:

- **LLM for language, Python for arithmetic.** Groq classifies and explains;
  pandas computes. Money values are passed to the model pre-formatted
  (`"₹12.40 Cr"`) with an instruction to quote them verbatim, so it cannot
  re-derive or mis-round a figure.
- **Runtime source of truth is Monday.com.** Seed spreadsheets live in
  `seed_data/` and are read only by a developer profiling tool. Nothing in the
  runtime path opens them.
- **Column IDs are discovered, never assumed.** A resolver maps canonical fields
  to whatever the board actually has.
- **Missing ≠ zero.** An unparseable or absent deal value is excluded from totals
  and reported, so the pipeline is never silently understated as ₹0.
- **Metrics are suppressed, not guessed.** No probability column means no weighted
  pipeline — the app says so instead of inventing weights.
- **Sector is the only cross-board join key.** Customer codes differ between the
  boards (`COMPANY089` vs `WOCOMPANY_002`) with no mapping table, and masked deal
  names repeat, so record-level joins are refused.
- **Caching is a latency optimisation, not a source of truth.** Short TTL, visible
  timestamp, explicit refresh, and stale data is labelled as stale.

---

## Limitations

- **No historical trend analysis.** Monday.com returns current board state; the app
  does not snapshot history, so it cannot say "pipeline grew 12 % vs last quarter".
  Every figure is a point-in-time view.
- **Weighted pipeline uses assumed weights.** The board records `High`/`Medium`/`Low`,
  not a percentage. The mapping (0.75 / 0.45 / 0.20) is a documented assumption,
  surfaced with every weighted figure, and configurable in `utils/text.py`.
- **No deal-to-work-order linkage.** Cross-board analysis is aggregate by sector
  only. It indicates imbalance; it does not prove causation, and it cannot tell you
  which specific deal produced which work order.
- **Currency conversion is refused, not performed.** Non-INR amounts are excluded
  from INR totals and reported as a data-quality issue.
- **GST bases differ.** Order value is exclusive of GST while billed/collected are
  inclusive, so billing ratios are indicative — the app states this caveat.
- **"Delayed" means "past the planned end date and not complete."** There is no
  separate revised-deadline field, so a legitimately re-planned project reads as
  delayed. Use `DELAY_GRACE_DAYS` to soften the threshold.
- **Duplicates are retained.** Masked deal names repeat legitimately, so exact
  duplicates are flagged and counted rather than dropped; totals may include them.
- **The cache is per process.** Multiple Streamlit Cloud replicas each hold their
  own snapshot, so two users may briefly see data fetched seconds apart.
- **Board size.** Pagination is capped at 200 pages (~50k items at the default page
  size); beyond that the agent stops and logs a warning rather than looping.
- **English only**, and the sector alias tables are tuned to this dataset's
  vocabulary — a new sector name maps through fuzzy matching or is passed through
  unchanged.

---

## Potential improvements

Ordered by the value they would add, not by effort.

1. **Daily board snapshots.** The single biggest limitation is that Monday returns
   only current state, so no trend is possible. Persisting a daily snapshot to
   SQLite or Postgres would unlock the questions a founder actually asks most:
   pipeline movement quarter on quarter, stage-conversion rates, deal velocity,
   ageing, and "what changed since last week". This would change the product more
   than anything else on this list.
2. **A customer mapping table between the two boards.** Reconciling `COMPANY089`
   with `WOCOMPANY_002` — even a hand-maintained CSV — would lift cross-board
   analysis from sector-level indication to record-level truth: true deal→delivery
   tracing, per-customer lifetime value, and won-deal-to-work-order conversion time.
3. **A learned deal-risk model**, once win/loss history with timestamps exists,
   replacing the current heuristics. The heuristics are explainable but static;
   a model trained on actual outcomes would rank risk by what has historically
   predicted a loss at this company.
4. **Charts.** A funnel by stage, a sector treemap and a delivery-timeline Gantt
   would carry more than the current tables, and Streamlit makes them cheap. The
   analytics layer already returns exactly the shapes a chart needs.
5. **A golden-question regression suite.** Fixed board fixtures plus expected
   figures for ~20 canonical questions, run in CI, so any analytics change that
   moves a number is caught deliberately rather than discovered in a meeting.
6. **Real Monday date/number column types.** The importer often lands spreadsheet
   columns as text. The parsers handle it, but native `date`/`numbers` columns
   would remove a whole class of ambiguity and let some filtering be pushed into
   the GraphQL query rather than done in pandas.
7. **Server-side filtering via GraphQL `query_params`.** Currently the full board
   is fetched and filtered locally, which is fine at 344 + 176 rows but would not
   scale to tens of thousands. Pushing sector/status filters into the query would.
8. **Semantic caching of planner results**, removing one LLM round-trip for
   repeated or near-identical questions and roughly halving perceived latency.
9. **Streaming narration**, so the answer begins rendering immediately instead of
   after the full completion.
10. **Per-user Monday OAuth** instead of one service token, so board permissions
    are respected per viewer — necessary before this could be shared beyond the
    leadership team.
11. **Scheduled leadership updates** — the briefing generator already exists;
    emailing or Slacking it every Monday morning is a small addition with
    disproportionate value.
12. **Configurable business rules in the UI.** The probability weights, the
    late-stage definition and the delay grace period are the three judgement calls
    most likely to be contested. Exposing them as settings would let the founder
    align the tool with how they actually think, rather than accepting my defaults.
