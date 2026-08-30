# Decision Log

Skylark Drones — Monday.com Business Intelligence Agent

---

## 1. Problem interpretation

The brief asks for a founder-facing analyst, not a chart tool and not a chatbot
over a spreadsheet. A founder asking *"how's the pipeline this quarter?"* wants a
number, the one or two things that number implies, and an honest note when the
underlying data is incomplete.

That framing drove three decisions:

- **Correctness outranks fluency.** A confidently wrong ₹ figure is worse than
  "this cannot be calculated because 52 % of deals have no value recorded."
- **Data quality is part of the answer**, not an appendix. The supplied data is
  substantially incomplete, so every metric carries what was included, what was
  excluded, and why.
- **Insight, not a table dump.** Answers lead with the figure, then say what it
  means for the business.

Ambiguities were resolved by making an assumption, surfacing it in the answer, and
recording it here.

## 2. Architecture choice

A layered pipeline with one direction of dependency:

```
UI → planner (LLM) → Monday client → normalisation → analytics → validation → narrator (LLM)
```

Each layer is independently testable, which is why the suite can cover the whole
stack without a network. `analytics/` has no knowledge of Monday or Groq; it takes
DataFrames and returns dicts. `monday/` has no knowledge of deals or work orders;
it returns generic board snapshots. The domain meaning lives entirely in
`monday/column_map.py` and `analytics/`.

The orchestrator is the only component that knows about all of it, and it is the
only place that catches broad exceptions — so no failure escapes to the user as a
stack trace.

## 3. Why Groq

Specified by the brief, and a good fit. Two LLM calls per question (plan, then
narrate) sit on the interactive path, so latency is felt directly; Groq's inference
speed keeps a full answer in the low seconds.

The default is `openai/gpt-oss-120b` — the most capable model available on the test
account, and one that supports JSON mode, which the planner requires. It is a
reasoning model, so token budgets allow headroom for reasoning output before the
answer.

Groq retires models on its own schedule (`llama-3.3-70b-versatile`, the original
default, was decommissioned mid-build). Rather than pin a model that will rot,
`GROQ_MODEL` is configurable, a `model_not_found` response raises a distinct
`LLMModelNotFoundError` with a fix instruction, and `python -m tools.list_models`
prints the currently valid IDs. A dead model name is a one-line fix, not a silent
degradation to keyword mode.

## 4. Why Streamlit

The brief specifies it, and for a data-centric internal tool it is genuinely the
right call: chat, dataframes, expanders, spinners and caching come built in, and
the deployment story on Streamlit Community Cloud is a `requirements.txt` and four
secrets. A React frontend plus an API service would have cost most of the time
budget and delivered no analytical capability.

The UI is deliberately shaped as an executive tool rather than a generic chat: a
KPI strip, a data-quality panel with severity coding, an "Analysis details" panel
showing the parsed intent and the tables behind the narrative, and a "Data sources"
panel showing exactly which Monday column each canonical field resolved to.

**Trade-off:** Streamlit needs a persistent WebSocket server, so it cannot be
hosted on serverless-only platforms such as Vercel. Streamlit Community Cloud,
Render, Railway, Fly.io and Hugging Face Spaces all work unchanged.

## 5. Why deterministic analytics + LLM, not LLM-only calculation

LLMs are unreliable arithmetic engines and — more importantly — *silently*
unreliable. A model that sums 18 deal values wrongly produces a plausible number
with no error signal. For a founder making decisions on these figures, that is
disqualifying.

So every sum, count, percentage, share, ratio and rank is computed in pandas. The
model receives a JSON block of finished facts. Money is passed **pre-formatted**
(`{"amount": 124000000, "display": "₹12.40 Cr"}`) with a prompt instruction to
quote the `display` string verbatim, which removes the opportunity to re-derive or
mis-round a figure. The narrator prompt forbids stating anything absent from the
JSON.

Two further guards:

- `validate_facts()` sanity-checks the computed block before narration (negative
  pipeline, a breakdown exceeding its total, overlapping work-order status counts).
- If Groq fails or returns something unusable, a deterministic markdown renderer
  produces the answer from **the same facts**. Correctness never depends on the LLM
  being available.

## 6. Monday.com API decision

GraphQL v2, `items_page` with cursor pagination, `API-Version` header pinned.

The significant decision was **not to assume column IDs or titles.** Monday
generates opaque IDs (`text0`, `date4`, `numbers`) that bear no relation to the
titles, and any evaluator setting up the boards may rename a column. So the client
first queries board metadata, then a resolver scores every (canonical field,
column) pair using an alias table, prefix/substring matching, token overlap, fuzzy
similarity and a column-type preference, assigning greedily by descending score so
one column is never claimed twice. Explicit overrides win outright.

Consequences that matter: `Deal Value`, `Amount` and `Masked Deal value` all
resolve to the deal value; a board missing a column produces an explicit
"unavailable" for the dependent metrics rather than a crash; and the resolution is
shown in the UI so a wrong match is visible rather than silent.

Errors are mapped to typed exceptions (auth, board-not-found, rate limit, outage)
each carrying a safe `user_message`. Retries use exponential backoff with jitter.

**Caching:** a 5-minute TTL, in-process. Cache is a latency optimisation, never the
source of truth — the fetch timestamp is displayed, "Refresh data" always bypasses
it, and if a refresh fails the previous snapshot is only reused when explicitly
labelled stale.

## 7. Data normalisation assumptions

Written against the actual seed files, not a hoped-for schema:

- **Currency.** `1 Lakh = 1e5`, `1 Crore = 1e7`; `K/M/B` western. Base currency is
  INR. A `$` amount is parsed to its magnitude, tagged `USD`, and **excluded from
  INR totals** with a data-quality note — inventing an exchange rate would be a
  fabrication.
- **Dates.** Day-first (Indian convention) for ambiguous `dd/mm/yyyy`, overridden
  when one component exceeds 12. Excel serials are recognised.
- **Fiscal year starts in April** (`FISCAL_YEAR_START_MONTH=4`), so "this quarter"
  means the fiscal quarter, and the period is always stated explicitly in the answer
  ("Q4 FY2025-26 (Jan 2026–Mar 2026)"). Set to `1` for calendar quarters.
- **Closure probability is categorical**, not numeric. Weighted pipeline maps
  High → 0.75, Medium → 0.45, Low → 0.20. This is an **assumption**, published with
  every weighted figure alongside the count of deals that had no probability.
- **Deal status is authoritative**; the funnel stage fills in only when status is
  missing, and such rows are flagged `status_inferred`.
- **`Dead` maps to `Lost`**, which folds "not relevant" outcomes into losses. This
  makes the win rate conservative, and it is stated in the win-rate basis.
- **Sectors stay distinct.** `Renewables` and `Powerline` are not merged into
  "Energy" — that would misstate sector totals. Instead an umbrella lookup means a
  question about "energy" reaches both. Unrecognised sectors are **preserved**
  title-cased rather than bucketed into "Others", so a new sector stays visible.
- **A named sector that is not on the board returns an empty scope**, not the whole
  board — answering about everything would be a silent, dangerous fallback.
- **"Delayed"** = active (not started / in progress / blocked) **and** past the
  planned end date. Work orders with an unrecognised status are excluded from both
  the active and delayed counts, because their state is genuinely unknown.
  `DELAY_GRACE_DAYS` softens the threshold.

## 8. Handling missing and inconsistent data

The dataset is roughly half-empty in places (52 % of deals have no value; 20 % have
no expected close date), so missingness had to be a first-class concept.

- **Missing is never zero.** Unparseable and absent values become `NaN` with a
  recorded reason; a missing deal value is excluded from totals and counted, so the
  pipeline is understated-and-labelled rather than wrong-and-silent.
- **Three severities.** `excluded` (rows dropped from the calculation),
  `included_with_gap` (counted but a field is unknown), `info` (background). The UI
  colour-codes them and the narrator is told to surface at most two.
- **Originals are preserved.** `sector_raw` sits beside `sector`; nothing is
  destroyed by normalisation.
- **Header rows pasted into data are detected and removed** — the seed workbook
  contains two, and a spreadsheet import carries them into Monday.
- **Duplicates are flagged and reported but retained.** Masked deal names repeat
  legitimately across the dataset, so dropping "duplicates" would understate the
  pipeline; the count is surfaced so the founder can judge.
- **Denominators are honest.** Delay percentage is measured against work orders
  that *can* be assessed (active, with a planned end date), not against everything.

## 9. Interpretation of leadership updates

A leadership update is treated as a **standing briefing**, not a filtered query.

- Pipeline and workload cover the **whole open book**, because a founder preparing
  for a leadership meeting wants the full position; the expected-close block then
  shows how that pipeline falls across quarters. A `scope_note` states this
  explicitly in the facts so the model cannot misdescribe it.
- If no period is named, it is **anchored to the current fiscal quarter** for
  labelling purposes.
- It always runs cross-board analysis: sales-versus-delivery balance is exactly the
  kind of thing a leadership meeting needs and no single board can show.
- **Risks and opportunities must be data-supported.** The prompt requires each one
  to trace to a supplied figure and to say "no data-supported risk was identified"
  rather than pad the section. The deterministic renderer follows the same rule.
- Fixed section order and a 400-word ceiling, so the output can be pasted straight
  into a meeting document.

## 10. Trade-offs due to the time constraint

- **No historical snapshotting.** Trend language ("up 12 % on last quarter") is
  impossible from current board state, so it is not attempted — the prompt forbids
  invented comparisons.
- **Cross-board joins are sector-level only.** The boards share no reliable record
  key: clients are `COMPANY089` on one and `WOCOMPANY_002` on the other with no
  mapping table, and masked deal names repeat. Fuzzy-matching them would invent
  relationships, so the join policy explicitly documents the rejected keys.
- **In-process cache**, not Redis. Fine for this scale; multiple Streamlit Cloud
  replicas would each hold their own snapshot.
- **Fixed risk heuristics** rather than a learned model — with no outcome history,
  a scoring model would be unfounded. Each flagged deal lists the concrete signals
  that triggered it.
- **No write-back to Monday.** Read-only was sufficient for the brief and avoids a
  whole class of destructive failure.
- **`tools/seed_to_monday.py` (automated board creation) was cut** in favour of a
  precise manual import walkthrough in the README; Monday's own importer handles
  column typing better than an API script would.

## 11. What I would improve with more time

1. **Daily snapshots** to a small store, unlocking genuine trend and velocity
   analysis (pipeline movement, stage conversion rates, ageing).
2. **A learned deal-risk model** once win/loss history with timestamps exists,
   replacing the current heuristics.
3. **A customer mapping table** between the two boards' code schemes, enabling
   record-level deal→work-order tracing and true revenue attribution.
4. **Charts** — a pipeline funnel and a sector treemap would carry more than the
   current tables, and Streamlit makes this cheap.
5. **A golden-question regression suite**: fixed board fixtures with expected
   figures, run in CI, so an analytics change that shifts a number is caught.
6. **Semantic caching of planner results** to remove one LLM round-trip on repeated
   or near-identical questions.
7. **Per-user Monday tokens** via OAuth instead of one service token, so board
   permissions are respected per viewer.
8. **Streaming narration** for perceived latency, and a token-budget guard for very
   large boards.
