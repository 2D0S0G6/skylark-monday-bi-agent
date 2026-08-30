# Decision Log

Skylark Drones — Monday.com Business Intelligence Agent

Guiding principle: **the LLM does language, Python does arithmetic.** Every figure is
computed in pandas; the model only reads the question and explains the result. A
confidently wrong ₹ figure is worse than an honest "this cannot be calculated".
Architecture, setup and full assumption tables: [README.md](README.md).

---

## 1. Key assumptions

**Business semantics**

- **Fiscal year starts in April**, so "this quarter" is the fiscal quarter and the
  resolved window is always printed ("Q4 FY2025-26 (Jan–Mar 2026)"). Configurable.
- **Closure probability is categorical, not numeric.** The board records
  High/Medium/Low, so weighted pipeline maps them to **0.75 / 0.45 / 0.20** — the
  largest judgement call here. It is published with every weighted figure, alongside
  the count of deals with no probability, and sits in one constant so it can be retuned.
- **Deal status is authoritative**; funnel stage fills in only when status is missing,
  and those rows are flagged `status_inferred`. **`Dead` maps to `Lost`**, which makes
  the win rate conservative — stated in the win-rate basis.
- **Late stage** = *Proposal/Commercials Sent* + *Negotiations*, from the board's own
  `A.`–`O.` ordering. **Delayed** = active and past the planned end date; there is no
  revised-deadline field, so `DELAY_GRACE_DAYS` softens it. Work orders with an
  unrecognised status count as neither active nor delayed — their state is unknown.
- **Win rate needs ≥ 5 closed deals**, below which it is suppressed with a reason.

**Units** — `1 Lakh = 1e5`, `1 Crore = 1e7`; `K/M/B` western. Base currency is INR; a
`$` amount is parsed, tagged `USD` and **excluded from INR totals**, because inventing
an exchange rate would be fabrication. Ambiguous `dd/mm/yyyy` is read **day-first**.
Order value is excl. GST while billed/collected are incl., so billing ratios carry an
"indicative only" caveat.

**Missing data** — the dataset is roughly half-empty (52 % of deals have no value,
21 % no expected close date), so missingness is a first-class concept:

- **Missing is never zero.** Absent and unparseable values become `NaN` with a reason,
  then are excluded from totals and counted — understated-and-labelled rather than
  wrong-and-silent. This extends to grouped sums (`min_count=1`), so an all-missing
  month reads "value missing", not "₹0".
- **Duplicates are flagged but retained.** Masked deal names repeat legitimately, so
  dropping them would understate the pipeline. The count is surfaced instead.
- **Unknown sectors are preserved**, not bucketed into "Others". `Renewables` and
  `Powerline` stay distinct from `Energy`, with an umbrella lookup so "energy" reaches
  both. A named sector absent from the board returns an **empty scope**, never the
  whole board — silently widening the answer would be a dangerous fallback.

---

## 2. Trade-offs chosen, and why

- **Deterministic analytics over LLM computation.** Costs flexibility — the agent can
  only answer what the analytics layer supports. Buys correctness that is
  unit-testable, which for financial figures is the right trade. Money reaches the
  model pre-formatted (`"₹12.40 Cr"`) to be quoted verbatim, so it cannot mis-round it.
- **Two LLM calls (plan, then narrate) rather than one.** Costs latency and tokens.
  Buys separation: the planner runs in JSON mode at temperature 0 and is
  Pydantic-validated; the narrator sees only finished facts, never raw rows. Merging
  them would let question-understanding errors leak into the numbers.
- **Aggregate cross-board comparison over record-level joins.** The boards share no
  reliable key — clients are `COMPANY089` on one and `WOCOMPANY_002` on the other, and
  masked deal names repeat. Fuzzy-matching would invent relationships, so the join is
  sector-only and the rejected keys are documented.
- **Read-only Monday access.** Costs write-back. Buys the elimination of a whole class
  of destructive failure: no question, however phrased, can alter a board.
- **Fixed risk heuristics over a learned model.** With no outcome history a scoring
  model would be unfounded; each flagged deal lists the signals that triggered it.
- **No trend language, and an in-process cache rather than Redis.** Monday returns
  current state only, so prior-period comparisons are impossible and the prompt forbids
  inventing them; the cache costs cross-replica consistency, buys zero infrastructure.

---

## 3. What I would do differently with more time

1. **Daily board snapshots.** The single biggest limitation is that only current state
   is available, so no trend is possible. Snapshots would unlock what founders ask
   most: pipeline movement quarter on quarter, stage conversion, velocity, ageing.
2. **A customer mapping table** between the two code schemes, lifting cross-board work
   from sector-level indication to record-level deal-to-delivery tracing.
3. **A golden-question regression suite** — fixed fixtures with expected figures, in
   CI, so any change that moves a number is caught deliberately.
4. **Charts.** A stage funnel and a sector treemap would carry more than the current
   tables; the analytics layer already returns the right shapes.
5. **A learned deal-risk model** once win/loss history exists, and **per-user Monday
   OAuth** rather than one service token, so permissions are respected per viewer.

---

## 4. How I interpreted "leadership updates"

A leadership update is treated as a **standing briefing, not a filtered query**.

- **Pipeline and workload cover the whole open book**, because someone preparing for a
  leadership meeting wants the full position, not a slice; the expected-close block
  then shows how that pipeline falls across quarters. A `scope_note` states this in the
  facts so the model cannot misdescribe the scope.
- If no period is named it is **anchored to the current fiscal quarter** for labelling.
- It **always runs cross-board analysis**: sales-versus-delivery balance is exactly
  what a leadership meeting needs, and no single board shows it.
- **Risks and opportunities must be data-supported** — each traces to a supplied
  figure, and the instruction is to write "no data-supported risk was identified"
  rather than pad the section. The deterministic renderer follows the same rule.
- **Fixed section order and a 400-word ceiling** — Executive Summary, Pipeline,
  Operations, Key Risks, Opportunities, Data Quality — so it can be pasted straight
  into a meeting document.
