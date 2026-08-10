# Metron — Phase 2 UI Specification

**Status**: Converged — validated through 5 clickable-prototype rounds with Luis (2026-08-09/10)
**Companion**: `DESIGN_WEB_REBUILD.md` (architecture); prototype artifact "Metron — UX Prototype R5"
**Purpose**: The build reference for the Phase-2 frontend. Every screen, behavior, and interaction rule here was exercised in the prototype; deviations during the build should be deliberate and noted.

---

## 1. Scope

Phase 2 builds three surfaces to this spec:

1. **Dashboard** (`/`) — program-scoped inbox, campaign health, recent analyses
2. **Analysis screen** (`/analyses/{id}`) — the core loop
3. **Campaign page** (`/campaigns/{id}`) — summary, results, SPC, trends, compare, envelope, reports, settings

Out of Phase-2 scope (later phases): run-detail page, `/tools`, `/configs`, admin, window sweep UI (API exists; UI lands late Phase 2 or Phase 3), batch multi-select.

**The interaction budget is a requirement, not a metric**: the happy path from "run in inbox" to "analysis saved" must be ≤ 10 deliberate interactions; the prototype achieves 5 (Assign & Analyze → Confirm window → Pass QC → Run analysis → Save). Regressions against this budget need a reason.

---

## 2. Global shell

### 2.1 Layout

- Fixed 44 px topbar: wordmark `MET RON` (accent on second syllable), nav (Dashboard · Runs · Campaigns · Configs), right-aligned user chip (`signed in as <user> · <role>`).
- Content area fills the rest; each page owns its scroll. Minimum supported width 1100 px (engineering desktops; no mobile layout in Phase 2).

### 2.2 Routing (URL-addressable everything)

| Route | View |
|---|---|
| `/` | Dashboard (program from user subscription, `?program=` override) |
| `/runs/{id}` | Run detail (Phase 3; links may 404-stub in Phase 2) |
| `/analyses/{id}` | Analysis screen, deep-linkable at any lifecycle state |
| `/campaigns/{id}` | Campaign page, `?tab=spc&param=c_star` addressable |

Every view a user can reach must be shareable as a URL that restores it.

### 2.3 Design tokens

Dark-first. Base palette (from the Qt token layer, carried through all prototype rounds):

```css
--bg:#1e1e1e; --panel:#252526; --panel-2:#2d2d30; --panel-3:#333336;
--border:#3c3c3c; --text:#d6d6d6; --text-strong:#f0f0f0; --muted:#8f8f8f; --faint:#6b6b6b;
--accent:#3987e5;
--good:#34a361; --warn:#c98500; --bad:#e66767;        /* status — reserved, never series colors */
```

**Chart series palette** (categorical, CVD-validated against `#1e1e1e` — lightness band, chroma floor, adjacent-pair ΔE, contrast all pass; do not re-order or substitute without re-validating):

| Slot | Hex | Prototype assignment |
|---|---|---|
| 1 | `#3987e5` | chamber pressure / primary series |
| 2 | `#d95926` | ox manifold / secondary |
| 3 | `#199e70` | fuel manifold / MR chart |
| 4 | `#c98500` | (reserved: doubles as warn — never alongside a warn-colored element in the same chart) |
| 5 | `#d55181` | thrust |

Typography: `Segoe UI Variable / Segoe UI / system-ui`; monospace (`ui-monospace / Consolas`) for **all numerals, IDs, and hashes**, with `font-variant-numeric: tabular-nums` wherever digits align. Section headers: 11 px, letter-spaced, uppercase, muted. **Never uppercase-transform strings containing units or symbols** (`m/s`, `ηc*`) — R4 finding.

Components to build once and reuse: `Chip` (default/good/warn/bad/accent), `MetricCard` (name + kind badge / value + unit / ±u + rel %), `KpiStrip`, `SideCard`, `ResultsTable` (dense, right-aligned numerics, hover row), `KeyValueEditor` (§6.8), `Stepper`, `InfoBanner` (accent-left-border note), `Toast`.

### 2.4 Cross-cutting interaction rules

1. **Never re-render an input's container on its own change event** — updates go to the model and targeted DOM only. (R4 finding: naive re-render stole focus mid-typing in the template editor. This is a code-review-able rule: form components must survive a keystroke without remounting.)
2. **Every chart point has a hover tooltip** (run ID + values ±u). Nearest-point within ~10 px; tooltip follows cursor; never blocks interaction.
3. **Wheel-guard all numeric inputs** (`onWheel → blur`) so scrolling a form never changes a value.
4. Two-channel feedback: contextual `InfoBanner` in the affected panel **and** transient `Toast` for actions with off-screen effects.
5. Downstream invalidation is always **visible**: state resets get a warn toast naming what was invalidated and why.
6. Empty states explain their precondition ("Run the analysis first"), never bare-empty or an error.
7. Keyboard: `F5` re-run current stage, `Ctrl+Enter` advance/confirm, `Ctrl+S` save. Honor throughout.

---

## 3. Dashboard (`/`)

### 3.1 Program context

- Top bar of the page: `Program` selector (bold, prominent) + right-aligned summary line (`N active campaigns · M runs total`).
- The user's **subscribed programs** (`user_programs`) populate the selector; default = last used. Switching programs re-scopes campaign health, recent analyses, **and the inbox**.

### 3.2 Inbox

- Card list of runs in `READY` state with no campaign assignment, **filtered to the selected program** via each run's suggested campaign; header link toggles `show all programs (N hidden)` ⇄ `filter to this program`.
- Collapsed row shows only what's needed to act: expander (▸), run name + `new` chip, time · duration · channel count @ rate, **bench review chip** (✓/! from `report_json`), campaign select (pre-filled by the campaign's match rule), `Assign`, `Assign & Analyze →` (primary).
- **Expander (▾) reveals the meta block** (3-column key/value grid): channels sha, sequence sha, raw size, samples, rate groups, bench rule results, user tags, operator note. Hashes are never in the collapsed row — R2 finding.
- `Assign & Analyze` = assign + create draft analysis + navigate to `/analyses/{id}` in one action.

### 3.3 Campaign health cards

One card per campaign in the program (active first, then closed). Rows:

| Row | Content |
|---|---|
| head | name · test type · `● ACTIVE/CLOSED` |
| runs | count · `last <date>` |
| QC pass | `NN %` colored (≥90 good / ≥75 warn / else bad) · `pass/total` |
| cadence | runs-per-week sparkline (min 6 weeks, last week emphasized at full opacity, others 45%) · `N/wk` |
| headline | **configurable stat** (`campaigns.headline_metric`) · ⚙ inline editor · `open →` |

The ⚙ swaps the stat display for a select of available stats (Cpk per measurement, mean ±u per measurement, …); choosing one persists to the campaign (manager-only, audited) and toasts. Cpk-kind stats color by the 1.33/1.0 thresholds; mean-kind stats are neutral.

### 3.4 Also on the dashboard

Recent analyses (program-scoped; run · campaign · headline measurement ±u · QC chip · when/who; row → analysis), and a bench-config card (current channels sha, overlay + review status).

---

## 4. Analysis screen (`/analyses/{id}`)

### 4.1 Layout

Three columns: **stepper rail** (~170 px) | **stage panel** (~390 px) | **plot workspace** (flex). Header: back link, run ID, campaign chip, **state badge** (DRAFT → WINDOWED → QC_RUN → ANALYZED → SAVED; SAVED gets the good treatment), provenance mini-block (raw/channels/seq hashes truncated, overlay + processing version) right-aligned.

**The plot workspace never unmounts across stage changes** — pan/zoom/visibility state survives the whole loop.

### 4.2 Stepper

Steps: Align · Steady window · QC gate · Analyze · Save. States: `done` (✓, good ring), `active` (filled accent), `pending` (hollow, disabled), `stale` (!, warn — was done, upstream changed). Done steps are clickable to revisit. Sub-labels name the method (`auto · 50 Hz`, `events + CV`, plugin name/version).

**Invalidation rule (the DAG made visible)**: any change to the steady window — drag, numeric edit, re-suggest — while QC/Analyze/Save are done resets those steps to pending, clears their results, reverts the state badge, and toasts `Window changed — QC and analysis invalidated`. It must be impossible for a stale result to remain on screen or be saved. Same for role-map or overlay changes (reset from QC down).

### 4.3 Stages

**Align** — auto-completed at entry; read-only card (rate groups, samples, gaps, NaN policy from overlay, timebase). No action required; step exists so the record shows what happened.

**Steady window** — seeded at entry from run events (`fire_window()`), refined by CV, labelled `seeded from sequence events (FIRE → SEQ END), refined by CV`.
- Numeric `start` / `end` fields + computed duration + `Re-suggest`.
- **Bidirectional binding**: draggable region overlay on every plot panel (edges resize with `ew-resize` cursor + visible mid-height handles; body moves with `grab`); one `{t0,t1}` store written by both drag and fields; a `source` discriminator prevents update loops. Drag commits on release.
- Live stats table over the current window, updating during drag: per role-channel mean and CV% (CV colored by threshold), sample count. Served from the warm cache — must feel instant.
- `Confirm window →` advances.

**QC gate** — runs automatically on entry (progress bar while running).
- Findings list: icon (✓/!/✕) · check name · optional right-aligned detail. Blocking failures pinned on top.
- **Bench review panel** below, visually separate and labelled `(Spark Studio)`: the bench's own rule verdicts. Bench review ≠ Metron QC; both visible, neither substitutes.
- Summary note (`1 warning, 0 blocking failures. Warnings travel with the record.`). Continue button names the residue: `Continue with 1 warning →`.
- Blocking failure → continue disabled; override is a separate, explicit action requiring a typed reason (dialog), recorded and audited.

**Analyze** — two sub-states:
1. *Readiness* (before run): three sections — **Channels → roles** (role · resolved tag), **Metadata** (field · value · source layer: `run user_meta` / `campaign template` / `operator`), **Config** (overlay + review-against-sha status, uncertainty coverage `12/12 channels`). All-green enables `Run analysis`; any red row names exactly what's missing and where it would come from.
2. *Results* (after job): `MetricCard` grid — every value `value ± u (· rel %)` with a `measured`/`derived` kind badge; coverage factor stated in the header (`k = 2`). Provenance card: raw/channels/sequence hashes, overlay snapshot id, processing version, analyst + date. `Save analysis` sits directly under the results with the consequence spelled out (`saves 6 measurements + QC findings to <campaign>`).

**Save** — confirmation state: success banner (`saved as the current result … previous analyses kept and marked superseded`), summary table (measurements count, QC outcome, window, campaign n), actions: `Download test report`, `Window sweep…`, `Back to inbox` (primary).

---

## 5. Campaign page (`/campaigns/{id}`)

### 5.1 Header

Breadcrumb (`← Dashboard / <program> / <campaign>`), name + type chip + status, `Campaign report` button, and the **KPI strip**: runs · QC pass % · headline stat · last run · cadence. KPI strip is always visible above the tabs.

Tabs: **Summary · Results · SPC · Trends · Compare · Envelope · Reports · Settings**.

### 5.2 Summary

Campaign-mean `MetricCard`s for the test type's key measurements (±u, rel %); latest-runs list with QC chips; a note surfacing anything actionable elsewhere (e.g. an SPC violation).

### 5.3 Results

Dense table: run · date · serial · each measurement as `value ± u` · QC chip. Right-aligned tabular numerals; hover rows; row click → run's current analysis (Phase 3: run detail). Headers keep their written case (`Pc · bar`, `ηc* · %`).

### 5.4 SPC

Layout: chart (I above MR, shared x) | sidebar (parameter select · spec limits · capability · violations).

- **I-chart**: CL = μ; UCL/LCL = μ ± 2.66·MR̄ (dashed, labelled with values); USL/LSL as amber dashed lines (labelled); series line + points; **violating points enlarged, red, labelled with the rule ID** (`R1`). Port the full Western Electric rule set (R1–R6) from `core/spc.py` — the prototype shows R1 only.
- **MR chart**: moving ranges, CL = MR̄, UCL = 3.267·MR̄.
- **Sidebar**: parameter select (any campaign measurement); **editable USL/LSL** — editing recomputes chart + capability live and persists to the campaign (toast: `USL updated (campaign setting)`); capability card (μ, σ = MR̄/1.128, Cp, Cpk — colored ≥1.33 good / ≥1.0 warn / else bad); violations list (`R1 · run 11 · 1455.2 outside 3σ`) — each entry links to that run's analysis.

### 5.5 Trends

Parameter over run sequence with **k=2 uncertainty bars** (toggleable) and a dashed linear **drift fit** (toggleable, series-2 orange). Sidebar reads the fit back in words: slope (unit/run), total drift, relative %, and the verdict sentence — **"vs. per-run uncertainty: drift (not) significant"**. The sentence is the feature; the chart supports it.

### 5.6 Compare

- Mode: `Run vs Run` | `Run vs Golden`. Run selects; golden shows the stored reference (`★ golden` chip) or a prompt to set one.
- `★ Set A as golden` persists run A's measurement set as the campaign's golden reference (a real record, not session state).
- Table per metric: A ±u · B ±u · Δ · Δ% · **verdict chip** — `within uncertainty` (good) or `significant` (warn), by |Δ| > k·√(u²ᴀ+u²ʙ), k = 2. The rule is stated in a note under the table. Δ column text turns warn when significant.

### 5.7 Envelope

O/F–Pc scatter (axes fixed per test type for Phase 2): **target box** (green translucent fill, dashed border, `target` label) editable in the sidebar and persisted as a campaign setting; points inside = series blue, outside = amber **with run label**; sidebar coverage `13 / 14 in target box`. Hover: run · O/F ±u · Pc ±u · inside/outside.

### 5.8 Reports

Generate buttons (Campaign report · HTML, Qualification package · ZIP, Data export · CSV) with an include-uncertainties/traceability toggle; **Generated** table (artifact · format · when · by · download). Artifacts are server-generated and stored (`reports` table) — links are permanent. Generation is a job; the row appears when done.

### 5.9 Settings (campaign-manager role)

- **Headline stat** select (same options as the dashboard ⚙).
- **Spec limits** per measurement (feeds SPC defaults).
- **Metadata template** — a **dynamic key/value editor** (`KeyValueEditor`): add/rename/remove any field. Rows whose key matches the test type plugin's metadata fields show `✓ analysis` (they feed the readiness resolver); all others show `custom` and simply travel with every record. Badges update live as keys are typed; per-run fields (serial, operator notes) are explicitly out of the template (explained in the footnote). Fixed forms are wrong here — R4 decision.
- **Auto-assign rule** (match expression for inbox suggestions).
- Single `Save campaign settings` (labelled `campaign-manager only · audited`).

---

## 6. Chart rendering contract

- All statistical charts consume **result data** (measurements, campaign tables) — small, no decimation.
- Time-series panels in the analysis screen consume `GET /runs/{id}/series?tags&t0&t1&px` — server-decimated to the panel's pixel width; zoom refetches. Numbers on screen (window stats, QC, results) are **never computed from decimated data**.
- Event markers: dashed verticals; labels in the top margin in **two staggered rows** (close events collide otherwise — R1 finding); FIRE emphasized in red.
- Steady region overlay: translucent accent fill, 2 px edges, mid-height grab handles; drawn on every panel, synced.
- Grid lines ~5% white; axis text 10 px mono muted; y-ticks left, x-ticks bottom.
- Error bars at k=2 with end caps, series color at 50% alpha.

---

## 7. Schema/API deltas discovered by the UX track

Already folded into `DESIGN_WEB_REBUILD.md`: `user_programs` (subscriptions), `campaigns.headline_metric`, `campaigns.metadata_template`, `campaigns.settings` (spec limits per measurement + envelope target box), golden references as records, program-scoped run/inbox filters on `GET /api/runs`.

---

## 8. Definition of done (Phase 2 UI)

1. Happy path ≤ 10 interactions, measured the prototype's way (deliberate actions, not navigation).
2. The invalidation rule (§4.2) holds under adversarial clicking — no stale result can be displayed or saved.
3. Window-drag stats round-trip < 100 ms warm (perceptually instant).
4. Every number rendered anywhere carries its uncertainty or is visibly a count/identity.
5. All §2.4 rules pass review; keyboard path works end-to-end without a mouse (except window drag, which has numeric-field parity).
6. Playwright E2E: inbox → assign → analyze → QC (incl. one warning) → save → report download, against the compose stack.
