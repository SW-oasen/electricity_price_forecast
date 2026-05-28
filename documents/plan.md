Created memory file [](file:///c%3A/Users/seewi/AppData/Roaming/Code/User/workspaceStorage/231765b78765c0c2ab83dcb60c835917/GitHub.copilot-chat/memory-tool/memories/NDdiYjBmM2UtYzcwNi00YWNlLTgyMmYtNzAyOGNlY2JjYzAz/plan.md)

## Plan: Price Data Expansion Foundation

Your PROJECT_CONTEXT direction is strong. The main improvement is to stop growing the current demand-specific wide schema and move to a normalized multi-series time-series backbone before adding many new sources.

TL;DR: stabilize data architecture first, then ingest in waves, then build leakage-safe price features and benchmarks.

**Steps**
1. Phase 1: Freeze the data contract for price forecasting.
Define target as DE/LU day-ahead hourly price, forecast horizon as D+1, and canonical time policy (store UTC, render Berlin). This prevents later ETL/model drift.
2. Phase 1: Build a complete series catalog from your planned predictors.
For each series: source, SMARD filter/region/resolution, unit, expected start date, refresh cadence, quality bounds.
3. Phase 2: Refactor storage to normalized structure.
Use a shared timestamp axis + series catalog + fact table (time, series_id, value, version, source, fetched_at). Keep legacy tables/views temporarily for compatibility.
4. Phase 2: Upgrade ingestion to idempotent multi-series orchestration.
Support historical backfill windows and daily incremental updates with retry, per-series failure isolation, and resume capability.
5. Phase 2: Add quality gates before data is promoted.
Checks: completeness, null ratio, duplicate timestamps, value bounds, timezone/DST continuity, freshness SLA.
6. Phase 3: Expand data in prioritized waves.
Wave A: 4169 + neighbor prices + 410/411 + residual + core wind/solar (+ generation forecasts).
Wave B: remaining generation mix and optional control-area detail.
Wave C: external signals (fuel/commodity, import/export, capacity proxies).
7. Phase 3: Move feature engineering to training-time builders.
Generate lags/rolling/interactions by series family with strict cutoff logic to avoid leakage in day-ahead setup.
8. Phase 4: Establish benchmark/evaluation protocol.
Run rolling-origin backtests against naive baselines, your ML baseline, and official references where available.
9. Phase 4: Operationalize.
Set daily jobs, backfill jobs, run logs, quality dashboard, and alert thresholds.

**Relevant files**
- PROJECT_CONTEXT.md
- smard_api.md
- config.py
- smard_client.py
- etl.py
- fetch_prepare_data.py
- streamlit_app_etl.py

**Verification**
1. Catalog completeness: every predictor in PROJECT_CONTEXT is mapped to a concrete source identifier and ingestion method.
2. Backfill reproducibility: one-year replay yields deterministic row counts and no duplicate keys.
3. Incremental idempotency: second run on same interval inserts zero rows unless source changed.
4. DST robustness: transition weeks are continuous in UTC and correct in Berlin-local views.
5. QA enforcement: synthetic anomaly injections are rejected and logged.
6. Modeling readiness: generated feature matrix is leakage-free at prediction cutoff.

**Decisions**
- Included now: data architecture, ingestion robustness, quality governance, feature-prep strategy.
- Deliberately deferred: heavy model tuning until data foundation is stable.
- Assumed: primary target remains DE/LU day-ahead hourly price.

I also saved this as the working plan in /memories/session/plan.md so we can iterate it quickly in the next step.