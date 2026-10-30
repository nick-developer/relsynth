# Tables larger than memory: fitting inside the database

A data warehouse's fact tables can hold billions of rows, while its dimensions are usually small. relsynth therefore fits large facts from statistics that the database computes, and loads only what the model needs row by row. This is called *pushdown*.

## What is pushed down

The planner decides per table:

| Table | Fitted from |
|---|---|
| Entities: dimensions, customers, products, any table other tables reference | All rows, loaded. Each row carries its own latent segment, whose posterior needs the row (Prop. 105) |
| A fact that no other table references, without type 2 history | **Statistics computed in the database plus a uniform sample** |
| A fact that other tables reference (the header of a nested fact, the parent of a link) | All rows, loaded (it is an entity of its own rows) |
| A fact with type 2 history (`scd`) | All rows, loaded |
| A fact from which an aggregate column of another table is computed, when the source does not store that column | All rows, loaded (the column is recomputed from the rows) |

Nested facts (lines inside a parent fact), facts split into order headers, bases of views and facts with running balances are all eligible.

```bash
relsynth plan spec.yaml --pushdown on        # AdventureWorks DW (excerpt)
```

```
pushdown mode on; samples of 500,000 rows
  DimCustomer                  18,484 rows  loaded
  DimProduct                      606 rows  loaded
  FactCurrencyRate             14,264 rows  statistics in the source + sample
  FactInternetSales            60,398 rows  loaded (other tables reference it)
  FactInternetSalesReason      64,515 rows  statistics in the source + sample
  FactProductInventory        776,286 rows  statistics in the source + sample
  FactResellerSales            60,855 rows  statistics in the source + sample
```

FactInternetSales stays loaded because FactInternetSalesReason refers to its order lines.

## Settings

```yaml
synthesis:
  pushdown:
    mode: auto          # auto | on | off
    min_rows: 1000000   # auto: push down facts with at least this many rows
    sample_rows: 500000 # rows sampled for the column models of each pushed fact
```

| Setting | Meaning |
|---|---|
| `mode: auto` (default) | Push down eligible facts with at least `min_rows` rows |
| `mode: on` | Push down every eligible fact, whatever its size |
| `mode: off` | Load every table (the v0.9 behaviour) |
| `sample_rows` | Larger samples give tighter column models and take more memory. Above about 200,000 rows per fact the scorecard rarely changes |

On the command line, `relsynth run spec.yaml --pushdown on` overrides the mode; in Python, use `rs.fit(spec, pushdown="on")`.

## What the database computes

For a pushed fact with key columns $k_1..k_m$ (its links and composite-key parts) and continuous columns $c$, the database computes the following:

1. `COUNT(*)` grouped by every key tuple the model uses, such as the pair of linked dimensions or the composite key with its links. A null key value forms its own group.
2. For up to three continuous columns, the sums over each key tuple of the **normal scores** $z=\Phi^{-1}(r/(N+1))$ and of their squares. Here $r$ is the value's average rank among the $N$ non-null values, computed with `RANK()` and `COUNT()` window functions (the average rank of a tie is its first rank plus half the tie's size minus one). These are the rank features that segment discovery uses.
3. For order headers, the first row of every document key in primary-key order (`ROW_NUMBER()`) and the number of cells that differ from it. For running balances, the first row of every group in the rule's order, the non-null count and the minimum.
4. `COUNT(*)`, non-null counts and the distinct referenced values of each key.
5. A seeded, uniform Bernoulli sample of `sample_rows` rows for the column models.

Proposition 100 in [proofs/relsynth-proofs.pdf](proofs/relsynth-proofs.pdf) shows that items 1–4 are everything the relational model needs from the fact. Proposition 104 shows that the column models fitted on the sample are an exact posterior given the sampled rows, because the sample's inclusion does not depend on the values. Synthetic table sizes follow the exact row count, not the sample's.

### Per engine

| Engine | $\Phi^{-1}$ | Row hash for the sample | Notes |
|---|---|---|---|
| PostgreSQL | Acklam's approximation in SQL (relative error < 1.15e-9) | `hashtextextended(text, seed)` | |
| SQL Server | Acklam | `HASHBYTES('SHA2_256', ...)` | the seed and algorithm are inlined as literals: HASHBYTES returns NULL for some bound parameters |
| Oracle | Acklam | `ORA_HASH(..., 4294967295, seed)` | |
| MySQL, MariaDB | Acklam | first 32 bits of `SHA2('seed\|key', 256)` | ranks are taken in `DOUBLE` (`x + 0E0`): MySQL divides integers as `DECIMAL` with 4 digits. Text keys are grouped and filtered in the binary collation of their character set, because the default collations ignore case, accents and (MariaDB) trailing spaces |
| SQLite | exact (`scipy.special.ndtri`, registered function) | BLAKE2b (registered function) | |
| Databricks SQL | Acklam | first 32 bits of `sha2('seed\|key', 256)`, as on MySQL | ranks are taken in `DOUBLE` (`FLOAT` is 4 bytes). Text compares byte by byte, so no collation is needed |
| CSV, Parquet (`engine: duckdb`, the default) | exact (`scipy.special.ndtri` as a vectorised DuckDB function) | first 32 bits of SHA-256, as on MySQL | DuckDB runs the same SQL over views of the files and reads only the columns a query needs. CSV columns are typed by the pandas reader's rule. First rows follow the file order |
| CSV, Parquet (`engine: pandas`) | exact, in pandas | a seeded Bernoulli draw while streaming | the 1.0 path: only the key and measure columns are read, one statistic at a time |

Long key lists are handled in two ways. A filter on more than 2,000 values (`MAX_IN_LIST`) is not sent as an `IN (...)` list; the query groups by the filtered column as well, and the filter is applied to the grouped result. Filters on another table's keys use a semi-join, so the database never receives a long literal list.

## Guarantees and checks

| Claim | Proposition | Test |
|---|---|---|
| The SQL statistics equal the in-memory ones (counts exact; score sums within 1e-8) | 100, 102 | `test_counts_and_scores_match`, `test_statistics_match_the_in_memory_reference[postgresql, mssql, oracle]` |
| A sample of every row gives exactly the in-memory synthetic database (plain, nested, header-split, view base, running balance) | 101 | `test_full_sample_pushdown_equals_in_memory[...]` |
| The same source on SQLite, PostgreSQL, SQL Server, Oracle, MySQL, MariaDB and Databricks gives the same synthetic database | 101 | `test_full_sample_pushdown_matches_sqlite_in_memory[...]` |
| Text statistics are exact under case-insensitive collations (MySQL, MariaDB) | 100 | `test_text_is_compared_exactly_under_a_case_insensitive_collation[...]` |
| Samples are uniform, seeded and of the requested size | 103 | `test_sample_is_uniform_reproducible_and_sized`, `test_python_hash_is_uniform_and_seeded`, `test_sample_hash_is_sha256_of_the_key[mysql, mariadb]` |
| A sampled fit keeps the scorecard and uses exact counts on the real side | 104 | `test_sampled_pushdown_keeps_quality_and_memory_bound` |

## Real databases: in memory against pushdown

Each database was run twice with the final v1.0 code and the same seed: once with every table loaded, once with every eligible fact pushed down (`mode: on`, `min_rows: 0`). The scorecard is the full one without privacy checks; the domain checks are each database's own ground-truth checks from layer 2 (rules, key structure, business invariants), compared real against synthetic.

| Database | Source | Pushed facts | Scorecard in memory | Scorecard pushdown | Domain checks |
|---|---|---|---|---|---|
| EBA COREP reporting (bi) | CSV, 68 k rows (8 tables) | `FACT_ReportedValue` (65,272 rows; 40,000-row sample) | 177/178 | 178/178 | B1–B8 identical; B6 median per-cell KS 0.10 in memory, 0.13 pushed |
| Chinook | SQLite, 15.6 k rows (11 tables) | `InvoiceLine`, `PlaylistTrack` | 144/146 | 144/146 | the same two `InvoiceLine.TrackId` cardinality failures in both |
| Contoso V2 100k | Parquet, 753 k rows | `orderrows` (223,974 rows; 40,000-row sample) | 253/253 | 253/253 | T1–T17 within 0.015 of each other |
| AdventureWorks DW | SQLite, 1.05 M rows | 10 facts, including the 776,286-row inventory with its running balance (30,000-row samples) | 1229/1286 | 1237/1285 | G1–G13 equal, except lines per order (FIS 2.144 vs 2.187, real 2.184) and lines without a sales reason (0.123 vs 0.122, real 0.106) |

What the remaining failures are:
- **bi**: one rows-per-cell distribution (`FACT_ReportedValue.CellKey`), at the limit (KS 0.18 against 0.12) in memory; it passed with the sample.
- **Chinook**: the distribution of invoice lines per track, just over its limits (KS 0.138 against 0.12; mean error 0.117 against 0.10).
- **AdventureWorks** (57 in memory): 22 are combinations of reference keys that never occur together in the source (a currency with a product subcategory, a territory with a reseller's geography); the rest are dependence and join checks (the finance and sales-quota facts, wide product and customer dimensions), two repeated-pair shares, four rows-per-parent checks and one unique key (`DimProduct` alternate key with start date, 5% duplicates). The in-memory count equals v0.9's (1229/1286).
- **Contoso domain checks** that fall short in both modes are long-standing limits, the same as in v0.6: the share of physical-store sales in the customer's own geography area (T5), one price factor per category and day (T11) and the customer's age as whole years to one date (T14).

The in-memory runs give the same scorecards as v0.9 (AdventureWorks 1229/1286, Contoso 253/253, Chinook 144/146).

## Performance

**Memory.** A loaded table costs about rows × columns × 8 bytes, plus pandas overhead for text. A pushed fact costs its distinct key tuples (a few numbers each) plus the sample. The engine's own model (segments, rates) grows with the number of entity rows, not with fact rows.

**Time.**
- The database does the grouping, so a pushed fact costs a handful of `GROUP BY` queries, each one scan of the table.
- On SQL servers, give the database indexes on the foreign-key columns of large facts. They usually exist already, and grouped counts then read the indexes rather than the table.
- On files, DuckDB computes the statistics with the same SQL as the servers, using streaming, multi-threaded operators that spill to disk when memory runs short (`source.duckdb.temp_directory`). Measurements are under *Files: DuckDB against pandas* below. With `engine: pandas`, statistics are computed in pandas one column group at a time (the 1.0 path), so each statistic re-reads its columns.

**Other knobs.**

| Knob | Effect |
|---|---|
| `relsynth[speed]` | numba kernels for the Gibbs sweeps (same output; faster on schemas with many segments and links) |
| `synthesis.scale` | the synthetic size; sampling is linear in it |
| `--no-privacy` | skips the distance-to-closest-record checks, which compare every synthetic row with the source (the slowest part of evaluation on wide tables) |
| `--no-write` | scores without writing (for tuning) |
| `synthesis.posterior.datasets` | each extra dataset is one more sampling pass, but no more fitting |
| `target.if_exists: truncate` with `structure: existing` | loads into prepared tables without DDL, which suits CI against a shared test database |

**Measured** (one 4-core machine, SQLite source, writes to CSV, scorecard without privacy checks):

| Database | Rows | Mode | Wall time | Peak memory | Scorecard |
|---|---|---|---|---|---|
| AdventureWorks DW | 1.05 M (27 tables) | in memory | 803 s | 2.1 GiB | 1229/1286 |
| AdventureWorks DW | | pushdown (10 facts, 30,000-row samples) | 668 s | 1.6 GiB | 1237/1285 |
| Contoso V2 100k (Parquet) | 753 k (8 tables) | in memory | 269 s | 1.4 GiB | 253/253 |
| Contoso V2 100k (Parquet) | | pushdown (order lines, 40,000-row sample) | 241 s | 1.3 GiB | 253/253 |
| EBA COREP (CSV) | 70 k (9 tables) | in memory | 140 s | 425 MiB | 177/178 |
| EBA COREP (CSV) | | pushdown (1 fact, 40,000-row sample) | 126 s | 407 MiB | 178/178 |

### Files: DuckDB against pandas

The statistics workload of a Contoso fit (13 queries on `orderrows`) was replayed, each engine in its own process, on Contoso's own fact (224 k rows) and on the same fact repeated 100 times with new keys (22.4 M rows, 234 MB of Parquet). "Load" reads the whole fact into pandas, as a fit without pushdown does, and computes nothing.

| Fact | Engine | Time | Peak memory |
|---|---|---|---|
| 22.4 M rows | load whole (no pushdown) | 143 s | 13.1 GB |
| | pandas pushdown (1.0) | 219 s | 4.4 GB |
| | DuckDB pushdown | 18 s | 1.7 GB |
| 224 k rows | load whole | 0.8 s | 324 MiB |
| | pandas pushdown | 1.4 s | 309 MiB |
| | DuckDB pushdown | 1.3 s | 281 MiB |

DuckDB's results are fetched as columnar frames, not as Python row tuples. Fetched as tuples, the same DuckDB run took 205 s and 6.3 GB. The default settings were best: a 1 GB or 3 GB `memory_limit` did not lower the peak further (1.66 and 1.65 GB), because the peak is relsynth's own frames.

Whole Contoso fits from Parquet (fit and sample in one process):

| Mode | Engine | Fit | Peak memory | Scorecard |
|---|---|---|---|---|
| in memory | pandas | 201 s | 1,377 MiB | 253/253 |
| full-sample pushdown | pandas | 194 s | 1,384 MiB | 253/253 |
| full-sample pushdown | DuckDB | 187 s | 1,501 MiB | 253/253 |
| 40,000-row sample | pandas | 179 s | 1,285 MiB | 253/253 |
| 40,000-row sample | DuckDB | 173 s | 1,322 MiB | 253/253 |

With a sample of every row, both engines give a synthetic database identical to the in-memory one, table by table (Proposition 101). At Contoso's size the fact is a small part of memory and DuckDB adds about 100 MiB of its own. The engine pays off once the fact is larger than the engine's model, as the 22.4 M-row replay shows.

These two databases fit in memory, so they show that the results agree, not how far pushdown scales. For a fact of N rows and p columns with G distinct key tuples, memory grows with G + sample_rows instead of N × p.
