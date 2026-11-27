# SPDX-License-Identifier: Apache-2.0
"""IBM Transactions for Anti-Money Laundering (IT-AML, Altman et al. 2023; CDLA-Sharing-1.0), one
variant's *_Trans.csv as a four-table Parquet source.

    python aml_parquet.py TRANS.csv OUTDIR

The CSV is one flat file of transfers. The tables follow the dataset's entity model:

- `bank.parquet`: bank_id (text: codes keep their leading zeros, "010" is not "10");
- `account.parquet`: account_key (surrogate, 1..n in bank_id, account_id order), bank_id, account_id.
  An account id is unique only within its bank, so the key is the pair; the surrogate lets a fact
  reference it with one column;
- `currency.parquet`: currency (the names of both currency columns), usd_rate (units per US Dollar: the
  rate the simulator converts at, read from the cross-currency payments);
- `trans.parquet`: trans_id (file row order: the CSV has no key), ts (minute), from_bank, from_account_key,
  to_bank, to_account_key, amount_received, receiving_currency, amount_paid, payment_currency,
  payment_format, is_laundering. Every value is as published; duplicate rows stay.

Written with DuckDB, so the five million rows never pass through pandas.
"""
import sys
from pathlib import Path

import duckdb

src, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)


def lit(p) -> str:
    return "'" + str(p).replace("'", "''") + "'"


con = duckdb.connect()
# both account columns are named "Account" in the header: read by position, every value as text
con.execute(f"""CREATE TEMP VIEW raw AS SELECT row_number() OVER () AS trans_id, * FROM read_csv({lit(src)},
    header = true, all_varchar = true, names = ['ts', 'fb', 'fa', 'tb', 'ta', 'ar', 'rc', 'ap', 'pc', 'pf', 'lab'])""")
con.execute("CREATE TEMP TABLE acct AS SELECT row_number() OVER (ORDER BY b, a) AS account_key, b AS bank_id, "
            "a AS account_id FROM (SELECT fb AS b, fa AS a FROM raw UNION SELECT tb, ta FROM raw)")
con.execute(f"COPY (SELECT DISTINCT bank_id FROM acct ORDER BY bank_id) TO {lit(out / 'bank.parquet')} (FORMAT parquet)")
con.execute(f"COPY (SELECT * FROM acct ORDER BY account_key) TO {lit(out / 'account.parquet')} (FORMAT parquet)")
# the simulator converts at one rate per currency for the whole period, and the rates of all pairs are
# consistent through the US Dollar (A->B = usd(B) / usd(A) within 0.2 %, rounding of small amounts):
# usd_rate = units per US Dollar, the median rate of the cross-currency payments of at least 100 units
# that start or end in US Dollar (6 significant digits)
con.execute("""CREATE TEMP TABLE ccy AS WITH c AS (SELECT DISTINCT x AS currency FROM (SELECT rc AS x FROM raw
        UNION SELECT pc FROM raw)),
    fwd AS (SELECT rc AS currency, median(ar::DOUBLE / ap::DOUBLE) AS r FROM raw
        WHERE pc = 'US Dollar' AND rc <> pc AND ap::DOUBLE >= 100 GROUP BY 1),
    rev AS (SELECT pc AS currency, median(ap::DOUBLE / ar::DOUBLE) AS r FROM raw
        WHERE rc = 'US Dollar' AND rc <> pc AND ar::DOUBLE >= 100 GROUP BY 1)
    SELECT c.currency, CASE WHEN c.currency = 'US Dollar' THEN 1.0 ELSE
        (SELECT round(v, 5 - floor(log10(v))::INTEGER) FROM (SELECT coalesce(fwd.r, rev.r) AS v)) END AS usd_rate
    FROM c LEFT JOIN fwd USING (currency) LEFT JOIN rev USING (currency)""")
missing = con.execute("SELECT list(currency) FROM ccy WHERE usd_rate IS NULL").fetchone()[0]
if missing:
    raise SystemExit(f"no US Dollar rate for {missing}")
con.execute(f"COPY (SELECT * FROM ccy ORDER BY currency) TO {lit(out / 'currency.parquet')} (FORMAT parquet)")
con.execute(f"""COPY (SELECT r.trans_id, strptime(r.ts, '%Y/%m/%d %H:%M') AS ts,
        r.fb AS from_bank, f.account_key AS from_account_key, r.tb AS to_bank, t.account_key AS to_account_key,
        r.ar::DOUBLE AS amount_received, r.rc AS receiving_currency, r.ap::DOUBLE AS amount_paid,
        r.pc AS payment_currency, r.pf AS payment_format, r.lab::INTEGER AS is_laundering
    FROM raw r JOIN acct f ON f.bank_id = r.fb AND f.account_id = r.fa
    JOIN acct t ON t.bank_id = r.tb AND t.account_id = r.ta ORDER BY r.trans_id)
    TO {lit(out / 'trans.parquet')} (FORMAT parquet, COMPRESSION zstd)""")
for t in ("bank", "account", "currency", "trans"):
    print(t, con.execute(f"SELECT count(*) FROM read_parquet({lit(out / (t + '.parquet'))})").fetchone()[0], "rows")
