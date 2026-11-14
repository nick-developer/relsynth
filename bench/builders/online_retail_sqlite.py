# SPDX-License-Identifier: Apache-2.0
"""Build a SQLite star schema from UCI Online Retail II (CC BY 4.0, Chen 2012) (G7).

    python online_retail_sqlite.py online_retail_ii.zip OUT.db

The release is one flat sheet per year (2009-2010, 2010-2011) of invoice lines.
The builder splits it into the tables an order system would hold, keeping every
messy value as it is (cancellations "C...", negative quantities, missing
customers, zero prices, free-text descriptions that vary per stock code):

* customer(customer_id, country): the country of the customer's first line
* product(stock_code, description): the most frequent description
* invoice(invoice, invoice_date, customer_id NULL): one row per invoice
* invoice_line(line_id, invoice, stock_code, quantity, price)
"""
import io
import sqlite3
import sys
import zipfile
from pathlib import Path

import pandas as pd
from sqlite_ddl import create_table

src, out = Path(sys.argv[1]), Path(sys.argv[2])
with zipfile.ZipFile(src) as z:
    name = next(n for n in z.namelist() if n.endswith(".xlsx"))
    sheets = pd.read_excel(io.BytesIO(z.read(name)), sheet_name=None, dtype={"Invoice": str, "StockCode": str})
df = pd.concat(sheets.values(), ignore_index=True)
df["Customer ID"] = df["Customer ID"].astype("Int64")
df = df.sort_values(["InvoiceDate", "Invoice"], kind="stable").reset_index(drop=True)
customer = df.dropna(subset=["Customer ID"]).groupby("Customer ID", sort=True)["Country"].first().reset_index()
product = (df.dropna(subset=["Description"]).groupby(["StockCode", "Description"]).size().reset_index(name="n")
           .sort_values(["StockCode", "n", "Description"], ascending=[True, False, True])
           .drop_duplicates("StockCode"))
product = pd.DataFrame({"StockCode": sorted(df["StockCode"].unique())}).merge(product, how="left")
invoice = df.groupby("Invoice", sort=False).agg(InvoiceDate=("InvoiceDate", "min"),
                                                CustomerID=("Customer ID", "first")).reset_index()
invoice["InvoiceDate"] = invoice["InvoiceDate"].dt.strftime("%Y-%m-%d %H:%M:%S")
line = df[["Invoice", "StockCode", "Quantity", "Price"]].copy()
line.insert(0, "line_id", range(1, len(line) + 1))

out.unlink(missing_ok=True)
con = sqlite3.connect(out)
create_table(con, "customer", [("customer_id", "INTEGER", True), ("country", "TEXT", False)], ["customer_id"])
create_table(con, "product", [("stock_code", "TEXT", True), ("description", "TEXT", False)], ["stock_code"])
create_table(con, "invoice", [("invoice", "TEXT", True), ("invoice_date", "DATETIME", True),
                              ("customer_id", "INTEGER", False)], ["invoice"],
             foreign_keys=[(["customer_id"], "customer", ["customer_id"])])
create_table(con, "invoice_line", [("line_id", "INTEGER", True), ("invoice", "TEXT", True),
                                   ("stock_code", "TEXT", True), ("quantity", "INTEGER", True),
                                   ("price", "REAL", True)], ["line_id"],
             foreign_keys=[(["invoice"], "invoice", ["invoice"]), (["stock_code"], "product", ["stock_code"])])
customer.set_axis(["customer_id", "country"], axis=1).to_sql("customer", con, if_exists="append", index=False)
product[["StockCode", "Description"]].set_axis(["stock_code", "description"], axis=1).to_sql(
    "product", con, if_exists="append", index=False)
invoice.set_axis(["invoice", "invoice_date", "customer_id"], axis=1).to_sql("invoice", con, if_exists="append",
                                                                            index=False)
line.set_axis(["line_id", "invoice", "stock_code", "quantity", "price"], axis=1).to_sql(
    "invoice_line", con, if_exists="append", index=False, chunksize=50_000)
con.commit()
con.close()
print(f"customer {len(customer)}, product {len(product)}, invoice {len(invoice)}, invoice_line {len(line)}")
