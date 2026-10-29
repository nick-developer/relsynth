# Test data

## `specs_v1/`

relsynth 1.0 specs of the four golden real databases, copied from the layer 2 runs (`realdata/` in the project
files): Chinook (two drafts), Contoso V2 (SQLite and Parquet), AdventureWorks DW (v0.5 and v0.7 drafts) and the
bi/COREP_LR reporting schema (five versions, one of them an introspect draft that does not validate). They hold
table and column names, keys, rules and settings only, no rows. `tests/unit/test_ir.py` converts each to the IR
and back. The source path of `adventureworks.yaml` was shortened to `aw.db` (it pointed into a scratch folder).

## `standards/`

Published schemas used to check what relsynth writes. Both are Apache License 2.0.

| File | Source |
|---|---|
| `odcs-json-schema-v3.2.0.json` | Open Data Contract Standard v3.2.0, `schema/odcs-json-schema-latest.json` of https://github.com/bitol-io/open-data-contract-standard (Bitol, LF AI & Data) |
| `openlineage/OpenLineage.json` (2-0-2) and the five `*DatasetFacet.json` files | `spec/` of https://github.com/OpenLineage/OpenLineage (LF AI & Data) |
