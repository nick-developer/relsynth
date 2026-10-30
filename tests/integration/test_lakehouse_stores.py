# SPDX-License-Identifier: Apache-2.0
"""Iceberg and Delta Lake tables on object storage: Amazon S3 (any S3-compatible
endpoint; CI and docs/testing.md use moto's server) and Azure Blob Storage /
ADLS Gen2 (Azurite).  Credentials come from environment variables through
`storage_env`, never from the spec.  Each store gets the connector contract and
the decisive check: the scenario copied into the store gives the same
synthetic database as the in-memory source.

    RELSYNTH_TEST_S3_ENDPOINT=http://127.0.0.1:5555        (moto_server -p 5555, or MinIO with
                                                           RELSYNTH_TEST_S3_KEY / RELSYNTH_TEST_S3_SECRET)
    RELSYNTH_TEST_AZURITE_ENDPOINT=http://127.0.0.1:10000  (azurite-blob --blobPort 10000)
"""
from __future__ import annotations

import copy
import os
import uuid

import numpy as np
import pandas as pd
import pytest
from conftest import write_scenario

import relsynth as rs
from relsynth.connectors import from_spec

pytest.importorskip("pyiceberg")
pytest.importorskip("deltalake")

AZ_ACCOUNT = "devstoreaccount1"
# Azurite's published development key (https://learn.microsoft.com/azure/storage/common/storage-use-azurite)
AZ_KEY = "Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw=="  # nosec B105


def _s3(kind, tmp_path):
    ep = os.environ.get("RELSYNTH_TEST_S3_ENDPOINT")
    if not ep:
        pytest.skip("RELSYNTH_TEST_S3_ENDPOINT not set: no S3 endpoint")
    boto3 = pytest.importorskip("boto3")
    key = os.environ.get("RELSYNTH_TEST_S3_KEY", "test")
    os.environ.setdefault("RELSYNTH_TEST_S3_SECRET", "test")
    bucket = f"rs-{uuid.uuid4().hex[:10]}"
    boto3.client("s3", endpoint_url=ep, aws_access_key_id=key, aws_secret_access_key=os.environ["RELSYNTH_TEST_S3_SECRET"],
                 region_name="us-east-1").create_bucket(Bucket=bucket)
    if kind == "delta":
        return {"kind": "delta", "path": f"s3://{bucket}/lake",
                "storage": {"AWS_ENDPOINT_URL": ep, "AWS_ACCESS_KEY_ID": key, "AWS_REGION": "us-east-1",
                            "AWS_ALLOW_HTTP": "true"},
                "storage_env": {"AWS_SECRET_ACCESS_KEY": "RELSYNTH_TEST_S3_SECRET"}}
    return {"kind": "iceberg", "schema": "test",
            "catalog": {"type": "sql", "uri": f"sqlite:///{tmp_path / 'catalog.db'}", "warehouse": f"s3://{bucket}/lake"},
            "storage": {"s3.endpoint": ep, "s3.access-key-id": key, "s3.region": "us-east-1"},
            "storage_env": {"s3.secret-access-key": "RELSYNTH_TEST_S3_SECRET"}}


def _azure(kind, tmp_path):
    ep = os.environ.get("RELSYNTH_TEST_AZURITE_ENDPOINT")
    if not ep:
        pytest.skip("RELSYNTH_TEST_AZURITE_ENDPOINT not set: no Azurite")
    blob = pytest.importorskip("azure.storage.blob")
    conn = f"DefaultEndpointsProtocol=http;AccountName={AZ_ACCOUNT};AccountKey={AZ_KEY};BlobEndpoint={ep}/{AZ_ACCOUNT};"
    container = f"rs{uuid.uuid4().hex[:10]}"
    blob.BlobServiceClient.from_connection_string(conn).create_container(container)
    os.environ["RS_TEST_AZ_KEY"] = AZ_KEY
    os.environ["RS_TEST_AZ_CONN"] = conn
    if kind == "delta":
        return {"kind": "delta", "path": f"az://{container}/lake",
                "storage": {"AZURE_STORAGE_ACCOUNT_NAME": AZ_ACCOUNT, "AZURE_STORAGE_USE_EMULATOR": "true",
                            "AZURE_STORAGE_ALLOW_HTTP": "true"},
                "storage_env": {"AZURE_STORAGE_ACCOUNT_KEY": "RS_TEST_AZ_KEY"}}
    return {"kind": "iceberg", "schema": "test",
            "catalog": {"type": "sql", "uri": f"sqlite:///{tmp_path / 'catalog.db'}",
                        "warehouse": f"abfs://{container}@{AZ_ACCOUNT}.dfs.core.windows.net/lake"},
            "storage": {"adls.account-name": AZ_ACCOUNT}, "storage_env": {"adls.connection-string": "RS_TEST_AZ_CONN"}}


STORES = {"s3": _s3, "azure": _azure}


@pytest.mark.parametrize("store", list(STORES))
@pytest.mark.parametrize("kind", ["iceberg", "delta"])
def test_contract_on_object_storage(store, kind, tmp_path):
    sec = STORES[store](kind, tmp_path)
    df = pd.DataFrame({"id": np.arange(2000), "v": np.random.default_rng(1).normal(size=2000).round(3),
                       "c": [f"{i % 7:03d}" for i in range(2000)]})
    spec = {"tables": {"t": {"primary_key": "id"}}, "relationships": []}
    from_spec(sec).write_tables(spec, {"t": df})
    c = from_spec({**sec, "include": ["t"]})
    assert c.has_table("t") and c.count("t") == 2000 and c.describe("t").primary_key == ["id"]
    pd.testing.assert_frame_equal(c.read_table("t"), df, check_dtype=False)
    st = c.stats("t")
    assert st.n_rows == 2000 and st.group_counts(["c"])["n"].sum() == 2000
    s1, whole = c.sample("t", 500, seed=3)
    assert not whole and sorted(s1["id"]) == sorted(from_spec(sec).sample("t", 500, seed=3)[0]["id"])
    assert "s3cr3t" not in repr(c) and AZ_KEY not in repr(c)


@pytest.mark.slow
@pytest.mark.parametrize("store", list(STORES))
@pytest.mark.parametrize("kind", ["iceberg", "delta"])
def test_same_synthetic_database_from_object_storage(store, kind, tmp_path):
    sec = STORES[store](kind, tmp_path)
    lite = write_scenario("S9_small_shop", {"kind": "sqlite", "path": str(tmp_path / "ref.db")})
    lake = write_scenario("S9_small_shop", sec)
    lake["source"] = {**lake["source"], "include": list(lite["tables"])} if kind == "delta" else lake["source"]
    ref = rs.fit(copy.deepcopy(lite), seed=3).sample(dataset=0)
    spec = copy.deepcopy(lake)
    spec["synthesis"] = {**spec.get("synthesis", {}), "pushdown": {"mode": "on", "min_rows": 0, "sample_rows": 10**7}}
    got = rs.fit(spec, seed=3).sample(dataset=0)
    for t in ref:
        pd.testing.assert_frame_equal(ref[t].reset_index(drop=True), got[t].reset_index(drop=True), check_dtype=False,
                                      obj=f"{store} {kind} {t}")
