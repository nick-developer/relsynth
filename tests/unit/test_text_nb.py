# SPDX-License-Identifier: Apache-2.0
"""The naive Bayes behind the text TSTR metric: the sparse path for conditions
with very many levels gives the same predictions as the dense table."""
import importlib

import numpy as np

E = importlib.import_module("relsynth.evaluate")   # the package exports a function of the same name


def _data(seed, n=400, classes=60):
    rng = np.random.default_rng(seed)
    words = [f"w{i}" for i in range(300)]
    labels = [f"c{int(x)}" for x in rng.integers(0, classes, n)]
    texts = [" ".join(rng.choice(words[int(lab[1:]) * 3:int(lab[1:]) * 3 + 40], int(rng.integers(1, 9))))
             for lab in labels]
    return texts, labels


def test_sparse_path_matches_dense(monkeypatch):
    texts, labels = _data(1)
    model = E._nb_fit(texts, labels)
    probe = _data(2, n=200)[0] + ["", "unseen words only"]
    dense = E._nb_predict(model, probe)
    monkeypatch.setattr(E, "_NB_DENSE_CELLS", 0)
    sparse = E._nb_predict(model, probe)
    assert sparse == dense


def test_many_levels_do_not_build_the_table(monkeypatch):
    texts = [f"tok{i} common" for i in range(8000)]
    labels = [f"id{i}" for i in range(8000)]          # 8000 classes x 8001 words = 64 M cells
    calls = []
    real = E._nb_predict_sparse
    monkeypatch.setattr(E, "_nb_predict_sparse", lambda *a: calls.append(1) or real(*a))
    out = E._nb_predict(E._nb_fit(texts, labels), ["tok17 common", "tok7999"])
    assert calls and out == ["id17", "id7999"]
