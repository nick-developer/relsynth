# Example: the exported Chinook model

Written by `relsynth export-model spec.yaml --draws 3` (v0.9) on the Chinook sample database.

| File | What it is |
|---|---|
| [model.pdf](model.pdf) / [model.md](model.md) | The Bayesian generative model with this database's numbers, the verification table (7/7 checks pass) and the replica's scorecard (139/140) |
| model.json | Structure, posterior mean and 3 posterior draws of every parameter block (draw m is the parameter of dataset m of `relsynth run`). Aggregates only: text, names and identifiers are not exported |
| replicate.py + relsynth_expr.py | Standalone sampler: needs only NumPy, pandas and SciPy |
| run_model.py | Re-runs the engine's own calls; output identical byte for byte to `relsynth run` (needs relsynth and the source) |
| spec.yaml | The specification the model was fitted from; `source.path` points to `Chinook.db` next to it |

Run the standalone sampler (tested in a folder without relsynth installed):

```
python replicate.py model.json --out synthetic --draw 0 --reference ref
```

`--reference` is a folder with the copied reference tables as CSV (Genre, MediaType and Playlist for this model), because copied tables are not modelled. `--draw mean` uses the posterior mean; `--scale 2` doubles the database.
