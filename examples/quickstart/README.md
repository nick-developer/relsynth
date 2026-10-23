# Quick start example

`shop.db` is a small SQLite music shop (9 tables, 6,680 rows) generated from a known model. `spec.yaml` is the specification `relsynth introspect` drafted from it, unedited except for its name.

```bash
cd examples/quickstart
relsynth check spec.yaml --connect
relsynth run spec.yaml --target sqlite:synthetic.db --report scorecard.md
relsynth doctor diagnostics_bundle.json
relsynth verify spec.yaml
relsynth export-model spec.yaml --out model
```

Each step, with its real output, is explained in the [user guide](../../docs/user-guide.md).
