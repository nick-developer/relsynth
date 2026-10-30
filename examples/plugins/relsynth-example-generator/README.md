# relsynth-example-generator

A complete third-party generator plug-in in about 60 lines. It shows everything
a plug-in needs: a `TableGenerator` subclass with a name, a model version and
capabilities, `fit`, `sample`, `posterior_draw`, and one entry point in
`pyproject.toml`.

The generator (`empirical`) draws each attribute column independently from the
values observed in the row's latent group, with Bayesian-bootstrap weights
redrawn for every dataset. It is a teaching example, not a recommended model:
it repeats observed values, so use the built-in generators for real data.

```bash
pip install ./examples/plugins/relsynth-example-generator
relsynth generators list          # shows "empirical" from relsynth-example-generator 0.1.0
```

Then, in a spec:

```yaml
tables:
  customer:
    generator: {name: empirical, version: ">=0.1"}
```

relsynth itself is not changed: the plug-in is found through the
`relsynth.generators` entry-point group.
