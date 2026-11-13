# Binary columns

Binary columns hold images, documents and other files (`varbinary`, `BLOB`, `bytea`, `image`, Parquet binary,
`0x`-hex text in CSV exports). relsynth 2.1 wrote them as NULL. Since gap plan G4 they can get synthetic content
that never contains a byte of the source.

```yaml
synthesis:
  binary:
    generator: placeholder      # none (default, NULL) | placeholder | random
tables:
  DimProduct:
    columns:
      LargePhoto:
        type: binary
        binary: {generator: placeholder, mime: image/gif}   # per column, optional
```

## What is learned from the source

Only three things, all in `relsynth.binary`:

| Quantity | Model |
|---|---|
| null share | `p ~ Beta(nulls + 1/2, values + 1/2)` |
| MIME type of each value | read from its first bytes (PNG, GIF, JPEG, PDF, TIFF, WebP, ZIP, gzip, OLE compound files); a type seen fewer than `privacy.suppress_small_counts` times is pooled into `application/octet-stream`; `mime` overrides it |
| length | a length shared by at least `privacy.suppress_small_counts` values of a type is kept exactly (a repeated "no image" file); every other length is released only as its power-of-two bin `[2^b, 2^(b+1))` and drawn log-uniformly in it |

The (type, length class) cells get a Dirichlet(counts + 1/2) posterior. Each dataset draws the null share and
the cell shares once (`posterior.mode: draw`) or uses their means (`mean`).

## What is written

- `placeholder`: a fixed, valid file of the cell's type that ships with relsynth (a one-pixel PNG, GIF or JPEG,
  a one-page PDF), padded with zero bytes after its end marker to the drawn length. Image and PDF readers
  ignore the trailing bytes. A type without a placeholder gets random bytes.
- `random`: uniform random bytes of the drawn length.
- `none`: NULL.

## Guarantees and limits

- **No source bytes.** Content is relsynth's placeholder or random bytes; the only information from the source
  is the null share, the type mix and the length classes above (Prop. 1781). A random value equals a given
  source value of length `L` with probability `256^-L`, so only values of a few bytes can match by chance.
- **Nothing else changes.** Values are drawn from a stream keyed by the data stream's state and the column,
  without drawing from the data stream, so every other column of the dataset is byte-identical whether the
  generator is on or off (Prop. 1780). Partitions and datasets get their own values.
- The content does not depend on the row's other columns (a product photo does not follow its category).
- DP mode keeps binary columns NULL.
- Writers store the bytes as each target does: `LargeBinary` in SQL targets, binary in Parquet, `0x`-prefixed hex
  in CSV.
