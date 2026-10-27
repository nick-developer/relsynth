# Theory: the model, its guarantees and the v1.0 propositions

This page summarises the statistical model behind relsynth and states, with proofs, the six propositions added in v1.0 for fitting inside the database. The complete treatment has 105 numbered propositions, each with its proof and the check that tests it. It is in [proofs/relsynth-proofs.pdf](proofs/relsynth-proofs.pdf), with the LaTeX source alongside. The algorithms, their complexity and the layer 1 derivations are in [proofs/algorithms.pdf](proofs/algorithms.pdf). Proposition numbers below refer to relsynth-proofs.pdf.

## 1. The model in brief

A database is a set of tables $T$ linked by foreign keys. relsynth models it in three layers.

**Latent segments for entities.** Every row $i$ of an entity table $e$ (a dimension, or any table other tables reference) has a latent segment $z_i \in \{1..K_e\}$. The segments carry the dependence *between* tables. A customer in segment 3 buys more of product segment 7, and its own attributes follow segment 3's distribution. Segments are fitted jointly over the whole schema by Gibbs sampling. $K_e$ starts high and is merged down using block statistics (Props. 50, 51, 82), and the segments are identifiable up to relabelling (Prop. 4).

**Facts as counts between segments.** A fact linking entities $e_1, e_2$ is a set of rows per pair $(i, j)$. The number of rows per pair is Poisson with a rate $\eta_{z_i z_j}$ per pair of segments times degree factors (Props. 5, 6). It may also be zero-inflated, with structural zeros and ones per block (Props. 17–19, 52, 87). Multiplicities beyond one row per pair follow spike-and-slab or Dirichlet-process count laws (Props. 21, 32, 89). Participation, meaning whether an entity row has any fact row at all, has its own hurdle (Props. 34, 70, 91, 92). Nested facts refine a parent fact (Prop. 27). Order headers are split losslessly from lines (Prop. 43), and running balances are rebuilt from flows (Prop. 49).

**Columns within a table.** Columns follow a Bayesian Gaussian copula over mixed types:
- the marginals are exact (empirical or Dirichlet-process; Props. 1, 2);
- the correlation has a rank likelihood with parameter expansion (Props. 3, 28);
- the copula is per segment, or pooled when cells are small (Prop. 95);
- rare columns stay independent (Props. 29, 81).

Hierarchies are Galton–Watson trees (Props. 9, 10, 38, 54). Time-stamped facts are thinned by season (Prop. 11), and text has a proper predictive law with memorisation guards (Props. 12, 13).

**Rules and keys.** Rules are imposed by exact conditioning:
- rejection (Prop. 14);
- three-valued logic that agrees with SQL (Prop. 15);
- computed columns that hold exactly (Prop. 16);
- redraws instead of deletions (Prop. 45);
- calibrated link draws (Props. 62, 63);
- checks on the draws that are kept (Prop. 93).

Unique keys are a conditioning event too (Props. 24, 94).

**Synthesis is posterior predictive.** Dataset $m$ draws every parameter once from its posterior with the stream of dataset $m$, then the rows (Props. 30, 66). Two datasets therefore differ by parameter uncertainty and by sampling noise, as two draws from the posterior predictive distribution should. The exported model's draw $m$ is the parameter set of dataset $m$ (Prop. 83, checked byte for byte).

**Evaluation is calibrated.** Every scorecard limit is the larger of a fixed tolerance and the sampling limit of its statistic, under the null that the synthetic data is a posterior predictive draw of the real data's law (Prop. 98). Maxima over families are Šidák-corrected (Prop. 99), so the expected number of false failures is the reported sum of the checks' levels.

## 2. Fitting inside the database (v1.0)

A fact table is *pushed down* when it is a fact that no table references, has no type 2 history, and is not the source of an aggregate column that the source does not store. Its rows never leave the database whole. The engine reads exact aggregates computed with `GROUP BY` and window functions, plus a uniform sample of at most `sample_rows` rows. The next sections show that this changes nothing in the relational model, and that the column models fitted on the sample are an exact posterior given the sampled rows.

**Notation.** Let a pushed fact $F$ have $N$ rows, key columns $k_1..k_m$ and continuous columns $c_1..c_p$. For a non-null value $x_i$ of column $c$ with average rank $r_i$ among the $N_c$ non-null values, its *normal score* is $z_i=\Phi^{-1}\bigl(r_i/(N_c+1)\bigr)$, and $z_i=0$ for a null. For a key tuple $u$:

$$n(u)=\#\{i:k(i)=u\},\qquad S_c(u)=\sum_{i:k(i)=u}z_i^{(c)},\qquad Q_c(u)=\sum_{i:k(i)=u}\bigl(z_i^{(c)}\bigr)^2,$$

with a null key value forming a group of its own.

### Proposition 100 (Sufficiency of the in-database statistics)

Given the loaded tables (entities and non-pushed facts), the full conditional of every latent quantity of the relational model that involves a pushed fact $F$ depends on the rows of $F$ only through the following statistics. The latent quantities covered are segment assignments, block and link rates, count and multiplicity laws, participation and inclusion, nested refinement, order headers and running balances.

1. The counts $n(u)$ over the key tuples the model uses: pairs of links, a composite key with its links, a header key with its header columns.
2. The score sums $S_c(u)$ and $Q_c(u)$ for the (at most three) continuous columns that segment discovery uses.
3. For an order header, the first row of every document key in primary-key order, and the number of cells that differ from it. For a running balance, the first row of every group in the rule's order, the number of non-null values and the minimum.
4. $N$, the number of non-null values of each key, and the distinct referenced values of each key.

*Proof.* Each listed model block is a product over the rows of $F$ of a factor that depends on a row only through its key tuple $u$ and, for segment discovery, through the row's rank features $z^{(c)}$.
- **Rates and multiplicities.** Block-rate and multiplicity likelihoods are products of Poisson or multinomial terms. Their exponents are the counts of rows per pair of linked rows, so they are functions of $n(u)$; the conjugate posteriors of Props. 5 and 21 use sums of $n(u)$ over segment pairs.
- **Rank features.** The rank-feature factor is Gaussian in the scores, with a mean per pair of segments and a common variance per column. Its log-likelihood over the rows of a key tuple is therefore $a\,n(u)+b\,S_c(u)+c\,Q_c(u)$, with coefficients that do not depend on the rows, and the conjugate updates of the means and variances use the same three sums.
- **Participation and inclusion.** These use which keys occur (the distinct values) and how often ($n(u)$).
- **Nested facts.** A nested fact is linked to its parent fact through its composite key. The counts over (composite key, links) therefore give the parent row and the dimension rows of every group, with $n(u)$ as the group's weight.
- **Order headers.** An order header is the first row of its document key (Prop. 43). Its split is lossless exactly when no cell differs from that row, which is what item 3 counts.
- **Running balances.** A running balance needs its opening value per group (the first row in the rule's order) and its floor (the minimum). The rest of the column is a deterministic function of the flows (Prop. 49).

Nothing else of $F$ enters these factors. ∎

### Proposition 101 (Agreement with in-memory fitting)

Let the loaded rows of every table be put in a canonical order: by primary key, or by a hash of the row's values when there is none. Then two things hold:
1. If the sample of a pushed fact holds all its rows, the fitted model and every synthetic dataset are identical, value for value, to those of fitting the whole table in memory with the same seed.
2. The model fitted from a database does not depend on the order in which the database returns rows.

*Proof.* By Proposition 100 the relational blocks see the same numbers, computed either in pandas (`FrameStats`) or in SQL (`SqlStats`), and the attribute models see the same rows. The fit is a deterministic function of its inputs, the seed, and the order in which rows are visited, because Gibbs sweeps visit rows in order. The canonical order is a function of the set of rows only, so two backends, or two scans of one table, give the same input sequence. The only remaining difference is floating point:
- SQL sums are accumulated in another order than pandas uses;
- servers compute $\Phi^{-1}$ with Proposition 102.

Both perturb $S_c$ and $Q_c$ by relative errors below $10^{-8}$, far under the resolution at which a segment draw changes. ∎

*Checked.* The synthetic tables are compared cell by cell, and they are identical in two settings:
- all four pushed fact shapes (plain, nested, header-split with a running balance, view base) on SQLite;
- a pushed star schema on PostgreSQL, SQL Server and Oracle against SQLite in memory.

Without the canonical order, the same data gave different models on different servers, because SQL Server and Oracle returned rows in another order than SQLite and PostgreSQL.

### Proposition 102 (Acklam's quantile approximation)

PostgreSQL, SQL Server and Oracle have no standard-normal quantile function. relsynth computes $\Phi^{-1}(p)$ in SQL with Acklam's rational approximation, whose relative error is below $\delta = 1.15\times10^{-9}$ on $(0,1)$. The approximation is a rational function of $q=p-\tfrac12$ on $[0.02425, 0.97575]$, and of $\sqrt{-2\log p}$ in the tails. Hence

$$|\hat S_c(u)-S_c(u)|\le \delta\sum_{i:k(i)=u}|z_i|,\qquad |\hat Q_c(u)-Q_c(u)|\le (2\delta+\delta^2)\,Q_c(u).$$

SQLite registers the exact function (`scipy.special.ndtri`).

*Proof.* The error bound of the approximation is Acklam's (2003). The test `test_acklam_error_bound` confirms it in double precision on 4,081 points spanning $[10^{-12},1-10^{-12}]$: the relative error where $|\Phi^{-1}(p)|>10^{-3}$, and an absolute error below $10^{-8}$ everywhere. For a sum, $|\sum\hat z_i-\sum z_i|\le\sum|\hat z_i-z_i|\le\delta\sum|z_i|$. For squares, $|\hat z^2-z^2|=z^2\,|(1+\epsilon)^2-1|\le(2\delta+\delta^2)z^2$. ∎

### Proposition 103 (Seeded hash sampling is a Bernoulli sample)

The sample keeps row $i$ when $h(s,\kappa_i)<\lfloor q\,2^{32}\rfloor$, where:
- $\kappa_i$ is the row's primary key (all its columns when there is none);
- $s$ is the seed;
- $q=\min(1,n/N)$ for the requested size $n$;
- $h$ is the database's hash reduced to $[0,2^{32})$.

If $h(s,\cdot)$ behaves as a uniform random function independent of the data, then:

1. the inclusion indicators are i.i.d. Bernoulli$(q)$ and independent of every column value, so the sample is a uniform random subset with $|S|\sim\mathrm{Bin}(N,q)$;
2. the same seed gives the same sample on every run;
3. $P\bigl(\bigl||S|-Nq\bigr|>8\sqrt{Nq(1-q)}+10\bigr)<1.3\times10^{-15}$. The guard that stops a sample of that size, such as one from a hash returning NULL, therefore does not stop a valid sample in practice.

*Proof.*
1. Under the assumption, the values $h(s,\kappa_i)$ are i.i.d. uniform on $[0,2^{32})$ because the keys are distinct. Each comparison is therefore Bernoulli with probability $\lfloor q2^{32}\rfloor/2^{32}$, which differs from $q$ by less than $2^{-32}$.
2. The comparison is a deterministic function of $(s,\kappa_i)$.
3. With $\sigma^2=Nq(1-q)$ and $t=8\sigma+10$, Bernstein's inequality gives $P(||S|-Nq|>t)\le 2\exp\bigl(-t^2/(2\sigma^2+2t/3)\bigr)$, which tends to $2e^{-32}<2.6\times10^{-14}$ as $\sigma\to\infty$. The exact binomial tails, computed for $N$ from 10 to $10^9$ and $q\in[10^{-6},1-10^{-6}]$, are at most $1.24\times10^{-15}$. That is the normal value $2\Phi(-8)$, approached for large $\sigma$; for small $\sigma$ the additive 10 makes the tail far smaller. ∎

*Checked.* `test_python_hash_is_uniform_and_seeded` runs a KS test of uniformity. `test_sample_is_uniform_reproducible_and_sized` checks size, uniformity across value classes, and reproducibility on SQLite and on each server.

### Proposition 104 (Attribute models from the sample are an exact posterior)

With the sample drawn as in Proposition 103, the posterior of the attribute-model parameters $\theta$ of the pushed fact given the sampled rows,

$$p(\theta\mid x_S,S)\propto p(\theta)\prod_{i\in S}p(x_i\mid\theta,\text{keys}_i),$$

is the exact posterior given the observed data: the sampling mechanism is ignorable. Synthetic datasets are therefore posterior-predictive draws, given the sampled rows for the attributes and given all rows for the relational part. Their sizes follow the exact $N$.

*Proof.* The inclusion indicators are independent of the values and of $\theta$ (Proposition 103, item 1), so $p(S\mid x,\theta)=p(S)$. The likelihood of the observed data then factors as $p(S)\prod_{i\in S}p(x_i\mid\theta)$, and the factor $p(S)$ does not involve $\theta$: the data are missing completely at random (Rubin 1976). Proposition 100 makes the relational blocks depend on all rows through exact statistics. The sizes use $N$ because the scale rule is applied to the exact counts. ∎

*Consequence.* The posterior is wider than with all rows (its variance scales with $1/|S|$ for regular parameters). The scorecard's real side therefore uses exact statistics where they exist: row counts, null keys, cardinalities, duplicate rates and pair multiplicities. Elsewhere it uses the sample, with limits computed for the sample's size.

### Proposition 105 (Eligibility)

A table cannot be fitted from aggregates alone when any of the following holds:
- it is an entity: each of its rows carries its own latent segment, whose full conditional needs that row's attributes and links (Prop. 47);
- another table references it: it is then an entity of its own rows for that table's links, as is the parent of a nested fact;
- it has type 2 history: versions are drawn per business key and row (Prop. 72);
- an aggregate column of another table is computed from its rows and the source does not store that column, so the column must be recomputed from the rows.

Every other fact is eligible, including nested facts, facts split into order headers, bases of views and facts with running balances.

*Proof.* In each excluded case, the named proposition's full conditional has a factor per row of the table that is not a function of the statistics of Proposition 100. For the eligible shapes, Proposition 100 lists the statistics that replace the rows. ∎

## Generator plug-ins (v2.0)

The attribute columns of each table come from a *generator*. The default is the copula of section 1; `independent`,
`cart` and third-party generators can be chosen per table ([generators.md](generators.md)). The engine keeps the
relational skeleton: rows, keys, links, latent segments, counts, scale and rules. A generator draws a row's
attributes given its context, which is the row's latent group $g$, the coarser group $\bar g$ it backs off to, and
(for generators with `parent` conditioning) the attributes $a$ of its synthetic parents. Propositions 700–706 state
what still holds. They are numbered from 700 so that the parallel 2.x threads do not collide. The full proofs are in
`docs/proofs/sec_v20_generators.tex`, and the checks are in `tests/unit/test_generators.py`.

### Proposition 700 (Skeleton and rules do not depend on the generator)

For any generator that returns exactly its table's own attribute columns:

1. Keys, links, trees, row counts and scale follow the 1.0 laws.
2. Computed rules (derive, aggregate, cumulative, computed columns) hold exactly.
3. A row that breaks a check rule or a unique attribute key is redrawn from the generator's own conditional. An
   accepted row has the generator's law restricted to the rules. With acceptance probability $\pi$, the chance of
   a violation left after $r$ rounds is $(1-\pi)^{r+1}$, and such violations are reported.

*Proof sketch.* Keys and links are drawn from the relational model and the column encoding, both of which exist
before a generator is attached. The engine refuses generator output with other columns. Computed columns are
evaluated afterwards from the final values. The redraws are independent draws from the same conditional, so the
standard rejection-sampling argument applies. ∎ *Checked:* `test_reference_generators_keep_keys_links_and_rules`,
and integrity 100% on every suite for every generator (T03 report).

### Proposition 701 (Seed determinism)

If a generator draws all its randomness from the generator it is handed, the model and every dataset are
deterministic functions of the spec, the data and the seed. ∎ *Checked:* `test_reference_generators_are_seeded_and_have_a_posterior`.

### Proposition 702 (The copula plug-in is relsynth 1.0)

With the default generator, the output is byte-identical to 1.0. The code moved without changing an operation, and
attaching a generator draws no random number. ∎ *Checked:* SHA-256 of every output file. The comparison covers
main and 2.0 on S1–S11 (two datasets each), Chinook (in memory, and pushed down with two datasets), Contoso SQLite
and Parquet, bi/COREP and AdventureWorks DW. It also covers `test_copula_is_the_default_and_naming_it_changes_nothing`.

### Proposition 703 (Posterior predictive datasets)

Suppose `posterior_draw` draws a generator's parameters from their posterior given the table and the fitted
partition. Then dataset $m$ is a posterior predictive draw of the joint model, as in 1.0, because the attribute
blocks' posteriors factorise given the partition and the contexts.

- With an approximate posterior, the dataset is a draw from the approximate predictive.
- With `posterior: false`, datasets share the attribute parameters.
- `copula` and `independent` are exact for their models.
- `cart` redraws leaf weights given trees fitted once (Prop. 706).

∎

### Proposition 704 (What group-only conditioning keeps and loses)

With group-only conditioning, a fact's attributes are independent of its parents' attributes given the group, and
$\operatorname{Cov}(x,a)=\operatorname{Cov}_g(\mathbb E[x\mid g],\mathbb E[a\mid g])$. Parent conditioning can also
carry the dependence within a group. *Proof:* the law of total covariance. ∎ *Checked:*
`test_parent_conditioning_carries_within_group_dependence`.

### Proposition 705 (Independent marginals are a latent class model)

`independent` is the copula with the identity correlation and the same marginal posteriors. Within a group its
columns are independent, and between groups they covary through the group means (Goodman 1974; Dunson and Xing
2009). ∎ *Checked:* `test_independent_columns_are_independent_within_a_group`.

### Proposition 706 (CART leaf donors with Bayesian-bootstrap weights)

A synthetic value drawn from leaf $\ell$ with weights $w\sim\mathrm{Dir}(1,\dots,1)$ has, averaged over $w$, the
empirical law of the leaf. The probability of a value with count $c$ has mean $c/n_\ell$ and variance
$c(n_\ell-c)/(n_\ell^2(n_\ell+1))$. $w$ is the posterior of the leaf law under a Dirichlet process whose
concentration tends to 0 (Rubin 1981; Ferguson 1973). *Proof:* exchangeability, and the Dirichlet aggregation
property. ∎ *Checked:* `test_cart_leaf_weights_are_a_bayesian_bootstrap`.

## 3. Memory

A pushed fact costs memory proportional to the number of distinct key tuples plus the sample, instead of rows times columns. On AdventureWorks DW, all ten eligible facts were pushed, including the 776,286-row inventory fact with its running balance. Peak memory fell from 2.1 GiB to 1.6 GiB with 30,000-row samples, and the scorecard and every domain check (amount identities, order headers, running balances, date offsets) agreed with the in-memory run. See [pushdown.md](pushdown.md).

## 4. The IR (2.0): Propositions 106–108

Version 2.0 describes the data by one modality-neutral document, the IR ([guide](ir-guide.md)). The engine still runs the relational 1.0 spec $\lambda(E)$ that an IR document $E$ lowers to; a 1.0 spec $s$ lifts to $\ell(s)$; $\nu$ is `spec.normalise`. Full proofs are in [proofs/sec_v20_ir.tex](proofs/sec_v20_ir.tex).

### Proposition 106 (Loss-free conversion and identical output)

For every 1.0 spec $s$ the 1.0 key schema accepts, $\nu(\lambda(\ell(s)))=\nu(s)$, so the IR document and the spec give the same model and the same synthetic data for the same seed.

*Proof sketch.* $\ell$ and $\lambda$ are field-by-field inverses on the 1.0 keys (the table in `ir/v1.py`); fields that $\ell$ derives are values $\nu$ fills in the same way; order is kept. The engine reads only $\nu(s)$, the data and the seed. ∎ Checked on all eleven golden specs and S1–S11, and by comparing the synthetic output of the four real databases byte by byte.

### Proposition 107 (Ownership along reference paths)

Under referential integrity, the composition of many-to-one and one-to-one references along a path from entity $A$ to the unit $U$ maps each record to at most one unit. If the reference graph is acyclic, the records of $A$ owned by unit $u$ are the union over the simple paths that `privacy.trace` returns, and a record has at most as many owners as paths. ∎

### Proposition 108 (Removing a unit keeps referential integrity)

If the reference graph is acyclic and no hierarchy or excluded reference points into an owned record, removing a set of units with all their owned records leaves every non-null reference resolvable. Otherwise the offending references are the cross-unit links that `trace` reports. ∎

## 5. References

- Bitol project (Linux Foundation AI & Data). Open Data Contract Standard v3.0–3.2.
- OpenLineage project. Dataset facets: Schema 1-2-0, DatasetType 1-0-1, Documentation 1-1-0, Tags 1-0-0, ColumnLineage 1-2-0.
- Apache Iceberg table specification, partition transforms.
- M.-A. Baazizi, D. Colazzo, G. Ghelli, C. Sartiani (2019). Parametric schema inference for massive JSON datasets. *VLDB Journal* 28.
- K. Cai, X. Xiao, G. Cormode (2023). PrivLava: synthesizing relational data with foreign keys under differential privacy. *SIGMOD*.
- N. Ponomareva et al. (2026). How to DP-fy your data: a practical guide to generating synthetic data with differential privacy. *JAIR* 86. arXiv:2512.03238.
- P. J. Acklam (2003). An algorithm for computing the inverse normal cumulative distribution function.
- D. B. Rubin (1976). Inference and missing data. *Biometrika* 63(3).
- P. D. Hoff (2007). Extending the rank likelihood for semiparametric copula estimation. *Annals of Applied Statistics* 1(1).
- B. Karrer, M. E. J. Newman (2011). Stochastic blockmodels and community structure in networks. *Physical Review E* 83.
- E. I. George, R. E. McCulloch (1993). Variable selection via Gibbs sampling. *JASA* 88(423).
- J. P. Reiter (2005). Using CART to generate partially synthetic public use microdata. *J. Official Statistics* 21(3).
- B. Nowok, G. M. Raab, C. Dibben (2016). synthpop: Bespoke creation of synthetic data in R. *J. Statistical Software* 74(11).
- L. A. Goodman (1974). Exploratory latent structure analysis. *Biometrika* 61(2); D. B. Dunson, C. Xing (2009), *JASA* 104(487).
- D. B. Rubin (1981). The Bayesian bootstrap. *Annals of Statistics* 9(1).
- Further references are in the proofs and the algorithms paper.
