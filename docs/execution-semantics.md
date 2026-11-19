# Execution semantics

The runtime follows the block semantics of the Stan Reference Manual for Stan 2.40 (the
version of stanc 2.40 and CmdStan 2.40, the pinned reference implementation).

| Block | When it runs | What the runtime does |
| --- | --- | --- |
| `data` | Once, in the model constructor | Reads CmdStan JSON, checks shapes and declared constraints; failures are `Data` errors |
| `transformed data` | Once, in the constructor, after `data` | Values are computed once and stored in the model; random draws use a generator seeded by the constructor's seed (`rng::TRANSFORMED_DATA_STREAM` by convention) |
| `parameters` | Every evaluation | `Reader` reads the unconstrained vector in declaration order and constrains each parameter, adding its log Jacobian when requested |
| `transformed parameters` | Every log density evaluation and every `write_array` | Evaluated from the constrained parameters; constraints are checked and a violation rejects the proposal |
| `model` | Every log density evaluation | Contributions accumulate into the returned value |
| `generated quantities` | Only in `write_array` with `outputs.generated_quantities` | Draws from the generator passed in; never part of a log density or gradient |

## Constrained and unconstrained parameters

Inference works on the unconstrained vector `θ ∈ ℝⁿ`. Each parameter `x = T(y)` is obtained
from its slice `y` of `θ` by the transform its declaration selects (for example
`σ = exp(z)` for `real<lower=0> sigma`). The constrained and unconstrained sizes can differ:
`simplex[K]` uses `K - 1` values, `cholesky_factor_corr[K]` uses `K(K - 1)/2`.

The density of `θ` is `p(T(θ)) |det J_T(θ)|`. With `jacobian = true` the runtime adds
`log |det J_T(θ)|` (for `σ = exp(z)` this is `z`); with `jacobian = false` it does not, which is
the density of the constrained parameters evaluated at `T(θ)`, as used for maximum likelihood
and maximum a posteriori optimization. The Jacobian is added once, by the transform that reads
the parameter, and never by the model block.

## Full and proportional densities

With `propto = false` every term is kept. With `propto = true`, `~` statements and
`_lupdf`/`_lupmf` calls drop the terms that do not depend on parameters, as Stan does. Which
terms are constant is decided by stanmath-rs from the argument types, so the runtime evaluates
`propto` densities with autodiff types even when only the value is requested. `target +=`
statements with `_lpdf` are always kept in full. Proportional and full densities differ by a
constant, so their gradients are identical (tested).

## Log density evaluation and generated quantities

A log density evaluation is a pure function of the data, `θ`, and the options: it never draws
random numbers, never mutates the model, and repeated calls return identical bits (tested). The
generated quantities of a draw are computed separately by `write_array`, which the services
layer calls for accepted draws with its own random number stream. One generator passed to
consecutive calls continues its stream; the runtime never reseeds.

## Evaluation and inference

The runtime evaluates a model at points chosen by someone else. It does not choose points,
adapt step sizes, manage chains, or retry initializations: `model::random_unconstrained` draws
one candidate in `(-2, 2)ⁿ` as Stan does, and the retry policy belongs to stan-services-rs.
