# Compatibility

Reference implementation: Stan 2.40 (stanc 2.40, CmdStan 2.40). Mathematics: stanmath-rs at
revision `04dc871`. stanc-rs: revision `242398e`.

## Differences from Stan

* **Random numbers.** Generated quantities and transformed data draw from `rand`'s `StdRng`
  (ChaCha12) through stanmath-rs's samplers. Draws are reproducible for a seed but are not
  bit-for-bit identical to CmdStan's, which uses boost's `ecuyer1988` generator.
* **Error messages.** Error kinds follow Stan's distinction between rejected proposals and fatal
  errors, but the message text differs.
* **Unconstrained names.** The names match stanc3, including its mismatch between the
  column-major order of the names and the row-major order of the values for multidimensional
  arrays (see [parameter conventions](parameter-conventions.md)).
* **Data.** Integers must be JSON integers in the 32-bit range; real values that are whole
  numbers are not converted to integers. Reals may be `"NaN"`, `"Inf"`, `"-Inf"`, `"Infinity"`,
  or `"-Infinity"`, quoted or bare (Stan's reader accepts the bare tokens, which strict JSON
  does not). Variables with no elements may be omitted, as in CmdStan.
* **Sizes.** Simplexes, sum to zero vectors and matrices, unit vectors, and stochastic matrices
  of size zero (in the dimension their transform reduces) are `Size` errors, as Stan's
  deserializer rejects them.
* **Print statements.** Stan passes an output stream to every model method; here the stream
  belongs to the calling thread (`print::set_sink`, `print::with_sink`), so model methods keep
  their signatures and chains on separate threads keep separate output. Without a sink, lines
  go to standard output, as in CmdStan. Lines are written when printed, not buffered until the
  evaluation returns, so lines printed before a `reject` are kept (stan-dev/stan issues 2311,
  3233).
* **Error locations.** `ModelError::locations` holds the Stan source locations of the failing
  statement and the user-defined function calls around it, innermost first, and `Display`
  prints them as Stan's `rethrow_located` does. Generated code records them with `locate`.

## stanc-rs integration

stanc-rs (revision `242398e`) generates code against its own crate `stanc-rs-runtime`, whose
contract its documentation says should move into stan-model-rs. Until stanc-rs targets
stan-model-rs, `stan-model-rs-stanc` adapts its models to `Model` and runs end-to-end tests.
The adapter cannot provide Hessians (the object-safe stanc-rs interface has no generic log
density) and reports every variable as unconstrained in the metadata (stanc-rs does not export
constraints). Its output names, values, gradients, constrained values, and generated quantities
match the hand-written models; its unconstrained names are stanc-rs's positional `theta.1`, ...

To target `StanModel` directly, the generated code needs these interface changes:

| stanc-rs-runtime | stan-model-rs |
| --- | --- |
| `GeneratedModel::model_name`, `num_unconstrained`, `var_info` | `StanModel::model_info` returning a `ModelInfo` built once, with blocks, element types, and constraints (`VarInfo { array_dims, base, constraint, slots }`) |
| `log_prob_generic<T: check::Scalar>` | `log_prob<T: Real>`; the generated code needs `Values` and `StanDisplay` only for printing and checks, so either those uses move to `f64` values or stan-model-rs exports an equivalent `Scalar` bound |
| `write_array_impl(theta, include_tp, include_gq, rng)` | `write_array_impl(theta, Outputs { transformed_parameters, generated_quantities }, rng)` |
| `unconstrain_json(&Data)` | `unconstrain_impl(&[f64])` on output-order values, or nothing when the metadata bounds suffice; `param_unconstrain_json` is provided |
| `DensityOptions` | `LogDensityOptions` |
| `ErrorKind` | adds `Constraint` (constraint violations of parameters and transformed parameters), `Unsupported`, and `Internal` |
| `Model::unconstrained_param_names` returns `theta.1`, ... | stanc3's names from the metadata |
| `rt::data::Data`, `rt::io::Reader` | `stan_model_rs::Data`, `transform::Reader` |
| no Hessians | `log_density_hessian` and Hessian-vector products through `Dual<Var>` |
| `print::{set_sink, with_sink, line}`, `ModelError::locations`, `locate` (in development after `242398e`) | the same functions in `stan_model_rs::print` and `stan_model_rs::error`, so generated code needs no change; the adapter forwards locations once its pin includes them |

Two facts the generated code must respect, both already true of stanc-rs:

* Data arguments of `_lupdf` calls must be passed as `f64`. stanmath-rs drops constant terms by
  argument type, so a data value lifted to an autodiff constant keeps its terms.
* Arrays of tuples are read element by element with the array indices in column-major order.

## nuts-rs

With feature `nuts`, `stan_model_rs::nuts::NutsModel` implements `nuts_rs::CpuLogpFunc` (nuts-rs
0.19) for any `Model`: the log density with Jacobian and without constants, its gradient, and
the constrained parameters of each draw. Recoverable model errors are reported as recoverable
`LogpError`s, so the sampler rejects the proposal.

## BridgeStan

The `Model` interface covers BridgeStan's model functions: `log_density`,
`log_density_gradient`, `log_density_hessian`, `log_density_hessian_vector_product`,
`param_constrain` (and `write_array` for transformed parameters and generated quantities),
`param_unconstrain`, `param_unconstrain_json`, `param_names`, `param_unc_names`
(`unconstrained_param_names`), `param_num` (`num_outputs`), and `param_unc_num`
(`num_unconstrained`). BridgeStan is not a dependency.
