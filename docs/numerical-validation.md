# Numerical validation

Every check below runs in `cargo test --workspace` (the NUTS check with `--features nuts`).

| Check | Reference | Tolerance | Test |
| --- | --- | --- | --- |
| Normal mean model: value for 24 `(y, mu)` pairs, all four option combinations | Closed form written out without stanmath-rs | 1e-14 relative | `reference_models.rs` |
| Normal mean model: gradient `-mu/25 + (y - mu)`, Hessian `-1/25 - 1` | Closed form | 1e-13 | `reference_models.rs` |
| Normal scale model: value with and without Jacobian, proportional value, gradient `-e^z + y² e^(-2z)`, Hessian, for `z` from -20 to 20 | Closed form | 1e-12 (Hessian 1e-10) | `reference_models.rs` |
| Normal scale model: overflow and underflow of `e^z`, invalid and boundary initial values | Expected error kinds | exact | `reference_models.rs` |
| Structured and regression models: value and gradient at 2 points, with and without Jacobian | CmdStan 2.40 `log_prob` (Stan C++) | 1e-12 relative | `cmdstan_reference.rs`, `end_to_end.rs` |
| Regression model value | Closed form | 1e-13 | `invariants.rs` |
| Jacobian terms of simplex, lower bounds, ordered, and a parameter-dependent bound | log determinant of the finite-difference Jacobian of `param_constrain` | 1e-6 | `invariants.rs` |
| Gradients of three models, three option sets | Central finite differences | 1e-6 relative | `invariants.rs` |
| Hessian | Symmetry and differenced gradients; Hessian-vector product equals `H v` | 1e-12, 1e-5 | `invariants.rs` |
| Round trips of all 16 transforms and tuples | `free(constrain(y)) = y` | 1e-9 | `transform.rs` |
| Round trips on random points (proptest) | Same | 1e-8 | `invariants.rs` |
| stanc-rs generated models: values, gradients, constrained values, round trips | Hand-written fixtures | 1e-12 | `end_to_end.rs` |
| Names of every constrained type, complex types, tuples | stanc 2.40 generated C++ | exact | `meta.rs` |
| Repeated and interleaved evaluations, 8 threads | Serial results | bit identical | `invariants.rs` |
| Generated quantities: reproducible per seed, isolated from the density | Same seed, different stream | exact | `invariants.rs` |
| NUTS (`nuts-rs`) on the normal mean model, 4000 draws | Closed-form posterior mean and variance | 6 standard errors, 15% | `nuts.rs` |

## Regenerating the CmdStan reference values

The values in `tests/stan/cmdstan_log_prob.json` come from the programs in `tests/stan/`, built
with CmdStan 2.40 and evaluated with

```sh
./structured log_prob unconstrained_params=params.json jacobian=1 output file=out.csv sig_figs=18
```

(and `jacobian=0`, and `data file=...` for the regression), where `params.json` holds
`{"params_r": [[...], ...]}` with the points in the file. CmdStan's `log_prob` drops constants
(`propto = true`), so the tests compare with `LogDensityOptions { propto: true, .. }`.

## Results

All checks pass on macOS (aarch64) with Rust 1.98. CI runs them on Linux, macOS, and Windows.
