# Runtime contract

This is the interface between model code (generated or hand written) and the runtime.

## What a model provides

```rust
pub trait StanModel {
    fn model_info(&self) -> &ModelInfo;
    fn log_prob<T: Real>(&self, theta: &[T], jacobian: bool, propto: bool) -> Result<T>;

    // Optional, with defaults driven by the metadata:
    fn write_array_impl(&self, theta: &[f64], outputs: Outputs, rng: &mut dyn Rng) -> Result<Vec<f64>>;
    fn unconstrain_impl(&self, constrained: &[f64]) -> Result<Vec<f64>>;
    fn bounds(&self, var: &VarInfo) -> Result<Bounds>;
}
```

A model type is constructed from `&Data` (and a seed if its transformed data draw random
numbers). The constructor reads every data variable with `Data::read_checked` (or
`read_with_bounds` when a bound depends on data), runs transformed data, checks their
constraints, and builds the `ModelInfo` once.

`log_prob` receives exactly `num_unconstrained` values (the runtime checks). It must:

1. read the parameters in declaration order with `transform::Reader`, passing the Jacobian
   accumulator only when `jacobian` is true;
2. evaluate the transformed parameters and check their constraints (`transform::check`);
3. add the model block, using the `_lupdf`/`_lupmf` forms for `~` statements when `propto` is
   true, with data arguments passed as `f64` (stanmath-rs decides which terms to drop from the
   argument types, so a data value lifted to an autodiff type would keep its terms);
4. return the sum, or an error. It must not panic on any input.

`write_array_impl` returns the constrained parameters, then the transformed parameters if
`outputs.transformed_parameters`, then the generated quantities if
`outputs.generated_quantities`, each variable flattened in output order. Generated quantities
draw only from `rng`. The default implements parameters only.

`unconstrain_impl` receives constrained parameter values in output order and returns the
unconstrained vector. The default frees each parameter with `bounds(var)`, which reads constant
bounds from the metadata. Models whose bounds depend on data override `bounds`; models whose
bounds depend on other parameters override `unconstrain_impl` (see `DependentBound` in the
fixtures).

## What the runtime provides (`Model`)

| Method | Meaning |
| --- | --- |
| `info`, `name`, `num_unconstrained`, `num_outputs` | Metadata |
| `param_names(outputs)` | Flattened output names, `beta.1`, `Sigma.2.1`, `z.real`, `p:1` |
| `unconstrained_param_names()` | stanc3's unconstrained names |
| `log_density(theta, opts)` | Log density; with `propto` it is evaluated with autodiff types |
| `log_density_gradient(theta, opts)` | Value and gradient, one reverse pass |
| `log_density_gradient_into(theta, opts, grad)` | The same, into a caller buffer |
| `log_density_hessian(theta, opts)` | Value, gradient, dense Hessian (forward over reverse) |
| `log_density_hessian_vector_product(theta, v, opts)` | `H v` without forming `H` |
| `check_gradient(theta, opts, h)` | Gradient against central finite differences |
| `write_array(theta, outputs, rng)` | Constrained values, transformed parameters, generated quantities |
| `param_constrain(theta)` | Constrained parameters only |
| `param_unconstrain(values)` | Unconstrained vector from constrained values in output order |
| `param_unconstrain_json(data)` | The same from values given by name (initial values) |
| `param_initialize(data, radius, rng)` | Stan's initialization: given parameters keep their values, missing ones are drawn uniformly in `(-radius, radius)` on the unconstrained scale |

`LogDensityOptions { jacobian, propto }` defaults to `{ true, false }`, the full density on the
unconstrained space. `LogDensityOptions::SAMPLING` is `{ true, true }`, what HMC uses and what
CmdStan's `log_prob` method computes.

## Errors

`ModelError { kind, message, variable, function, value }`.

| Kind | Raised when | Recoverable |
| --- | --- | --- |
| `Data` | Data missing, malformed, wrong shape, or violating a declared constraint | no |
| `Parameters` | Wrong number of values, or a constrained value on the boundary of its support | no |
| `Constraint` | A constrained value outside its support, or a transformed parameter violating its constraint | yes |
| `Domain` | A function argument outside its domain (from stanmath-rs) | yes |
| `Index`, `Size` | Index out of range, inconsistent sizes | no |
| `Reject` | The program executed `reject` | yes |
| `Fatal` | The program executed `fatal_error` | no |
| `Unsupported` | Operation not supported by the model or runtime | no |
| `Internal` | Inconsistent metadata or a bug | no |

`locations` lists the Stan source locations of the failing statement and of the
user-defined function calls around it, innermost first (`ModelError::at`, `error::locate`).

Output of `print` statements goes to the calling thread's sink (`print::with_sink`), or to
standard output when none is set.

`ErrorKind::is_recoverable` is true where Stan throws `std::domain_error`: a sampler rejects the
proposal and continues. A log density of negative infinity is a value, not an error. NaN values
are never replaced.
