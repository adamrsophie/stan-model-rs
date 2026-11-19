# Changelog

## 0.1.0 (unreleased)

* Model runtime: `StanModel` and `Model` traits, metadata, CmdStan JSON data, transforms for every
  Stan constraint and tuples, log densities, gradients, Hessians, Hessian-vector products,
  generated quantities, random number streams.
* Reference models, CmdStan 2.40 reference values, and end-to-end tests of stanc-rs output.
* Optional `nuts-rs` adapter.
* `param_initialize` for partial initial values; bare `NaN` and `Infinity` in JSON data;
  size zero structured transforms are errors instead of producing wrongly sized values.
