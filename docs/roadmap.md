# Roadmap

Done:

1. Core runtime: workspace, metadata, the two traits, data, log density and gradients, reference
   models, analytical tests, errors, documentation, and CI.
2. Constrained parameters: every Stan transform, inverse transforms, Jacobians, initial values,
   round trips.
3. Parameter metadata: arrays, vectors, matrices, complex values, tuples, stanc3's names and
   orders, JSON serialization.
4. Execution lifecycle: transformed data in constructors, transformed parameters, generated
   quantities with explicit random number streams.
5. Compiler integration: end-to-end tests of stanc-rs generated code through an adapter.
6. Higher derivatives: Hessians, Hessian-vector products, benchmarks.
7. Inference integration: `nuts-rs` adapter.

Next:

* stanc-rs generates `StanModel` implementations directly (interface changes listed in
  [compatibility](compatibility.md)); the adapter crate is then removed and Hessians become
  available for generated models.
* Reuse the autodiff tape across evaluations (stanmath-rs `GradientWorkspace`) behind a
  per-thread workspace, and constrain into preallocated buffers, after measuring the gain.
* More CmdStan reference points: covariance and correlation matrices, stochastic matrices,
  tuples, and models with transformed parameter constraints.
* Adapters for stan-inference-rs and stan-services-rs once their interfaces exist.
