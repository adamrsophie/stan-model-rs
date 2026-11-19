# Architecture

```text
Stan program ──stanc-rs──▶ Rust model code ──implements──▶ StanModel (generic in the scalar)
                                                               │ blanket impl
                                                               ▼
              inference (nuts-rs, stan-inference-rs) ◀──uses── Model (object safe)
                                                               │
                                     stanmath-rs: autodiff, densities, transforms, linear algebra
```

## Two traits

Automatic differentiation needs the log density as a generic function of the scalar type:
`f64` for plain values, `Var` for reverse mode, `Dual<Var>` for Hessian-vector products and
Hessians. Samplers need a trait object. The runtime therefore has two traits:

* `StanModel` is implemented by model code. Its central method is
  `log_prob<T: Real>(&self, theta: &[T], jacobian, propto) -> Result<T>`, monomorphized for each
  scalar type, so the inner loops have no dynamic dispatch. It also provides `model_info`,
  `write_array_impl` (constrained values, transformed parameters, generated quantities),
  `unconstrain_impl`, and `bounds`. Only `model_info` and `log_prob` are required; the others
  have metadata-driven defaults.
* `Model` is object safe and is implemented for every `StanModel` by a blanket implementation.
  It validates lengths, chooses the scalar type, runs the stanmath-rs drivers, and returns
  structured errors. Inference code depends only on `Model`.

A model whose code cannot implement `StanModel` (the current stanc-rs output, see
[compatibility](compatibility.md)) can implement `Model` directly; `stan-model-rs-stanc` does this.

## Modules of `stan-model-rs`

| Module | Responsibility |
| --- | --- |
| `meta` | `ModelInfo` and `VarInfo`: blocks, shapes, constraints, counts, flattened names, storage and output orders, JSON serialization |
| `data` | CmdStan JSON data, read once into storage order with shape and constraint checks |
| `transform` | `Reader` over the unconstrained vector; constraining, freeing, and checking each variable with the stanmath-rs transforms; Jacobian accumulation |
| `model` | The two traits, `LogDensityOptions`, the evaluation drivers, gradient checks, random initialization |
| `rng` | Seeded, independent random number streams |
| `error` | `ModelError` and `ErrorKind` |
| `nuts` | Optional `nuts-rs` adapter |

## What lives elsewhere

* Parsing, type checking, and code generation: stanc-rs.
* Special functions, densities, transforms, automatic differentiation: stanmath-rs. The runtime
  never reimplements a transform; it decides which transform applies to which slice of the
  unconstrained vector, in which order, and whether its Jacobian is added.
* Samplers, optimizers, adaptation, chains: stan-inference-rs and stan-services-rs.

## Thread safety

Model instances hold only immutable data. Every evaluation creates its own autodiff tape, so
there is no shared mutable state: a model that is `Sync` can be evaluated from many threads at
once (tested with eight threads against serial results). Random number generators are passed in
explicitly and never shared by the runtime.

## Performance

Metadata is built once at construction, data are parsed once, and the drivers evaluate the
value and gradient in one reverse pass. Dense Hessians cost `n` reverse passes; Hessian-vector
products cost about two gradients and should be preferred when they suffice. `cargo bench -p
stan-model-rs` measures the regression model. Constraining a parameter currently allocates a
small vector per variable; tape reuse across evaluations (stanmath-rs `GradientWorkspace`) is
not wired in yet, see the [roadmap](roadmap.md).
