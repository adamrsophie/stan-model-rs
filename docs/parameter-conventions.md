# Parameter conventions

All conventions were checked against the C++ that stanc 2.40 generates.

## Storage order and output order

* **Storage order** (the unconstrained vector, `Reader`, data in memory): array elements
  row-major (last array index fastest), each vector contiguous, each matrix column-major.
  This is the order in which stanc3's deserializer reads parameters.
* **Arrays of tuples** are the exception: stanc3 reads their elements with the array indices in
  column-major order, each element's components in order. The runtime stores them the same way.
* **Output order** (`write_array`, `param_names`, `param_unconstrain` input): column-major over
  all indices (first index fastest), complex values as real then imaginary part, tuples element
  by element with components in order.

`VarInfo::to_output_order` and `to_storage_order` convert between the two.

## Names

| Declaration | Output names |
| --- | --- |
| `real mu` | `mu` |
| `vector[2] beta` | `beta.1`, `beta.2` |
| `matrix[2, 2] Sigma` | `Sigma.1.1`, `Sigma.2.1`, `Sigma.1.2`, `Sigma.2.2` |
| `array[2] vector[3] a` | `a.1.1`, `a.2.1`, `a.1.2`, ... |
| `complex z` | `z.real`, `z.imag` |
| `tuple(real, vector[2]) p` | `p:1`, `p:2.1`, `p:2.2` |
| `array[2, 3] tuple(real, simplex[3]) q` | `q.1.1:1`, `q.1.1:2.1`, ..., `q.2.1:1`, ... |

Unconstrained names follow stanc3's `unconstrained_param_names`: the declared indices, except
that a transform that changes the size has reduced free dimensions:

| Type | Free dimensions | Unconstrained values |
| --- | --- | --- |
| `simplex[K]`, `sum_to_zero_vector[K]` | `K - 1` | `K - 1` |
| `unit_vector[K]`, `ordered[K]`, `positive_ordered[K]` | `K` | `K` |
| `sum_to_zero_matrix[N, M]` | `N - 1`, `M - 1` | `(N - 1)(M - 1)` |
| `column_stochastic_matrix[N, M]` | `N - 1`, `M` | `(N - 1) M` |
| `row_stochastic_matrix[N, M]` | `N`, `M - 1` | `N (M - 1)` |
| `cholesky_factor_corr[K]`, `corr_matrix[K]` | one | `K(K - 1)/2` |
| `cov_matrix[K]` | one | `K + K(K - 1)/2` |
| `cholesky_factor_cov[M, N]` | one | `N(N + 1)/2 + (M - N) N` |

Like stanc3, unconstrained names are listed in column-major order while the unconstrained vector
is in storage order, so for multidimensional arrays the i-th name does not label the i-th value
(CmdStan's `log_prob` output shows the same mismatch). Use the names as labels of the set of
values, not of positions, until Stan changes this.

## Transforms

The runtime supports these parameter constraints of Stan 2.40: lower, upper, lower and upper,
offset and multiplier, ordered, positive ordered, simplex, unit vector, sum to zero vector and
matrix, Cholesky factor of a correlation or covariance matrix, correlation matrix, covariance
matrix, column and row stochastic matrices, and tuples of these. Each transform and its log
Jacobian come from stanmath-rs. Bounds may be scalars or elementwise; bounds that depend on data
are supplied by `StanModel::bounds`, and bounds that depend on other parameters by the model's
own `unconstrain_impl` and `log_prob`.

Freeing a value checks its constraint: values outside the support are `Constraint` errors, and
values on the boundary (which map to infinite unconstrained values, for example `sigma = 0`)
are `Parameters` errors. `unit_vector` is not injective, so freeing returns the normalized
vector. Near the boundaries the transforms saturate: `exp(z)` overflows for `z > 709.78` and
underflows for `z < -745.13`; the runtime reports the resulting non-finite values or domain
errors and never replaces them.
