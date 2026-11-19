//! Composition of the constraining transforms: reading a parameter from the unconstrained vector,
//! constraining it, accumulating its log Jacobian, and the inverse (freeing) transforms.
//!
//! The mathematics of each transform is in stanmath-rs; this module decides, from a variable's
//! [`VarInfo`], which transform applies to which slice of the unconstrained vector and in what
//! order. Values are in storage order (see [`crate::meta`]): array elements row-major, each vector
//! contiguous, each matrix column-major.

use stanmath_rs::Matrix;
use stanmath_rs::prelude::*;

use crate::meta::{BaseType, Bound, Constraint, VarInfo};
use crate::{ErrorKind, ModelError, Result};

/// The values of a variable's bounds, resolved when the model runs.
///
/// A bound is either one value for every scalar of the variable, or one value per scalar in
/// storage order. Bounds are data (`f64`); a bound that depends on parameters is applied by the
/// model code itself with the stanmath-rs transform functions.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Bounds {
    /// Lower bound, or offset for `<offset, multiplier>`.
    pub lower: Option<BoundValue>,
    /// Upper bound, or multiplier for `<offset, multiplier>`.
    pub upper: Option<BoundValue>,
}

/// A resolved bound.
#[derive(Clone, Debug, PartialEq)]
pub enum BoundValue {
    /// The same value for every scalar.
    Scalar(f64),
    /// One value per scalar, in storage order.
    Each(Vec<f64>),
}

impl BoundValue {
    fn at(&self, i: usize, n: usize, name: &str) -> Result<f64> {
        match self {
            BoundValue::Scalar(v) => Ok(*v),
            BoundValue::Each(v) if v.len() == n => Ok(v[i]),
            BoundValue::Each(v) => Err(ModelError::new(
                ErrorKind::Size,
                format!("bound has {} values but the variable has {n}", v.len()),
            )
            .with_variable(name)),
        }
    }
}

impl Bounds {
    /// No bounds.
    pub const NONE: Bounds = Bounds {
        lower: None,
        upper: None,
    };

    /// The bounds written in `var`'s metadata. Fails with [`ErrorKind::Unsupported`] if a bound
    /// is an expression, which only the model code can evaluate.
    pub fn from_metadata(var: &VarInfo) -> Result<Bounds> {
        let value = |b: Bound| match b {
            Bound::Value(v) => Ok(Some(BoundValue::Scalar(v))),
            Bound::Expression => Err(ModelError::new(
                ErrorKind::Unsupported,
                "the bound is an expression; the model must supply its value",
            )
            .with_variable(var.name.clone())),
        };
        Ok(match var.constraint {
            Constraint::Lower(l) => Bounds {
                lower: value(l)?,
                upper: None,
            },
            Constraint::Upper(u) => Bounds {
                lower: None,
                upper: value(u)?,
            },
            Constraint::LowerUpper(l, u) | Constraint::OffsetMultiplier(l, u) => Bounds {
                lower: value(l)?,
                upper: value(u)?,
            },
            _ => Bounds::NONE,
        })
    }

    fn get(b: &Option<BoundValue>, i: usize, n: usize, name: &str) -> Result<f64> {
        match b {
            Some(b) => b.at(i, n, name),
            None => Err(ModelError::internal("missing bound value").with_variable(name)),
        }
    }
}

/// Constrains the unconstrained values `y` of one element (one array cell) of `var`. `offset` is
/// the storage position of the element's first constrained scalar, for elementwise bounds.
fn constrain_element<T: Real>(
    var: &VarInfo,
    bounds: &Bounds,
    y: &[T],
    offset: usize,
    lp: &mut Option<&mut T>,
) -> Result<Vec<T>> {
    let n = var.num_constrained();
    let name = var.name.as_str();
    let lo = |i: usize| Bounds::get(&bounds.lower, offset + i, n, name);
    let hi = |i: usize| Bounds::get(&bounds.upper, offset + i, n, name);
    let dims = var.base.dims();
    let mut dummy = T::zero();
    let jac = lp.is_some();
    let acc: &mut T = match lp {
        Some(l) => l,
        None => &mut dummy,
    };
    let elementwise = |f: &mut dyn FnMut(usize, T) -> Result<T>| -> Result<Vec<T>> {
        y.iter().enumerate().map(|(i, &v)| f(i, v)).collect()
    };
    let out = match var.constraint {
        Constraint::None => y.to_vec(),
        Constraint::Lower(_) => elementwise(&mut |i, v| {
            let l = lo(i)?;
            Ok(if jac {
                lb_constrain_lp(v, l, acc)
            } else {
                lb_constrain(v, l)
            })
        })?,
        Constraint::Upper(_) => elementwise(&mut |i, v| {
            let u = hi(i)?;
            Ok(if jac {
                ub_constrain_lp(v, u, acc)
            } else {
                ub_constrain(v, u)
            })
        })?,
        Constraint::LowerUpper(..) => elementwise(&mut |i, v| {
            let (l, u) = (lo(i)?, hi(i)?);
            Ok(if jac {
                lub_constrain_lp(v, l, u, acc)?
            } else {
                lub_constrain(v, l, u)?
            })
        })?,
        Constraint::OffsetMultiplier(..) => elementwise(&mut |i, v| {
            let (m, s) = (lo(i)?, hi(i)?);
            Ok(if jac {
                offset_multiplier_constrain_lp(v, m, s, acc)?
            } else {
                offset_multiplier_constrain(v, m, s)?
            })
        })?,
        Constraint::Ordered => {
            if jac {
                ordered_constrain_lp(y, acc)
            } else {
                ordered_constrain(y)
            }
        }
        Constraint::PositiveOrdered => {
            if jac {
                positive_ordered_constrain_lp(y, acc)
            } else {
                positive_ordered_constrain(y)
            }
        }
        Constraint::Simplex => {
            if jac {
                simplex_constrain_lp(y, acc)
            } else {
                simplex_constrain(y)
            }
        }
        Constraint::SumToZero => sum_to_zero_constrain(y),
        Constraint::UnitVector => {
            if jac {
                unit_vector_constrain_lp(y, acc)?
            } else {
                unit_vector_constrain(y)?
            }
        }
        Constraint::CholeskyFactorCorr => {
            let k = dims[0];
            if jac {
                cholesky_factor_corr_constrain_lp(y, k, acc)?
            } else {
                cholesky_factor_corr_constrain(y, k)?
            }
            .into_vec()
        }
        Constraint::CholeskyFactorCov => {
            let (m, c) = (dims[0], dims[1]);
            if jac {
                cholesky_factor_cov_constrain_lp(y, m, c, acc)?
            } else {
                cholesky_factor_cov_constrain(y, m, c)?
            }
            .into_vec()
        }
        Constraint::CorrMatrix => {
            let k = dims[0];
            if jac {
                corr_matrix_constrain_lp(y, k, acc)?
            } else {
                corr_matrix_constrain(y, k)?
            }
            .into_vec()
        }
        Constraint::CovMatrix => {
            let k = dims[0];
            if jac {
                cov_matrix_constrain_lp(y, k, acc)?
            } else {
                cov_matrix_constrain(y, k)?
            }
            .into_vec()
        }
        Constraint::SumToZeroMatrix => {
            let ym = Matrix::from_column_major(
                dims[0].saturating_sub(1),
                dims[1].saturating_sub(1),
                y.to_vec(),
            );
            sum_to_zero_matrix_constrain(&ym).into_vec()
        }
        Constraint::StochasticColumn => {
            let ym = Matrix::from_column_major(dims[0].saturating_sub(1), dims[1], y.to_vec());
            if jac {
                stochastic_column_constrain_lp(&ym, acc)
            } else {
                stochastic_column_constrain(&ym)
            }
            .into_vec()
        }
        Constraint::StochasticRow => {
            let ym = Matrix::from_column_major(dims[0], dims[1].saturating_sub(1), y.to_vec());
            if jac {
                stochastic_row_constrain_lp(&ym, acc)
            } else {
                stochastic_row_constrain(&ym)
            }
            .into_vec()
        }
    };
    Ok(out)
}

/// Rejects a combination of constraint and element type that Stan does not allow, so that a
/// malformed [`VarInfo`] fails with an error instead of a panic.
fn check_shape(var: &VarInfo) -> Result<()> {
    let ok = match (var.constraint, var.base) {
        (_, b) if b.is_complex() => var.constraint == Constraint::None,
        (Constraint::None, _)
        | (Constraint::Lower(_), _)
        | (Constraint::Upper(_), _)
        | (Constraint::LowerUpper(..), _)
        | (Constraint::OffsetMultiplier(..), _) => true,
        (
            Constraint::Ordered
            | Constraint::PositiveOrdered
            | Constraint::Simplex
            | Constraint::SumToZero
            | Constraint::UnitVector,
            BaseType::Vector(_),
        ) => true,
        (
            Constraint::CholeskyFactorCorr | Constraint::CorrMatrix | Constraint::CovMatrix,
            BaseType::Matrix(r, c),
        ) => r == c,
        (Constraint::CholeskyFactorCov, BaseType::Matrix(r, c)) => r >= c,
        (
            Constraint::SumToZeroMatrix | Constraint::StochasticColumn | Constraint::StochasticRow,
            BaseType::Matrix(..),
        ) => true,
        _ => false,
    };
    let ok = ok && !(var.base.is_int() && var.block == crate::Block::Parameters);
    if ok {
        Ok(())
    } else {
        Err(ModelError::internal(format!(
            "constraint {:?} is not valid for type {:?}",
            var.constraint, var.base
        ))
        .with_variable(var.name.clone()))
    }
}

/// Constrains a whole variable from its unconstrained values `y` (exactly
/// [`VarInfo::num_unconstrained`] of them), returning its constrained values in storage order.
/// With `lp`, adds the log absolute Jacobian determinant of the transform to it.
pub fn constrain<T: Real>(
    var: &VarInfo,
    bounds: &Bounds,
    y: &[T],
    mut lp: Option<&mut T>,
) -> Result<Vec<T>> {
    check_shape(var)?;
    if y.len() != var.num_unconstrained() {
        return Err(ModelError::internal(format!(
            "expected {} unconstrained values, found {}",
            var.num_unconstrained(),
            y.len()
        ))
        .with_variable(var.name.clone()));
    }
    let n_elem = var.num_elements();
    if n_elem == 0 {
        return Ok(Vec::new());
    }
    let per_y = y.len() / n_elem;
    let per_x = var.num_constrained() / n_elem;
    let mut out = Vec::with_capacity(var.num_constrained());
    for e in 0..n_elem {
        let chunk = &y[e * per_y..(e + 1) * per_y];
        let x = constrain_element(var, bounds, chunk, e * per_x, &mut lp)
            .map_err(|err| err.with_variable(var.name.clone()))?;
        out.extend(x);
    }
    Ok(out)
}

/// Inverse of [`constrain`]: the unconstrained values of `var` from its constrained values `x` in
/// storage order. Fails with [`ErrorKind::Constraint`] if `x` violates the declared constraint.
pub fn free(var: &VarInfo, bounds: &Bounds, x: &[f64]) -> Result<Vec<f64>> {
    check_shape(var)?;
    if x.len() != var.num_constrained() {
        return Err(ModelError::new(
            ErrorKind::Parameters,
            format!(
                "expected {} values, found {}",
                var.num_constrained(),
                x.len()
            ),
        )
        .with_variable(var.name.clone()));
    }
    let n_elem = var.num_elements();
    let mut out = Vec::with_capacity(var.num_unconstrained());
    if n_elem == 0 {
        return Ok(out);
    }
    let per_x = x.len() / n_elem;
    for e in 0..n_elem {
        let chunk = &x[e * per_x..(e + 1) * per_x];
        let y = free_element(var, bounds, chunk, e * per_x).map_err(|err| {
            let err = if err.kind == ErrorKind::Domain {
                ModelError {
                    kind: ErrorKind::Constraint,
                    ..err
                }
            } else {
                err
            };
            err.with_variable(var.name.clone())
        })?;
        out.extend(y);
    }
    Ok(out)
}

fn free_element(var: &VarInfo, bounds: &Bounds, x: &[f64], offset: usize) -> Result<Vec<f64>> {
    let n = var.num_constrained();
    let name = var.name.as_str();
    let lo = |i: usize| Bounds::get(&bounds.lower, offset + i, n, name);
    let hi = |i: usize| Bounds::get(&bounds.upper, offset + i, n, name);
    let dims = var.base.dims();
    let mat = || Matrix::from_column_major(dims[0], dims[1], x.to_vec());
    let elementwise = |f: &dyn Fn(usize, f64) -> Result<f64>| -> Result<Vec<f64>> {
        x.iter().enumerate().map(|(i, &v)| f(i, v)).collect()
    };
    Ok(match var.constraint {
        Constraint::None => x.to_vec(),
        Constraint::Lower(_) => elementwise(&|i, v| Ok(lb_free(v, lo(i)?)?))?,
        Constraint::Upper(_) => elementwise(&|i, v| Ok(ub_free(v, hi(i)?)?))?,
        Constraint::LowerUpper(..) => elementwise(&|i, v| Ok(lub_free(v, lo(i)?, hi(i)?)?))?,
        Constraint::OffsetMultiplier(..) => {
            elementwise(&|i, v| Ok(offset_multiplier_free(v, lo(i)?, hi(i)?)?))?
        }
        Constraint::Ordered => ordered_free(x)?,
        Constraint::PositiveOrdered => positive_ordered_free(x)?,
        Constraint::Simplex => simplex_free(x)?,
        Constraint::SumToZero => sum_to_zero_free(x)?,
        Constraint::UnitVector => unit_vector_free(x)?,
        Constraint::CholeskyFactorCorr => cholesky_factor_corr_free(&mat())?,
        Constraint::CholeskyFactorCov => cholesky_factor_cov_free(&mat())?,
        Constraint::CorrMatrix => corr_matrix_free(&mat())?,
        Constraint::CovMatrix => cov_matrix_free(&mat())?,
        Constraint::SumToZeroMatrix => sum_to_zero_matrix_free(&mat())?.into_vec(),
        Constraint::StochasticColumn => stochastic_column_free(&mat())?.into_vec(),
        Constraint::StochasticRow => stochastic_row_free(&mat())?.into_vec(),
    })
}

/// Checks that the values `x` (storage order) of a data, transformed data, transformed
/// parameter, or generated quantity variable satisfy its declared constraint.
///
/// Bounds are inclusive, as in Stan. Structured constraints (simplex, correlation matrix, ...) are
/// checked to the tolerance of the stanmath-rs inverse transforms.
pub fn check(var: &VarInfo, bounds: &Bounds, x: &[f64]) -> Result<()> {
    if x.len() != var.num_constrained() {
        return Err(ModelError::new(
            ErrorKind::Size,
            format!(
                "expected {} values, found {}",
                var.num_constrained(),
                x.len()
            ),
        )
        .with_variable(var.name.clone()));
    }
    if var.constraint == Constraint::None || var.base.is_complex() {
        return Ok(());
    }
    // Scalar bounds (also on integers, exact in f64 for Stan's 32-bit integers): compare directly
    // so that values on the bound pass and the message names the offending element.
    if matches!(
        var.constraint,
        Constraint::Lower(_) | Constraint::Upper(_) | Constraint::LowerUpper(..)
    ) {
        let n = x.len();
        for (i, &v) in x.iter().enumerate() {
            let lower = match &bounds.lower {
                Some(b) => Some(b.at(i, n, &var.name)?),
                None => None,
            };
            let upper = match &bounds.upper {
                Some(b) => Some(b.at(i, n, &var.name)?),
                None => None,
            };
            if v.is_nan() || lower.is_some_and(|l| !(v >= l)) || upper.is_some_and(|u| !(v <= u)) {
                return Err(violation(var, i, v, lower, upper));
            }
        }
        return Ok(());
    }
    if matches!(var.constraint, Constraint::OffsetMultiplier(..)) {
        return Ok(());
    }
    free(var, bounds, x).map(|_| ())
}

fn violation(
    var: &VarInfo,
    i: usize,
    v: f64,
    lower: Option<f64>,
    upper: Option<f64>,
) -> ModelError {
    let at = if var.num_constrained() > 1 {
        format!("[{}]", i + 1)
    } else {
        String::new()
    };
    let what = match (lower, upper) {
        (Some(l), Some(u)) => format!("in the interval [{l}, {u}]"),
        (Some(l), None) => format!("greater than or equal to {l}"),
        (None, Some(u)) => format!("less than or equal to {u}"),
        (None, None) => "valid".to_string(),
    };
    ModelError::constraint(format!("element{at} is {v}, but must be {what}"))
        .with_variable(var.name.clone())
        .with_value(v)
}

/// Sequential reader of an unconstrained parameter vector, used by model code to obtain each
/// parameter's constrained value in declaration order.
pub struct Reader<'a, T> {
    theta: &'a [T],
    pos: usize,
}

impl<'a, T: Real> Reader<'a, T> {
    /// Starts reading at the beginning of `theta`.
    pub fn new(theta: &'a [T]) -> Self {
        Reader { theta, pos: 0 }
    }

    /// Reads the next parameter, constrains it, and with `lp` adds its log Jacobian.
    /// Returns its constrained values in storage order.
    pub fn read(&mut self, var: &VarInfo, bounds: &Bounds, lp: Option<&mut T>) -> Result<Vec<T>> {
        let n = var.num_unconstrained();
        let end = self.pos + n;
        if end > self.theta.len() {
            return Err(ModelError::parameters(format!(
                "parameter vector too short: needed {end} values, found {}",
                self.theta.len()
            )));
        }
        let x = constrain(var, bounds, &self.theta[self.pos..end], lp)?;
        self.pos = end;
        Ok(x)
    }

    /// Reads the next parameter, whose bounds are all constants in its metadata.
    pub fn read_var(&mut self, var: &VarInfo, lp: Option<&mut T>) -> Result<Vec<T>> {
        self.read(var, &Bounds::from_metadata(var)?, lp)
    }

    /// Number of values read so far.
    pub fn position(&self) -> usize {
        self.pos
    }

    /// Fails unless every value was read.
    pub fn finish(&self) -> Result<()> {
        if self.pos == self.theta.len() {
            Ok(())
        } else {
            Err(ModelError::parameters(format!(
                "expected {} unconstrained parameters, found {}",
                self.pos,
                self.theta.len()
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::meta::Block;

    fn var(base: BaseType, arr: Vec<usize>, c: Constraint) -> VarInfo {
        VarInfo::new("x", Block::Parameters, arr, base, c)
    }

    fn close(a: &[f64], b: &[f64], tol: f64) -> bool {
        a.len() == b.len()
            && a.iter()
                .zip(b)
                .all(|(x, y)| (x - y).abs() <= tol * (1.0 + y.abs()))
    }

    /// Every supported constraint: constrain then free returns the input, and the constrained
    /// value has the declared size.
    #[test]
    fn round_trips() {
        let cases = [
            var(
                BaseType::Real,
                vec![3],
                Constraint::Lower(Bound::Value(1.0)),
            ),
            var(
                BaseType::Real,
                vec![],
                Constraint::Upper(Bound::Value(-2.0)),
            ),
            var(
                BaseType::Vector(2),
                vec![2],
                Constraint::LowerUpper(Bound::Value(-1.0), Bound::Value(3.0)),
            ),
            var(
                BaseType::Real,
                vec![],
                Constraint::OffsetMultiplier(Bound::Value(1.0), Bound::Value(2.0)),
            ),
            var(BaseType::Vector(4), vec![], Constraint::Ordered),
            var(BaseType::Vector(3), vec![2], Constraint::PositiveOrdered),
            var(BaseType::Vector(4), vec![2], Constraint::Simplex),
            var(BaseType::Vector(4), vec![], Constraint::SumToZero),
            var(BaseType::Vector(3), vec![], Constraint::UnitVector),
            var(
                BaseType::Matrix(3, 3),
                vec![],
                Constraint::CholeskyFactorCorr,
            ),
            var(
                BaseType::Matrix(4, 3),
                vec![],
                Constraint::CholeskyFactorCov,
            ),
            var(BaseType::Matrix(3, 3), vec![2], Constraint::CorrMatrix),
            var(BaseType::Matrix(3, 3), vec![], Constraint::CovMatrix),
            var(BaseType::Matrix(3, 4), vec![], Constraint::SumToZeroMatrix),
            var(BaseType::Matrix(3, 2), vec![], Constraint::StochasticColumn),
            var(BaseType::Matrix(2, 3), vec![], Constraint::StochasticRow),
        ];
        for v in &cases {
            let b = Bounds::from_metadata(v).unwrap();
            let n = v.num_unconstrained();
            let y: Vec<f64> = (0..n).map(|i| 0.3 * (i as f64) - 0.4).collect();
            let x = constrain::<f64>(v, &b, &y, None).unwrap();
            assert_eq!(x.len(), v.num_constrained(), "{:?}", v.constraint);
            check(v, &b, &x).unwrap_or_else(|e| panic!("{:?}: {e}", v.constraint));
            let back = free(v, &b, &x).unwrap();
            // The unit vector transform is not injective (any positive multiple of y maps to the
            // same x); its inverse returns the normalized vector.
            if v.constraint != Constraint::UnitVector {
                assert!(
                    close(&back, &y, 1e-9),
                    "{:?}: {back:?} vs {y:?}",
                    v.constraint
                );
            }
        }
    }

    #[test]
    fn jacobian_of_lower_bound() {
        let v = var(
            BaseType::Real,
            vec![2],
            Constraint::Lower(Bound::Value(0.0)),
        );
        let mut lp = 0.0;
        let x = constrain(
            &v,
            &Bounds::from_metadata(&v).unwrap(),
            &[0.5, -1.0],
            Some(&mut lp),
        )
        .unwrap();
        assert!(close(&x, &[0.5f64.exp(), (-1.0f64).exp()], 1e-15));
        assert!((lp - (-0.5)).abs() < 1e-15);
    }

    #[test]
    fn elementwise_bounds_and_violations() {
        let v = var(
            BaseType::Vector(2),
            vec![],
            Constraint::Lower(Bound::Expression),
        );
        assert_eq!(
            Bounds::from_metadata(&v).unwrap_err().kind,
            ErrorKind::Unsupported
        );
        let b = Bounds {
            lower: Some(BoundValue::Each(vec![0.0, 10.0])),
            upper: None,
        };
        let x = constrain::<f64>(&v, &b, &[0.0, 0.0], None).unwrap();
        assert_eq!(x, [1.0, 11.0]);
        let e = free(&v, &b, &[1.0, 5.0]).unwrap_err();
        assert_eq!(e.kind, ErrorKind::Constraint);
        let e = check(&v, &b, &[1.0, 5.0]).unwrap_err();
        assert_eq!(e.kind, ErrorKind::Constraint);
        assert_eq!(e.variable.as_deref(), Some("x"));
        // Values on the bound are allowed.
        check(&v, &b, &[0.0, 10.0]).unwrap();
        // A NaN never satisfies a bound.
        assert!(check(&v, &b, &[f64::NAN, 11.0]).is_err());
        let wrong = Bounds {
            lower: Some(BoundValue::Each(vec![0.0])),
            upper: None,
        };
        assert_eq!(
            free(&v, &wrong, &[1.0, 5.0]).unwrap_err().kind,
            ErrorKind::Size
        );
    }

    #[test]
    fn invalid_structured_values_are_rejected() {
        let s = var(BaseType::Vector(3), vec![], Constraint::Simplex);
        assert!(free(&s, &Bounds::NONE, &[0.5, 0.6, -0.1]).is_err());
        let o = var(BaseType::Vector(3), vec![], Constraint::Ordered);
        assert_eq!(
            free(&o, &Bounds::NONE, &[1.0, 0.0, 2.0]).unwrap_err().kind,
            ErrorKind::Constraint
        );
        let bad = var(BaseType::Vector(3), vec![], Constraint::CorrMatrix);
        assert_eq!(
            free(&bad, &Bounds::NONE, &[0.0; 3]).unwrap_err().kind,
            ErrorKind::Internal
        );
    }

    #[test]
    fn reader_reads_in_order() {
        let a = var(BaseType::Real, vec![], Constraint::None);
        let s = var(BaseType::Vector(3), vec![], Constraint::Simplex);
        let theta = [1.0, 0.0, 0.0];
        let mut r = Reader::new(&theta[..]);
        assert_eq!(r.read_var(&a, None).unwrap(), [1.0]);
        let x = r.read_var(&s, None).unwrap();
        assert!(close(&x, &[1.0 / 3.0; 3], 1e-12));
        r.finish().unwrap();
        assert_eq!(
            r.read_var(&a, None).unwrap_err().kind,
            ErrorKind::Parameters
        );
    }
}
