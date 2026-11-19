//! Hand-written models that implement the stan-model-rs contract the way generated code does.
//!
//! Each model documents the Stan program it implements. They serve as reference cases for the
//! numerical tests, as executable examples, and as the template for stanc-rs code generation.

use stan_model_rs::meta::{BaseType, Block, Bound, Constraint, ModelInfo, Outputs, VarInfo};
use stan_model_rs::rand::Rng;
use stan_model_rs::stanmath_rs::autodiff::{Broadcast, Promoted};
use stan_model_rs::stanmath_rs::prelude::{
    exponential_lpdf, exponential_lupdf, normal_lpdf, normal_lupdf, normal_rng,
};
use stan_model_rs::{Data, Reader, Real, Result, StanModel};

/// `normal_lpdf` or, with `propto`, `normal_lupdf`: what a `~ normal(...)` statement adds.
///
/// Data arguments must be passed as `f64`, never lifted to `T`: stanmath-rs decides which terms
/// `propto` may drop from the argument types, so a constant of an autodiff type counts as a
/// parameter and its terms are kept.
fn normal<Y, M, S>(y: Y, mu: M, sigma: S, propto: bool) -> Result<Promoted<(Y, M, S), 3>>
where
    (Y, M, S): Broadcast<3>,
{
    Ok(if propto {
        normal_lupdf(y, mu, sigma)?
    } else {
        normal_lpdf(y, mu, sigma)?
    })
}

/// `exponential_lpdf` or, with `propto`, `exponential_lupdf`.
fn exponential<Y, B>(y: Y, beta: B, propto: bool) -> Result<Promoted<(Y, B), 2>>
where
    (Y, B): Broadcast<2>,
{
    Ok(if propto {
        exponential_lupdf(y, beta)?
    } else {
        exponential_lpdf(y, beta)?
    })
}

fn data_real(name: &str) -> VarInfo {
    VarInfo::new(name, Block::Data, vec![], BaseType::Real, Constraint::None)
}

fn positive(name: &str, block: Block) -> VarInfo {
    VarInfo::new(
        name,
        block,
        vec![],
        BaseType::Real,
        Constraint::Lower(Bound::Value(0.0)),
    )
}

/// ```stan
/// data { real y; }
/// parameters { real mu; }
/// model {
///   mu ~ normal(0, 5);
///   y ~ normal(mu, 1);
/// }
/// ```
#[derive(Clone, Debug)]
pub struct NormalMean {
    info: ModelInfo,
    y: f64,
}

impl NormalMean {
    /// Reads and validates the data.
    pub fn new(data: &Data) -> Result<Self> {
        let y_var = data_real("y");
        let y = data.read_checked(&y_var)?[0];
        let info = ModelInfo::new("normal_mean_model", vec![y_var, VarInfo::real_param("mu")]);
        Ok(NormalMean { info, y })
    }
}

impl StanModel for NormalMean {
    fn model_info(&self) -> &ModelInfo {
        &self.info
    }

    fn log_prob<T: Real>(&self, theta: &[T], _jacobian: bool, propto: bool) -> Result<T> {
        let mu = theta[0];
        let mut lp = normal(mu, 0.0, 5.0, propto)?;
        lp += normal(self.y, mu, 1.0, propto)?;
        Ok(lp)
    }
}

/// ```stan
/// data { real<lower=0> y; }
/// parameters { real<lower=0> sigma; }
/// model {
///   sigma ~ exponential(1);
///   y ~ normal(0, sigma);
/// }
/// ```
#[derive(Clone, Debug)]
pub struct NormalScale {
    info: ModelInfo,
    y: f64,
}

impl NormalScale {
    /// Reads and validates the data.
    pub fn new(data: &Data) -> Result<Self> {
        let y_var = positive("y", Block::Data);
        let y = data.read_checked(&y_var)?[0];
        let info = ModelInfo::new(
            "normal_scale_model",
            vec![y_var, positive("sigma", Block::Parameters)],
        );
        Ok(NormalScale { info, y })
    }
}

impl StanModel for NormalScale {
    fn model_info(&self) -> &ModelInfo {
        &self.info
    }

    fn log_prob<T: Real>(&self, theta: &[T], jacobian: bool, propto: bool) -> Result<T> {
        let mut lp = T::zero();
        let sigma =
            Reader::new(theta).read_var(&self.info.variables[1], jacobian.then_some(&mut lp))?[0];
        lp += exponential(sigma, 1.0, propto)?;
        lp += normal(self.y, 0.0, sigma, propto)?;
        Ok(lp)
    }
}

/// ```stan
/// data {
///   int<lower=1> N;
///   vector[N] x;
///   vector[N] y;
/// }
/// parameters {
///   real alpha;
///   real beta;
///   real<lower=0> sigma;
/// }
/// transformed parameters {
///   real log_sigma = log(sigma);
/// }
/// model {
///   alpha ~ normal(0, 10);
///   beta ~ normal(0, 10);
///   sigma ~ exponential(1);
///   y ~ normal(alpha + beta * x, sigma);
/// }
/// generated quantities {
///   array[N] real y_rep = normal_rng(alpha + beta * x, sigma);
/// }
/// ```
#[derive(Clone, Debug)]
pub struct LinearRegression {
    info: ModelInfo,
    x: Vec<f64>,
    y: Vec<f64>,
}

impl LinearRegression {
    /// Reads and validates the data. `N` is read first because the other sizes depend on it.
    pub fn new(data: &Data) -> Result<Self> {
        let n_var = VarInfo::new(
            "N",
            Block::Data,
            vec![],
            BaseType::Int,
            Constraint::Lower(Bound::Value(1.0)),
        );
        let n = data.read_checked(&n_var)?[0] as usize;
        let vector = |name: &str| {
            VarInfo::new(
                name,
                Block::Data,
                vec![],
                BaseType::Vector(n),
                Constraint::None,
            )
        };
        let (x_var, y_var) = (vector("x"), vector("y"));
        let x = data.read_checked(&x_var)?;
        let y = data.read_checked(&y_var)?;
        let info = ModelInfo::new(
            "linear_regression_model",
            vec![
                n_var,
                x_var,
                y_var,
                VarInfo::real_param("alpha"),
                VarInfo::real_param("beta"),
                positive("sigma", Block::Parameters),
                VarInfo::new(
                    "log_sigma",
                    Block::TransformedParameters,
                    vec![],
                    BaseType::Real,
                    Constraint::None,
                ),
                VarInfo::new(
                    "y_rep",
                    Block::GeneratedQuantities,
                    vec![n],
                    BaseType::Real,
                    Constraint::None,
                ),
            ],
        );
        Ok(LinearRegression { info, x, y })
    }

    fn params<T: Real>(&self, theta: &[T], lp: Option<&mut T>) -> Result<(T, T, T)> {
        let mut r = Reader::new(theta);
        let p: Vec<&VarInfo> = self.info.parameters().collect();
        let alpha = r.read_var(p[0], None)?[0];
        let beta = r.read_var(p[1], None)?[0];
        let sigma = r.read_var(p[2], lp)?[0];
        Ok((alpha, beta, sigma))
    }
}

impl StanModel for LinearRegression {
    fn model_info(&self) -> &ModelInfo {
        &self.info
    }

    fn log_prob<T: Real>(&self, theta: &[T], jacobian: bool, propto: bool) -> Result<T> {
        let mut lp = T::zero();
        let (alpha, beta, sigma) = self.params(theta, jacobian.then_some(&mut lp))?;
        // Transformed parameters are evaluated (and their constraints checked) even when the
        // model block does not use them, as in Stan.
        let _log_sigma = sigma.ln();
        lp += normal(alpha, 0.0, 10.0, propto)?;
        lp += normal(beta, 0.0, 10.0, propto)?;
        lp += exponential(sigma, 1.0, propto)?;
        for (&xi, &yi) in self.x.iter().zip(&self.y) {
            lp += normal(yi, alpha + beta * xi, sigma, propto)?;
        }
        Ok(lp)
    }

    fn write_array_impl(
        &self,
        theta: &[f64],
        outputs: Outputs,
        rng: &mut dyn Rng,
    ) -> Result<Vec<f64>> {
        let (alpha, beta, sigma) = self.params(theta, None)?;
        let mut out = vec![alpha, beta, sigma];
        if outputs.transformed_parameters {
            out.push(sigma.ln());
        }
        if outputs.generated_quantities {
            for &xi in &self.x {
                out.push(normal_rng(alpha + beta * xi, sigma, rng)?);
            }
        }
        Ok(out)
    }
}

/// Parameters with structured constraints, to exercise ordering and Jacobians:
///
/// ```stan
/// parameters {
///   simplex[3] theta;
///   array[2] vector<lower=0>[2] tau;
///   ordered[2] c;
/// }
/// model {
///   target += dirichlet_lupdf(theta | [2, 3, 4]');   // written out below without constants
///   for (i in 1:2) tau[i] ~ exponential(1);
///   c ~ normal(0, 3);
/// }
/// ```
#[derive(Clone, Debug)]
pub struct Structured {
    info: ModelInfo,
}

impl Structured {
    /// The model has no data.
    pub fn new() -> Self {
        let p = |name: &str, arr, base, c| VarInfo::new(name, Block::Parameters, arr, base, c);
        let info = ModelInfo::new(
            "structured_model",
            vec![
                p("theta", vec![], BaseType::Vector(3), Constraint::Simplex),
                p(
                    "tau",
                    vec![2],
                    BaseType::Vector(2),
                    Constraint::Lower(Bound::Value(0.0)),
                ),
                p("c", vec![], BaseType::Vector(2), Constraint::Ordered),
            ],
        );
        Structured { info }
    }
}

impl Default for Structured {
    fn default() -> Self {
        Self::new()
    }
}

impl StanModel for Structured {
    fn model_info(&self) -> &ModelInfo {
        &self.info
    }

    fn log_prob<T: Real>(&self, theta: &[T], jacobian: bool, propto: bool) -> Result<T> {
        let mut lp = T::zero();
        let mut jac = T::zero();
        let mut r = Reader::new(theta);
        let vars: Vec<&VarInfo> = self.info.parameters().collect();
        let mut read = |var: &VarInfo| r.read_var(var, jacobian.then_some(&mut jac));
        let simplex = read(vars[0])?;
        let tau = read(vars[1])?;
        let c = read(vars[2])?;
        for (k, s) in simplex.iter().enumerate() {
            lp += s.ln() * (k as f64 + 1.0);
        }
        for &t in &tau {
            lp += exponential(t, 1.0, propto)?;
        }
        for &ci in &c {
            lp += normal(ci, 0.0, 3.0, propto)?;
        }
        Ok(lp + jac)
    }
}

/// A model without data or parameters: `model { }`. Its log density is 0.
#[derive(Clone, Debug)]
pub struct Empty {
    info: ModelInfo,
}

impl Empty {
    /// The empty model.
    pub fn new() -> Self {
        Empty {
            info: ModelInfo::new("empty_model", vec![]),
        }
    }
}

impl Default for Empty {
    fn default() -> Self {
        Self::new()
    }
}

impl StanModel for Empty {
    fn model_info(&self) -> &ModelInfo {
        &self.info
    }

    fn log_prob<T: Real>(&self, _theta: &[T], _jacobian: bool, _propto: bool) -> Result<T> {
        Ok(T::zero())
    }
}
