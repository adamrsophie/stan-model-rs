//! Regression tests for problems reported against the reference implementations (the
//! stan-dev/stan and roualdes/bridgestan issue trackers) that concern the model runtime. Each
//! test names the issues it covers.

use stan_model_rs::meta::{BaseType, Block, Bound, Constraint, ModelInfo, Outputs, VarInfo};
use stan_model_rs::transform::{Bounds, check, free};
use stan_model_rs::{
    Data, ErrorKind, LogDensityOptions, Model, Reader, Real, Result, StanModel, rng,
};
use stan_model_rs_fixtures::LinearRegression;

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * (1.0 + b.abs())
}

fn param(name: &str, array_dims: Vec<usize>, base: BaseType, c: Constraint) -> VarInfo {
    VarInfo::new(name, Block::Parameters, array_dims, base, c)
}

/// A model whose target is only the Jacobian of its parameters' transforms, built from metadata.
struct Params(ModelInfo);

impl Params {
    fn new(vars: Vec<VarInfo>) -> Self {
        Params(ModelInfo::new("params_model", vars))
    }
}

impl StanModel for Params {
    fn model_info(&self) -> &ModelInfo {
        &self.0
    }

    fn log_prob<T: Real>(&self, theta: &[T], jacobian: bool, _propto: bool) -> Result<T> {
        let mut jac = T::zero();
        let mut r = Reader::new(theta);
        for v in self.0.parameters() {
            r.read_var(v, jacobian.then_some(&mut jac))?;
        }
        r.finish()?;
        Ok(jac)
    }
}

fn regression() -> LinearRegression {
    let data = r#"{"N": 4, "x": [0.0, 1.0, 2.0, 3.5], "y": [0.3, 1.1, 2.4, 3.9]}"#;
    LinearRegression::new(&Data::from_json_str(data).unwrap()).unwrap()
}

fn data(json: &str) -> Data {
    Data::from_json_str(json).unwrap()
}

/// stan #2686, #637: Stan's JSON reader accepts bare `NaN` and `Infinity` tokens, which strict
/// JSON rejects; CmdStan writes them. String spellings and look-alike keys are unaffected.
#[test]
fn json_non_finite_tokens() {
    let v = VarInfo::new("x", Block::Data, vec![5], BaseType::Real, Constraint::None);
    let d = data(r#"{"x": [NaN, Infinity, -Infinity, Inf, -Inf], "NaN": "Infinity"}"#);
    let x = d.read(&v).unwrap();
    assert!(x[0].is_nan());
    assert_eq!(
        &x[1..],
        [
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY
        ]
    );
    assert!(d.contains("NaN"));
    // Not a number literal: an identifier that only starts like one is still invalid JSON.
    assert!(Data::from_json_str(r#"{"x": NaNa}"#).is_err());
    // Integers cannot be non-finite.
    let n = VarInfo::new("n", Block::Data, vec![], BaseType::Int, Constraint::None);
    assert_eq!(
        data(r#"{"n": NaN}"#).read(&n).unwrap_err().kind,
        ErrorKind::Data
    );
}

/// stan #683, #1909, #2468, #2867, #1435: containers with a zero dimension, given as empty lists
/// or omitted, are valid data.
#[test]
fn zero_size_data() {
    let cases = [
        (r#"{"x": []}"#, vec![], BaseType::Vector(0)),
        (r#"{"x": []}"#, vec![], BaseType::Matrix(0, 3)),
        (r#"{"x": [[], []]}"#, vec![], BaseType::Matrix(2, 0)),
        (r#"{"x": [[], []]}"#, vec![2, 0], BaseType::Real),
        (r#"{"x": []}"#, vec![0], BaseType::Vector(3)),
        (r#"{}"#, vec![2, 0], BaseType::Real),
    ];
    for (json, dims, base) in cases {
        let v = VarInfo::new("x", Block::Data, dims, base, Constraint::None);
        assert_eq!(data(json).read(&v).unwrap(), Vec::<f64>::new(), "{json}");
    }
    // A non-empty container is still required to have the declared size.
    let v = VarInfo::new(
        "x",
        Block::Data,
        vec![],
        BaseType::Vector(2),
        Constraint::None,
    );
    assert_eq!(
        data(r#"{"x": []}"#).read(&v).unwrap_err().kind,
        ErrorKind::Data
    );
}

/// stan #670, #204, #2208: integers outside Stan's 32-bit range are rejected, large whole-number
/// reals are read as reals, and numbers beyond the double range are a data error.
#[test]
fn numeric_ranges_in_data() {
    let n = VarInfo::new("n", Block::Data, vec![], BaseType::Int, Constraint::None);
    let e = data(r#"{"n": 3000000000}"#).read(&n).unwrap_err();
    assert_eq!(
        (e.kind, e.variable.as_deref()),
        (ErrorKind::Data, Some("n"))
    );
    let x = VarInfo::new("x", Block::Data, vec![], BaseType::Real, Constraint::None);
    assert_eq!(
        data(r#"{"x": 100000000000000000000000}"#).read(&x).unwrap(),
        [1e23]
    );
    assert_eq!(
        Data::from_json_str(r#"{"x": 1e400}"#).unwrap_err().kind,
        ErrorKind::Data
    );
}

/// stan #3437, #3161: complex values inside arrays of tuples are read with the right layout.
#[test]
fn complex_inside_array_of_tuples() {
    let slots = vec![
        VarInfo::new("", Block::Data, vec![], BaseType::Complex, Constraint::None),
        VarInfo::new("", Block::Data, vec![2], BaseType::Real, Constraint::None),
    ];
    let q = VarInfo::tuple("q", Block::Data, vec![2], slots);
    let d = data(r#"{"q": [{"1": [1, 2], "2": [3, 4]}, {"1": [5, 6], "2": [7, 8]}]}"#);
    let x = d.read(&q).unwrap();
    assert_eq!(x, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    assert_eq!(
        q.output_names(),
        [
            "q.1:1.real",
            "q.1:1.imag",
            "q.1:2.1",
            "q.1:2.2",
            "q.2:1.real",
            "q.2:1.imag",
            "q.2:2.1",
            "q.2:2.2"
        ]
    );
}

/// bridgestan #278: a variable with no elements has no flattened names but stays visible, with
/// its shape, in the metadata.
#[test]
fn empty_variables_are_listed() {
    let m = Params::new(vec![
        param("a", vec![10, 0], BaseType::Real, Constraint::None),
        param("b", vec![], BaseType::Real, Constraint::None),
    ]);
    assert_eq!(m.param_names(Outputs::ALL), ["b"]);
    let a = m.info().variable("a").unwrap();
    assert_eq!((a.dims(), a.num_constrained()), (vec![10, 0], 0));
    let json = m.info().to_json();
    assert_eq!(
        ModelInfo::from_json(&json)
            .unwrap()
            .variable("a")
            .unwrap()
            .dims(),
        [10, 0]
    );
}

/// stan #2455: the unit vector "Jacobian" is `-0.5 y'y`, Stan's density on the unconstrained
/// space that keeps the radius proper.
#[test]
fn unit_vector_adjustment() {
    let m = Params::new(vec![param(
        "u",
        vec![],
        BaseType::Vector(3),
        Constraint::UnitVector,
    )]);
    let y = [0.3, -1.2, 2.0];
    let lp = m.log_density(&y, LogDensityOptions::default()).unwrap();
    assert!(close(lp, -0.5 * (0.09 + 1.44 + 4.0), 1e-14));
    let u = m.param_constrain(&y).unwrap();
    assert!(close(u.iter().map(|v| v * v).sum::<f64>(), 1.0, 1e-14));
    // Freeing the zero vector, which is not a unit vector, is an error.
    let v = param("u", vec![], BaseType::Vector(3), Constraint::UnitVector);
    assert!(free(&v, &Bounds::NONE, &[0.0, 0.0, 0.0]).is_err());
}

/// stan #432, #2994: an infinite bound is the same as no bound: identity transform, no
/// Jacobian. One finite side of a two-sided bound behaves like the one-sided bound.
#[test]
fn infinite_bounds() {
    let inf = f64::INFINITY;
    let identity = [
        Constraint::Lower(Bound::Value(-inf)),
        Constraint::Upper(Bound::Value(inf)),
        Constraint::LowerUpper(Bound::Value(-inf), Bound::Value(inf)),
    ];
    for c in identity {
        let m = Params::new(vec![param("x", vec![], BaseType::Real, c)]);
        assert_eq!(m.param_constrain(&[0.7]).unwrap(), [0.7], "{c:?}");
        assert_eq!(
            m.log_density(&[0.7], LogDensityOptions::default()).unwrap(),
            0.0,
            "{c:?}"
        );
        assert_eq!(m.param_unconstrain(&[0.7]).unwrap(), [0.7], "{c:?}");
    }
    let lower = Params::new(vec![param(
        "x",
        vec![],
        BaseType::Real,
        Constraint::Lower(Bound::Value(1.0)),
    )]);
    let half = Params::new(vec![param(
        "x",
        vec![],
        BaseType::Real,
        Constraint::LowerUpper(Bound::Value(1.0), Bound::Value(inf)),
    )]);
    for z in [-2.0, 0.0, 3.0] {
        let opts = LogDensityOptions::default();
        assert_eq!(
            half.param_constrain(&[z]).unwrap(),
            lower.param_constrain(&[z]).unwrap()
        );
        assert!(close(
            half.log_density(&[z], opts).unwrap(),
            lower.log_density(&[z], opts).unwrap(),
            1e-15
        ));
    }
}

/// stan #255, bridgestan #127: constrained values are checked with Stan's tolerance of 1e-8, so
/// rounding in stored draws is accepted and real violations are not.
#[test]
fn constraint_tolerance() {
    let v = param("s", vec![], BaseType::Vector(3), Constraint::Simplex);
    let b = Bounds::NONE;
    let third = 1.0 / 3.0;
    check(&v, &b, &[third, third, third + 1e-9]).unwrap();
    free(&v, &b, &[third, third, third + 1e-9]).unwrap();
    assert_eq!(
        check(&v, &b, &[third, third, third + 1e-6])
            .unwrap_err()
            .kind,
        ErrorKind::Constraint
    );
}

/// stan #1995: a covariance matrix must be symmetric to be freed.
#[test]
fn covariance_symmetry() {
    let v = param("S", vec![], BaseType::Matrix(2, 2), Constraint::CovMatrix);
    let b = Bounds::NONE;
    free(&v, &b, &[2.0, 0.5, 0.5, 1.0]).unwrap();
    assert_eq!(
        free(&v, &b, &[2.0, 0.5, 0.4, 1.0]).unwrap_err().kind,
        ErrorKind::Constraint
    );
}

/// stan #2628: generated quantities can be requested without transformed parameters; the
/// transformed parameters are still evaluated (the generated quantities may use them) but not
/// written, and the draws match those of the full output.
#[test]
fn generated_quantities_without_transformed_parameters() {
    let m = regression();
    let theta = [0.2, 0.9, -0.4];
    let gq_only = Outputs {
        transformed_parameters: false,
        generated_quantities: true,
    };
    let names = m.param_names(gq_only);
    assert_eq!(
        names,
        [
            "alpha", "beta", "sigma", "y_rep.1", "y_rep.2", "y_rep.3", "y_rep.4"
        ]
    );
    let x = m
        .write_array(&theta, gq_only, &mut rng::stream(9, 0))
        .unwrap();
    let all = m
        .write_array(&theta, Outputs::ALL, &mut rng::stream(9, 0))
        .unwrap();
    assert_eq!(x.len(), names.len());
    assert_eq!(&x[..3], &all[..3]);
    assert_eq!(&x[3..], &all[4..]);
}

/// bridgestan #289, #336, stan #538: one model shared by threads, each drawing generated
/// quantities with its own generator, gives the serial results.
#[test]
fn write_array_from_many_threads() {
    let m = std::sync::Arc::new(regression());
    let serial: Vec<Vec<f64>> = (0..8)
        .map(|s| {
            m.write_array(&[0.1, 1.0, -0.5], Outputs::ALL, &mut rng::stream(4, s))
                .unwrap()
        })
        .collect();
    let handles: Vec<_> = (0..8)
        .map(|s| {
            let m = m.clone();
            std::thread::spawn(move || {
                (0..20)
                    .map(|_| {
                        m.write_array(&[0.1, 1.0, -0.5], Outputs::ALL, &mut rng::stream(4, s))
                            .unwrap()
                    })
                    .collect::<Vec<_>>()
            })
        })
        .collect();
    for (s, h) in handles.into_iter().enumerate() {
        for x in h.join().unwrap() {
            assert_eq!(x, serial[s]);
        }
    }
}

/// stan #1069, #2237, #3459, bridgestan #319: partial initialization. Given parameters keep
/// their values, missing ones (including arrays of tuples with constrained components) are
/// drawn, other names are ignored, and invalid given values are rejected.
#[test]
fn partial_initialization() {
    let m = regression();
    let values = data(r#"{"sigma": 0.5, "not_a_parameter": [1, 2]}"#);
    let a = m
        .param_initialize(&values, 2.0, &mut rng::stream(1, 0))
        .unwrap();
    let b = m
        .param_initialize(&values, 2.0, &mut rng::stream(1, 0))
        .unwrap();
    assert_eq!(a, b);
    assert!(close(a[2], 0.5f64.ln(), 1e-15));
    assert!(a[..2].iter().all(|v| v.abs() < 2.0));
    let zero = m
        .param_initialize(&values, 0.0, &mut rng::stream(1, 0))
        .unwrap();
    assert_eq!(&zero[..2], [0.0, 0.0]);
    let all = data(r#"{"alpha": 1.0, "beta": -2.0, "sigma": 0.5}"#);
    assert_eq!(
        m.param_initialize(&all, 2.0, &mut rng::stream(1, 0))
            .unwrap(),
        m.param_unconstrain_json(&all).unwrap()
    );
    let mut g = rng::stream(1, 0);
    assert_eq!(
        m.param_initialize(&data(r#"{"sigma": -1}"#), 2.0, &mut g)
            .unwrap_err()
            .kind,
        ErrorKind::Constraint
    );
    assert_eq!(
        m.param_initialize(&data(r#"{"sigma": [1, 2]}"#), 2.0, &mut g)
            .unwrap_err()
            .kind,
        ErrorKind::Parameters
    );
    assert_eq!(
        m.param_initialize(&values, f64::NAN, &mut g)
            .unwrap_err()
            .kind,
        ErrorKind::Parameters
    );

    let slots = vec![
        param(
            "",
            vec![],
            BaseType::Real,
            Constraint::Lower(Bound::Value(0.0)),
        ),
        param("", vec![], BaseType::Vector(3), Constraint::Simplex),
    ];
    let t = Params::new(vec![
        param("m", vec![], BaseType::Real, Constraint::None),
        VarInfo::tuple("q", Block::Parameters, vec![2], slots),
    ]);
    let init = t
        .param_initialize(&data(r#"{"m": 1.5}"#), 2.0, &mut g)
        .unwrap();
    assert_eq!(init.len(), t.num_unconstrained());
    assert_eq!(init[0], 1.5);
    let q = r#"{"q": [{"1": 2.0, "2": [0.2, 0.3, 0.5]}, {"1": 0.1, "2": [0.6, 0.2, 0.2]}]}"#;
    let init = t.param_initialize(&data(q), 2.0, &mut g).unwrap();
    let c = t.param_constrain(&init).unwrap();
    let expected = [2.0, 0.2, 0.3, 0.5, 0.1, 0.6, 0.2, 0.2];
    assert!(
        c[1..]
            .iter()
            .zip(expected)
            .all(|(a, b)| close(*a, b, 1e-12)),
        "{c:?}"
    );
}
