//! The runtime invariants: stable dimensions, metadata matching the flattened values, Jacobians
//! counted once, gradients matching their values, generated quantities isolated from the log
//! density, reproducible random numbers, and independent evaluations across threads.

use std::sync::Arc;

use proptest::prelude::*;
use stan_model_rs::rng;
use stan_model_rs::{Data, ErrorKind, LogDensityOptions, Model, Outputs};
use stan_model_rs_fixtures::{
    DependentBound, Empty, LinearRegression, NormalMean, NormalScale, Structured,
};

const LN_SQRT_2PI: f64 = 0.918_938_533_204_672_7;

fn regression() -> LinearRegression {
    LinearRegression::new(
        &Data::from_json_str(r#"{"N": 4, "x": [0.0, 1.0, 2.0, 3.5], "y": [0.3, 1.1, 2.4, 3.9]}"#)
            .unwrap(),
    )
    .unwrap()
}

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * (1.0 + b.abs())
}

/// log |det J| of `f` at `y` by central differences, for a square Jacobian.
fn numeric_log_det(f: impl Fn(&[f64]) -> Vec<f64>, y: &[f64]) -> f64 {
    let n = y.len();
    let h = 1e-6;
    let mut j = vec![vec![0.0; n]; n];
    for c in 0..n {
        let mut up = y.to_vec();
        let mut down = y.to_vec();
        up[c] += h;
        down[c] -= h;
        let (fu, fd) = (f(&up), f(&down));
        for r in 0..n {
            j[r][c] = (fu[r] - fd[r]) / (2.0 * h);
        }
    }
    // Gaussian elimination with partial pivoting.
    let mut log_det = 0.0;
    for k in 0..n {
        let p = (k..n)
            .max_by(|&a, &b| j[a][k].abs().total_cmp(&j[b][k].abs()))
            .unwrap();
        j.swap(k, p);
        let pivot = j[k][k];
        log_det += pivot.abs().ln();
        for r in k + 1..n {
            let factor = j[r][k] / pivot;
            for c in k..n {
                j[r][c] -= factor * j[k][c];
            }
        }
    }
    log_det
}

#[test]
fn structured_metadata_and_ordering() {
    let m = Structured::new();
    // simplex[3]: 2, array[2] vector[2]: 4, ordered[2]: 2.
    assert_eq!(m.num_unconstrained(), 8);
    assert_eq!(m.num_outputs(Outputs::PARAMETERS), 9);
    assert_eq!(
        m.param_names(Outputs::PARAMETERS),
        [
            "theta.1", "theta.2", "theta.3", "tau.1.1", "tau.2.1", "tau.1.2", "tau.2.2", "c.1",
            "c.2"
        ]
    );
    assert_eq!(
        m.unconstrained_param_names(),
        [
            "theta.1", "theta.2", "tau.1.1", "tau.2.1", "tau.1.2", "tau.2.2", "c.1", "c.2"
        ]
    );
    // The unconstrained vector holds tau in storage order (tau[1] first); the output is
    // column-major (tau[1][1], tau[2][1], ...).
    let theta = [0.0, 0.0, 0.1, 0.2, 0.3, 0.4, -1.0, 0.5];
    let c = m.param_constrain(&theta).unwrap();
    let e = f64::exp;
    let expected_tau = [e(0.1), e(0.3), e(0.2), e(0.4)];
    for (a, b) in c[3..7].iter().zip(expected_tau) {
        assert!(close(*a, b, 1e-15));
    }
    assert!(close(c[7], -1.0, 1e-15) && close(c[8], -1.0 + e(0.5), 1e-15));
    let back = m.param_unconstrain(&c).unwrap();
    for (a, b) in back.iter().zip(theta) {
        assert!(close(*a, b, 1e-12), "{back:?}");
    }
}

#[test]
fn jacobian_is_log_det_of_the_transform() {
    let m = Structured::new();
    let theta = [0.3, -0.2, 0.1, 0.2, 0.3, 0.4, -1.0, 0.5];
    let with = m.log_density(&theta, LogDensityOptions::default()).unwrap();
    let without = m
        .log_density(
            &theta,
            LogDensityOptions {
                jacobian: false,
                propto: false,
            },
        )
        .unwrap();
    // The simplex Jacobian is taken with respect to its first K - 1 coordinates, as in Stan.
    let f = |y: &[f64]| {
        let x = m.param_constrain(y).unwrap();
        let mut v = x[..2].to_vec();
        v.extend_from_slice(&x[3..]);
        v
    };
    let log_det = numeric_log_det(f, &theta);
    assert!(
        close(with - without, log_det, 1e-6),
        "{} vs {log_det}",
        with - without
    );
}

#[test]
fn gradients_match_finite_differences() {
    let models: Vec<(Box<dyn Model>, Vec<f64>)> = vec![
        (
            Box::new(Structured::new()),
            vec![0.3, -0.2, 0.1, 0.2, 0.3, 0.4, -1.0, 0.5],
        ),
        (Box::new(regression()), vec![0.2, 0.9, -0.4]),
        (
            Box::new(NormalScale::new(&Data::from_json_str(r#"{"y": 1.3}"#).unwrap()).unwrap()),
            vec![0.25],
        ),
    ];
    for (m, theta) in &models {
        for opts in [
            LogDensityOptions::default(),
            LogDensityOptions::SAMPLING,
            LogDensityOptions {
                jacobian: false,
                propto: false,
            },
        ] {
            let c = m.check_gradient(theta, opts, 1e-6).unwrap();
            assert!(c.max_rel_error < 1e-6, "{}: {c:?}", m.name());
            assert_eq!(c.value, m.log_density(theta, opts).unwrap());
        }
    }
}

#[test]
fn hessian_and_hessian_vector_product_agree() {
    let m = regression();
    let theta = [0.2, 0.9, -0.4];
    let opts = LogDensityOptions::default();
    let (value, grad, h) = m.log_density_hessian(&theta, opts).unwrap();
    let (v2, g2) = m.log_density_gradient(&theta, opts).unwrap();
    assert!(close(value, v2, 1e-14));
    for i in 0..3 {
        assert!(close(grad[i], g2[i], 1e-12));
        // Symmetric, and each column matches a finite difference of the gradient.
        let mut up = theta;
        let mut down = theta;
        up[i] += 1e-6;
        down[i] -= 1e-6;
        let gu = m.log_density_gradient(&up, opts).unwrap().1;
        let gd = m.log_density_gradient(&down, opts).unwrap().1;
        for r in 0..3 {
            assert!(close(h[(r, i)], h[(i, r)], 1e-12));
            assert!(close(h[(r, i)], (gu[r] - gd[r]) / 2e-6, 1e-5));
        }
    }
    let v = [0.5, -1.0, 2.0];
    let hv = m
        .log_density_hessian_vector_product(&theta, &v, opts)
        .unwrap();
    for r in 0..3 {
        let expected: f64 = (0..3).map(|c| h[(r, c)] * v[c]).sum();
        assert!(close(hv[r], expected, 1e-12));
    }
    assert_eq!(
        m.log_density_hessian_vector_product(&theta, &v[..2], opts)
            .unwrap_err()
            .kind,
        ErrorKind::Parameters
    );
}

#[test]
fn regression_matches_closed_form() {
    let m = regression();
    let (alpha, beta, z) = (0.2, 0.9, -0.4);
    let sigma = f64::exp(z);
    let x = [0.0, 1.0, 2.0, 3.5];
    let y = [0.3, 1.1, 2.4, 3.9];
    let norm = |v: f64, mu: f64, s: f64| -0.5 * ((v - mu) / s).powi(2) - s.ln() - LN_SQRT_2PI;
    let mut lp = norm(alpha, 0.0, 10.0) + norm(beta, 0.0, 10.0) - sigma + z;
    for i in 0..4 {
        lp += norm(y[i], alpha + beta * x[i], sigma);
    }
    let got = m
        .log_density(&[alpha, beta, z], LogDensityOptions::default())
        .unwrap();
    assert!(close(got, lp, 1e-13), "{got} vs {lp}");
}

#[test]
fn generated_quantities_are_reproducible_and_isolated() {
    let m = regression();
    let theta = [0.2, 0.9, -0.4];
    assert_eq!(
        m.param_names(Outputs::ALL),
        [
            "alpha",
            "beta",
            "sigma",
            "log_sigma",
            "y_rep.1",
            "y_rep.2",
            "y_rep.3",
            "y_rep.4"
        ]
    );
    let lp_before = m
        .log_density_gradient(&theta, LogDensityOptions::SAMPLING)
        .unwrap();
    let a = m
        .write_array(&theta, Outputs::ALL, &mut rng::stream(7, 1))
        .unwrap();
    let b = m
        .write_array(&theta, Outputs::ALL, &mut rng::stream(7, 1))
        .unwrap();
    let c = m
        .write_array(&theta, Outputs::ALL, &mut rng::stream(7, 2))
        .unwrap();
    assert_eq!(a, b);
    assert_eq!(a[..4], c[..4]);
    assert_ne!(a[4..], c[4..]);
    assert!(close(a[3], -0.4, 1e-15));
    // One generator, two calls: the second call continues the stream instead of reseeding.
    let mut r = rng::stream(7, 1);
    let first = m.write_array(&theta, Outputs::ALL, &mut r).unwrap();
    let second = m.write_array(&theta, Outputs::ALL, &mut r).unwrap();
    assert_eq!(first, a);
    assert_ne!(first[4..], second[4..]);
    // Generated quantities leave the log density unchanged.
    assert_eq!(
        m.log_density_gradient(&theta, LogDensityOptions::SAMPLING)
            .unwrap(),
        lp_before
    );
    // Parameters only, and the default implementation rejects unimplemented derived outputs.
    assert_eq!(m.param_constrain(&theta).unwrap().len(), 3);
    let s = Structured::new();
    let e = s
        .write_array(&[0.0; 8], Outputs::ALL, &mut rng::stream(1, 1))
        .unwrap();
    assert_eq!(e.len(), 9);
}

#[test]
fn evaluations_are_independent_across_threads() {
    let m = Arc::new(regression());
    let points: Vec<[f64; 3]> = (0..16)
        .map(|i| [0.1 * i as f64, -0.5, 0.05 * i as f64])
        .collect();
    let serial: Vec<_> = points
        .iter()
        .map(|p| {
            m.log_density_gradient(p, LogDensityOptions::SAMPLING)
                .unwrap()
        })
        .collect();
    let handles: Vec<_> = (0..8)
        .map(|t| {
            let m = Arc::clone(&m);
            let points = points.clone();
            std::thread::spawn(move || {
                (0..50)
                    .map(|k| {
                        let i = (t + k) % points.len();
                        (
                            i,
                            m.log_density_gradient(&points[i], LogDensityOptions::SAMPLING)
                                .unwrap(),
                        )
                    })
                    .collect::<Vec<_>>()
            })
        })
        .collect();
    for h in handles {
        for (i, r) in h.join().unwrap() {
            assert_eq!(r, serial[i]);
        }
    }
    // Independent instances built from the same data agree exactly.
    let other = regression();
    assert_eq!(
        other
            .log_density(&points[3], LogDensityOptions::default())
            .unwrap(),
        m.log_density(&points[3], LogDensityOptions::default())
            .unwrap()
    );
}

#[test]
fn zero_parameter_model() {
    let m = Empty::new();
    assert_eq!(m.num_unconstrained(), 0);
    assert!(m.param_names(Outputs::ALL).is_empty());
    assert_eq!(
        m.log_density(&[], LogDensityOptions::default()).unwrap(),
        0.0
    );
    let (v, g) = m
        .log_density_gradient(&[], LogDensityOptions::SAMPLING)
        .unwrap();
    assert_eq!((v, g.len()), (0.0, 0));
    assert!(m.param_constrain(&[]).unwrap().is_empty());
    assert!(m.param_unconstrain(&[]).unwrap().is_empty());
    assert_eq!(
        m.log_density(&[1.0], LogDensityOptions::default())
            .unwrap_err()
            .kind,
        ErrorKind::Parameters
    );
}

#[test]
fn invalid_data_is_rejected_before_inference() {
    let cases = [
        r#"{"N": 0, "x": [], "y": []}"#,
        r#"{"N": 2, "x": [1.0], "y": [1.0, 2.0]}"#,
        r#"{"N": 2.0, "x": [1.0, 2.0], "y": [1.0, 2.0]}"#,
        r#"{"N": 2, "x": [1.0, 2.0]}"#,
        r#"{"N": 2, "x": [1.0, "a"], "y": [1.0, 2.0]}"#,
    ];
    for json in cases {
        let e = LinearRegression::new(&Data::from_json_str(json).unwrap()).unwrap_err();
        assert_eq!(e.kind, ErrorKind::Data, "{json}");
        assert!(e.variable.is_some(), "{json}");
    }
}

#[test]
fn metadata_serializes() {
    let m = regression();
    let json = m.info().to_json();
    let back = stan_model_rs::ModelInfo::from_json(&json).unwrap();
    assert_eq!(&back, m.info());
    assert_eq!(back.num_unconstrained(), 3);
    assert!(json.contains("\"runtime_version\""));
}

#[test]
fn propto_drops_only_constants() {
    // propto and full densities differ by a constant, so their gradients are identical.
    let m = Structured::new();
    let a = [0.3, -0.2, 0.1, 0.2, 0.3, 0.4, -1.0, 0.5];
    let b = [-0.1, 0.4, 0.0, -0.3, 0.2, 0.1, 0.5, -0.5];
    let full = |t: &[f64]| m.log_density(t, LogDensityOptions::default()).unwrap();
    let propto = |t: &[f64]| m.log_density(t, LogDensityOptions::SAMPLING).unwrap();
    assert!(close(full(&a) - propto(&a), full(&b) - propto(&b), 1e-12));
    let ga = m
        .log_density_gradient(&a, LogDensityOptions::default())
        .unwrap()
        .1;
    let gp = m
        .log_density_gradient(&a, LogDensityOptions::SAMPLING)
        .unwrap()
        .1;
    assert_eq!(ga, gp);
    let _ = NormalMean::new(&Data::from_json_str(r#"{"y": 1}"#).unwrap()).unwrap();
}

#[test]
fn parameter_dependent_bound() {
    let m = DependentBound::new();
    let theta = [0.4, -0.3];
    let c = m.param_constrain(&theta).unwrap();
    assert!(close(c[1], 0.4 + f64::exp(-0.3), 1e-15));
    let back = m.param_unconstrain(&c).unwrap();
    assert!(close(back[0], 0.4, 1e-15) && close(back[1], -0.3, 1e-12));
    // b below a is outside the support.
    let e = m.param_unconstrain(&[1.0, 0.5]).unwrap_err();
    assert_eq!(e.variable.as_deref(), Some("b"));
    let with = m.log_density(&theta, LogDensityOptions::default()).unwrap();
    let without = m
        .log_density(
            &theta,
            LogDensityOptions {
                jacobian: false,
                propto: false,
            },
        )
        .unwrap();
    let log_det = numeric_log_det(|y| m.param_constrain(y).unwrap(), &theta);
    assert!(close(with - without, log_det, 1e-6));
    let c = m
        .check_gradient(&theta, LogDensityOptions::SAMPLING, 1e-6)
        .unwrap();
    assert!(c.max_rel_error < 1e-6, "{c:?}");
    // The metadata alone cannot evaluate the bound.
    let b = m.info().variable("b").unwrap();
    assert_eq!(
        stan_model_rs::Bounds::from_metadata(b).unwrap_err().kind,
        ErrorKind::Unsupported
    );
}

proptest! {
    #[test]
    fn constrain_unconstrain_round_trip(theta in prop::collection::vec(-4.0f64..4.0, 8)) {
        let m = Structured::new();
        let c = m.param_constrain(&theta).unwrap();
        let back = m.param_unconstrain(&c).unwrap();
        for (a, b) in back.iter().zip(&theta) {
            prop_assert!((a - b).abs() < 1e-8 * (1.0 + b.abs()), "{back:?} vs {theta:?}");
        }
        // The simplex sums to one and the ordered vector is increasing.
        prop_assert!((c[0] + c[1] + c[2] - 1.0).abs() < 1e-12);
        prop_assert!(c[8] > c[7]);
    }

    #[test]
    fn log_density_is_deterministic(mu in -50.0f64..50.0, y in -50.0f64..50.0) {
        let m = NormalMean::new(&Data::from_json_str(&format!(r#"{{"y": {y}}}"#)).unwrap()).unwrap();
        let a = m.log_density_gradient(&[mu], LogDensityOptions::SAMPLING).unwrap();
        let b = m.log_density_gradient(&[mu], LogDensityOptions::SAMPLING).unwrap();
        prop_assert_eq!(a, b);
    }
}
