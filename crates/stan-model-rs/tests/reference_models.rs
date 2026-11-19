//! The two reference models of the first milestones, checked against closed-form log densities
//! and gradients written out independently of stanmath-rs.

use stan_model_rs::{Data, ErrorKind, LogDensityOptions, Model, Outputs};
use stan_model_rs_fixtures::{NormalMean, NormalScale};

const LN_SQRT_2PI: f64 = 0.918_938_533_204_672_7;

fn data(json: &str) -> Data {
    Data::from_json_str(json).unwrap()
}

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * (1.0 + b.abs())
}

fn opts(jacobian: bool, propto: bool) -> LogDensityOptions {
    LogDensityOptions { jacobian, propto }
}

/// log N(x | m, s), written out.
fn normal_lpdf(x: f64, m: f64, s: f64) -> f64 {
    let z = (x - m) / s;
    -0.5 * z * z - s.ln() - LN_SQRT_2PI
}

#[test]
fn normal_mean_metadata() {
    let m = NormalMean::new(&data(r#"{"y": 0.7}"#)).unwrap();
    assert_eq!(m.name(), "normal_mean_model");
    assert_eq!(m.num_unconstrained(), 1);
    assert_eq!(m.param_names(Outputs::ALL), ["mu"]);
    assert_eq!(m.unconstrained_param_names(), ["mu"]);
    // The identity transform.
    assert_eq!(m.param_constrain(&[1.25]).unwrap(), [1.25]);
    assert_eq!(m.param_unconstrain(&[1.25]).unwrap(), [1.25]);
}

#[test]
fn normal_mean_density_and_gradient() {
    for y in [-3.0, 0.0, 0.7, 12.5] {
        let m = NormalMean::new(&data(&format!(r#"{{"y": {y}}}"#))).unwrap();
        for mu in [-10.0, -1.0, 0.0, 0.3, 2.0, 40.0] {
            let full = normal_lpdf(mu, 0.0, 5.0) + normal_lpdf(y, mu, 1.0);
            // propto drops -2 ln sqrt(2 pi) and -ln 5, which do not depend on mu.
            let propto = -mu * mu / 50.0 - 0.5 * (y - mu) * (y - mu);
            let grad = -mu / 25.0 + (y - mu);
            for jacobian in [true, false] {
                let lp = m.log_density(&[mu], opts(jacobian, false)).unwrap();
                assert!(close(lp, full, 1e-14), "y {y} mu {mu}: {lp} vs {full}");
                let lp = m.log_density(&[mu], opts(jacobian, true)).unwrap();
                assert!(
                    close(lp, propto, 1e-14),
                    "propto y {y} mu {mu}: {lp} vs {propto}"
                );
                for propto_flag in [false, true] {
                    let (v, g) = m
                        .log_density_gradient(&[mu], opts(jacobian, propto_flag))
                        .unwrap();
                    assert!(close(v, if propto_flag { propto } else { full }, 1e-14));
                    assert!(close(g[0], grad, 1e-13), "gradient {} vs {grad}", g[0]);
                }
            }
            let (_, _, h) = m
                .log_density_hessian(&[mu], LogDensityOptions::default())
                .unwrap();
            assert!(close(h[(0, 0)], -1.0 / 25.0 - 1.0, 1e-13));
        }
    }
}

#[test]
fn normal_mean_repeated_evaluation_is_identical() {
    let m = NormalMean::new(&data(r#"{"y": 0.7}"#)).unwrap();
    let first = m
        .log_density_gradient(&[0.4], LogDensityOptions::SAMPLING)
        .unwrap();
    for _ in 0..100 {
        // Interleave other points: a stale tape or accumulated adjoints would show up here.
        m.log_density_gradient(&[-3.0], LogDensityOptions::default())
            .unwrap();
        assert_eq!(
            m.log_density_gradient(&[0.4], LogDensityOptions::SAMPLING)
                .unwrap(),
            first
        );
    }
}

#[test]
fn normal_mean_errors() {
    let m = NormalMean::new(&data(r#"{"y": 0.7}"#)).unwrap();
    for bad in [&[][..], &[1.0, 2.0][..]] {
        let e = m
            .log_density(bad, LogDensityOptions::default())
            .unwrap_err();
        assert_eq!(e.kind, ErrorKind::Parameters);
        assert_eq!(
            m.log_density_gradient(bad, LogDensityOptions::default())
                .unwrap_err()
                .kind,
            ErrorKind::Parameters
        );
        assert_eq!(
            m.param_constrain(bad).unwrap_err().kind,
            ErrorKind::Parameters
        );
        assert_eq!(
            m.param_unconstrain(bad).unwrap_err().kind,
            ErrorKind::Parameters
        );
    }
    // A non-finite location is a domain error that rejects the proposal, never a panic.
    let e = m
        .log_density(&[f64::NAN], LogDensityOptions::default())
        .unwrap_err();
    assert_eq!(e.kind, ErrorKind::Domain);
    assert!(e.kind.is_recoverable());
    assert!(e.function.is_some());
    let mut g = [0.0; 2];
    assert_eq!(
        m.log_density_gradient_into(&[0.1], LogDensityOptions::default(), &mut g)
            .unwrap_err()
            .kind,
        ErrorKind::Parameters
    );

    // Data errors are raised at construction, before any evaluation.
    for bad in [
        r#"{}"#,
        r#"{"y": "x"}"#,
        r#"{"y": [1.0]}"#,
        r#"{"y": null}"#,
    ] {
        let e = NormalMean::new(&data(bad)).unwrap_err();
        assert_eq!(e.kind, ErrorKind::Data, "{bad}");
        assert_eq!(e.variable.as_deref(), Some("y"));
    }
}

#[test]
fn normal_scale_density_jacobian_and_gradient() {
    for y in [0.0, 0.4, 2.0, 15.0] {
        let m = NormalScale::new(&data(&format!(r#"{{"y": {y}}}"#))).unwrap();
        for z in [-20.0, -3.0, -0.5, 0.0, 0.8, 4.0, 20.0] {
            let sigma = f64::exp(z);
            let without_jacobian = -sigma + normal_lpdf(y, 0.0, sigma);
            let full = without_jacobian + z;
            // propto keeps -ln(sigma), which depends on the parameter, and drops -ln sqrt(2 pi)
            // and ln(beta) = 0.
            let propto = full + LN_SQRT_2PI;
            // d/dz [-e^z - z - y^2 e^{-2z} / 2 + z] = -e^z + y^2 e^{-2z}
            let grad = -sigma + y * y * f64::exp(-2.0 * z);
            let grad_no_jac = grad - 1.0;

            let lp = m.log_density(&[z], opts(true, false)).unwrap();
            assert!(close(lp, full, 1e-12), "y {y} z {z}: {lp} vs {full}");
            let lp = m.log_density(&[z], opts(false, false)).unwrap();
            assert!(close(lp, without_jacobian, 1e-12));
            let lp = m.log_density(&[z], opts(true, true)).unwrap();
            assert!(close(lp, propto, 1e-12), "propto: {lp} vs {propto}");

            let (_, g) = m.log_density_gradient(&[z], opts(true, true)).unwrap();
            assert!(close(g[0], grad, 1e-12), "y {y} z {z}: {} vs {grad}", g[0]);
            let (_, g) = m.log_density_gradient(&[z], opts(false, false)).unwrap();
            assert!(close(g[0], grad_no_jac, 1e-12));

            // Second derivative: -e^z - 2 y^2 e^{-2z}.
            if z.abs() < 10.0 {
                let (_, _, h) = m
                    .log_density_hessian(&[z], LogDensityOptions::default())
                    .unwrap();
                let expected = -sigma - 2.0 * y * y * f64::exp(-2.0 * z);
                assert!(
                    close(h[(0, 0)], expected, 1e-10),
                    "{} vs {expected}",
                    h[(0, 0)]
                );
            }

            // The transform.
            let c = m.param_constrain(&[z]).unwrap();
            assert!(close(c[0], sigma, 1e-15));
            let back = m.param_unconstrain(&c).unwrap();
            assert!(close(back[0], z, 1e-12));
        }
    }
}

#[test]
fn normal_scale_extremes_and_invalid_initial_values() {
    let m = NormalScale::new(&data(r#"{"y": 0.4}"#)).unwrap();
    // e^800 overflows: the density at an infinite scale is not finite, and the runtime reports
    // it instead of substituting a value.
    match m.log_density(&[800.0], LogDensityOptions::default()) {
        Ok(v) => assert!(!v.is_finite()),
        Err(e) => assert!(e.kind.is_recoverable()),
    }
    // e^-800 underflows to zero: the normal density rejects a zero scale.
    match m.log_density(&[-800.0], LogDensityOptions::default()) {
        Ok(v) => assert!(!v.is_finite() || v < -1e300),
        Err(e) => assert!(e.kind.is_recoverable()),
    }

    // Constrained initial values outside the support are rejected.
    let e = m.param_unconstrain(&[-1.0]).unwrap_err();
    assert_eq!(e.kind, ErrorKind::Constraint);
    assert_eq!(e.variable.as_deref(), Some("sigma"));
    assert_eq!(
        m.param_unconstrain(&[f64::NAN]).unwrap_err().kind,
        ErrorKind::Constraint
    );
    // A value on the boundary maps to an infinite unconstrained value.
    assert_eq!(
        m.param_unconstrain(&[0.0]).unwrap_err().kind,
        ErrorKind::Parameters
    );
    // Initial values by name.
    let init = m
        .param_unconstrain_json(&data(r#"{"sigma": 2.0}"#))
        .unwrap();
    assert!(close(init[0], 2f64.ln(), 1e-15));
    assert_eq!(
        m.param_unconstrain_json(&data(r#"{}"#)).unwrap_err().kind,
        ErrorKind::Parameters
    );

    // Data must satisfy <lower=0>.
    let e = NormalScale::new(&data(r#"{"y": -0.1}"#)).unwrap_err();
    assert_eq!(e.kind, ErrorKind::Data);
    assert_eq!(e.variable.as_deref(), Some("y"));
}
