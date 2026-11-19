//! End to end: Stan source compiled by stanc-rs (in the build script), compiled by rustc,
//! instantiated from JSON data, and evaluated through the stan-model-rs interface. The results
//! are compared with the hand-written reference models, closed-form values, and CmdStan.

use serde_json::Value;
use stan_model_rs::{Data, LogDensityOptions, Model, Outputs, rng};
use stan_model_rs_fixtures::{LinearRegression, NormalMean, NormalScale, Structured};
use stan_model_rs_stanc::generated::*;
use stan_model_rs_stanc::{Stanc, convert_data};

const REFERENCE: &str = include_str!("../../../tests/stan/cmdstan_log_prob.json");

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * (1.0 + b.abs())
}

const ALL_OPTIONS: [LogDensityOptions; 4] = [
    LogDensityOptions {
        jacobian: true,
        propto: false,
    },
    LogDensityOptions {
        jacobian: true,
        propto: true,
    },
    LogDensityOptions {
        jacobian: false,
        propto: false,
    },
    LogDensityOptions {
        jacobian: false,
        propto: true,
    },
];

/// The generated model and the hand-written one agree on metadata, values, and gradients.
fn same_model(generated: &dyn Model, reference: &dyn Model, points: &[Vec<f64>]) {
    assert_eq!(generated.num_unconstrained(), reference.num_unconstrained());
    assert_eq!(
        generated.param_names(Outputs::ALL),
        reference.param_names(Outputs::ALL)
    );
    for theta in points {
        for opts in ALL_OPTIONS {
            let (a, ga) = generated.log_density_gradient(theta, opts).unwrap();
            let (b, gb) = reference.log_density_gradient(theta, opts).unwrap();
            assert!(
                close(a, b, 1e-12),
                "{}: {opts:?} {a} vs {b}",
                generated.name()
            );
            for (x, y) in ga.iter().zip(&gb) {
                assert!(
                    close(*x, *y, 1e-12),
                    "{}: {ga:?} vs {gb:?}",
                    generated.name()
                );
            }
            assert!(close(generated.log_density(theta, opts).unwrap(), b, 1e-12));
        }
        let c = generated.param_constrain(theta).unwrap();
        let r = reference.param_constrain(theta).unwrap();
        for (x, y) in c.iter().zip(&r) {
            assert!(close(*x, *y, 1e-12));
        }
        let back = generated.param_unconstrain(&c).unwrap();
        for (x, y) in back.iter().zip(theta) {
            assert!(close(*x, *y, 1e-9), "{back:?} vs {theta:?}");
        }
    }
}

fn data(json: &str) -> Data {
    Data::from_json_str(json).unwrap()
}

#[test]
fn normal_mean() {
    for y in [-3.0, 0.7, 12.5] {
        let d = data(&format!(r#"{{"y": {y}}}"#));
        let generated = Stanc::new(normal_mean::Model::new(&convert_data(&d).unwrap(), 1).unwrap());
        assert_eq!(generated.name(), "normal_mean_model");
        let points: Vec<Vec<f64>> = [-10.0, 0.0, 0.3, 40.0].iter().map(|m| vec![*m]).collect();
        same_model(&generated, &NormalMean::new(&d).unwrap(), &points);
        // And the closed-form gradient of section 21.
        for mu in [-1.0, 2.0] {
            let (_, g) = generated
                .log_density_gradient(&[mu], LogDensityOptions::default())
                .unwrap();
            assert!(close(g[0], -mu / 25.0 + (y - mu), 1e-13));
        }
    }
}

#[test]
fn normal_scale() {
    let d = data(r#"{"y": 0.4}"#);
    let generated = Stanc::new(normal_scale::Model::new(&convert_data(&d).unwrap(), 1).unwrap());
    let points: Vec<Vec<f64>> = [-3.0, -0.5, 0.0, 0.8, 4.0]
        .iter()
        .map(|z| vec![*z])
        .collect();
    same_model(&generated, &NormalScale::new(&d).unwrap(), &points);
    // Invalid data is rejected by the generated constructor too.
    let bad = convert_data(&data(r#"{"y": -1.0}"#)).unwrap();
    assert!(normal_scale::Model::new(&bad, 1).is_err());
}

#[test]
fn structured_and_regression_match_hand_written_and_cmdstan() {
    let reference: Value = serde_json::from_str(REFERENCE).unwrap();
    let structured =
        Stanc::new(structured::Model::new(&convert_data(&Data::empty()).unwrap(), 1).unwrap());
    let reg_data =
        Data::from_json_value(reference["models"]["linear_regression"]["data"].clone()).unwrap();
    let regression =
        Stanc::new(linear_regression::Model::new(&convert_data(&reg_data).unwrap(), 1).unwrap());
    let models: [(&dyn Model, &str); 2] = [
        (&structured, "structured"),
        (&regression, "linear_regression"),
    ];
    for (m, name) in models {
        for case in reference["models"][name]["cases"].as_array().unwrap() {
            let theta: Vec<f64> = serde_json::from_value(case["theta"].clone()).unwrap();
            let opts = LogDensityOptions {
                jacobian: case["jacobian"].as_bool().unwrap(),
                propto: true,
            };
            let (v, g) = m.log_density_gradient(&theta, opts).unwrap();
            let lp = case["lp"].as_f64().unwrap();
            assert!(close(v, lp, 1e-12), "{name}: {v} vs {lp}");
            let expected: Vec<f64> = serde_json::from_value(case["gradient"].clone()).unwrap();
            for (a, b) in g.iter().zip(&expected) {
                assert!(close(*a, *b, 1e-12), "{name}: {g:?} vs {expected:?}");
            }
        }
    }
    let points = vec![vec![0.3, -0.2, 0.1, 0.2, 0.3, 0.4, -1.0, 0.5]];
    same_model(&structured, &Structured::new(), &points);
    same_model(
        &regression,
        &LinearRegression::new(&reg_data).unwrap(),
        &[vec![0.2, 0.9, -0.4]],
    );

    // Generated quantities: same names and deterministic parts; draws reproducible per seed.
    let theta = [0.2, 0.9, -0.4];
    let a = regression
        .write_array(&theta, Outputs::ALL, &mut rng::stream(3, 1))
        .unwrap();
    let b = regression
        .write_array(&theta, Outputs::ALL, &mut rng::stream(3, 1))
        .unwrap();
    assert_eq!(a, b);
    assert_eq!(a.len(), 8);
    assert!(close(a[3], -0.4, 1e-15));
    // Initial values by name.
    let init = regression
        .param_unconstrain_json(&data(r#"{"alpha": 1.0, "beta": -2.0, "sigma": 0.5}"#))
        .unwrap();
    assert!(close(init[2], 0.5f64.ln(), 1e-15));
    assert_eq!(
        regression
            .log_density_hessian(&theta, LogDensityOptions::default())
            .unwrap_err()
            .kind,
        stan_model_rs::ErrorKind::Unsupported
    );
}

/// A generated model's `print` output reaches stan-model-rs's per-thread sink, lines printed
/// before a `reject` are kept, and the rejection carries its Stan source location.
#[test]
fn print_and_error_locations() {
    use stan_model_rs::{ErrorKind, print};
    let d = convert_data(&data(r#"{"y": 1.0}"#)).unwrap();
    let m =
        Stanc::new(stan_model_rs_stanc::forward_print(|| print_reject::Model::new(&d, 1)).unwrap());
    let opts = LogDensityOptions::default();

    let (lp, out) = print::with_sink(Vec::new(), || m.log_density(&[0.5], opts));
    assert!(lp.unwrap().is_finite());
    assert_eq!(String::from_utf8(out).unwrap(), "mu = 0.5\n");

    let (r, out) = print::with_sink(Vec::new(), || m.log_density_gradient(&[11.0], opts));
    assert_eq!(String::from_utf8(out).unwrap(), "mu = 11\n");
    let e = r.unwrap_err();
    assert_eq!(e.kind, ErrorKind::Reject);
    assert!(e.message.contains("mu too large: 11"), "{e}");
    assert_eq!(e.locations.len(), 1, "{e}");
    assert!(
        e.locations[0].starts_with("'print_reject.stan', line 10"),
        "{e}"
    );
    assert!(e.to_string().ends_with(&format!("(in {})", e.locations[0])));
}
