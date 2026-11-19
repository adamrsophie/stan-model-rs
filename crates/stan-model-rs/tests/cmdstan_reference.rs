//! Log densities and gradients compared with CmdStan 2.40's `log_prob` method, which evaluates
//! the official Stan C++ model at the same unconstrained points. The reference values were
//! generated once from the programs in `tests/stan/` and are stored in
//! `tests/stan/cmdstan_log_prob.json`; this check catches any difference in the unconstrained
//! parameter order, the transforms, or the Jacobians.

use serde_json::Value;
use stan_model_rs::{Data, LogDensityOptions, Model};
use stan_model_rs_fixtures::{LinearRegression, Structured};

const REFERENCE: &str = include_str!("../../../tests/stan/cmdstan_log_prob.json");

fn check(model: &dyn Model, cases: &Value) {
    for case in cases.as_array().unwrap() {
        let theta: Vec<f64> = serde_json::from_value(case["theta"].clone()).unwrap();
        let jacobian = case["jacobian"].as_bool().unwrap();
        let lp = case["lp"].as_f64().unwrap();
        let gradient: Vec<f64> = serde_json::from_value(case["gradient"].clone()).unwrap();
        // CmdStan's log_prob drops constants (propto = true).
        let opts = LogDensityOptions {
            jacobian,
            propto: true,
        };
        let (v, g) = model.log_density_gradient(&theta, opts).unwrap();
        assert!(
            (v - lp).abs() <= 1e-12 * (1.0 + lp.abs()),
            "{}: {v} vs {lp}",
            model.name()
        );
        for (a, b) in g.iter().zip(&gradient) {
            assert!(
                (a - b).abs() <= 1e-12 * (1.0 + b.abs()),
                "{}: {g:?} vs {gradient:?}",
                model.name()
            );
        }
        assert_eq!(model.log_density(&theta, opts).unwrap(), v);
    }
}

#[test]
fn matches_cmdstan_log_prob() {
    let reference: Value = serde_json::from_str(REFERENCE).unwrap();
    let models = &reference["models"];
    check(&Structured::new(), &models["structured"]["cases"]);
    let data = Data::from_json_value(models["linear_regression"]["data"].clone()).unwrap();
    check(
        &LinearRegression::new(&data).unwrap(),
        &models["linear_regression"]["cases"],
    );
}
