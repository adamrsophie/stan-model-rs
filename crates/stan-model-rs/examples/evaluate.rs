//! Builds the regression reference model from JSON data and shows what the runtime provides:
//! metadata, a log density and its gradient, constrained values, generated quantities, and
//! unconstrained initial values.
//!
//! Run with `cargo run -p stan-model-rs --example evaluate`.

use stan_model_rs::{Data, LogDensityOptions, Model, Outputs, rng};
use stan_model_rs_fixtures::LinearRegression;

fn main() -> Result<(), stan_model_rs::ModelError> {
    let data =
        Data::from_json_str(r#"{"N": 4, "x": [0.0, 1.0, 2.0, 3.5], "y": [0.3, 1.1, 2.4, 3.9]}"#)?;
    let model = LinearRegression::new(&data)?;

    println!("model: {}", model.name());
    println!(
        "unconstrained parameters: {:?}",
        model.unconstrained_param_names()
    );
    println!("outputs: {:?}", model.param_names(Outputs::ALL));

    // Initial values on the constrained scale, by name.
    let init = Data::from_json_str(r#"{"alpha": 0.0, "beta": 1.0, "sigma": 0.5}"#)?;
    let theta = model.param_unconstrain_json(&init)?;
    println!("unconstrained initial values: {theta:?}");

    for (label, opts) in [
        ("full", LogDensityOptions::default()),
        ("sampling (propto)", LogDensityOptions::SAMPLING),
        (
            "no Jacobian",
            LogDensityOptions {
                jacobian: false,
                propto: false,
            },
        ),
    ] {
        let (lp, grad) = model.log_density_gradient(&theta, opts)?;
        println!("{label:>18}: log density {lp:.6}, gradient {grad:.6?}");
    }

    let check = model.check_gradient(&theta, LogDensityOptions::default(), 1e-6)?;
    println!(
        "gradient check: max relative error {:.2e}",
        check.max_rel_error
    );

    let draw = model.write_array(&theta, Outputs::ALL, &mut rng::stream(2026, 1))?;
    for (name, value) in model.param_names(Outputs::ALL).iter().zip(&draw) {
        println!("{name:>10} = {value:.4}");
    }
    Ok(())
}
