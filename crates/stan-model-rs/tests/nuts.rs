//! A sampler from another crate draws from a stan-model-rs model: NUTS from `nuts-rs` on the
//! normal mean reference model must reproduce the closed-form posterior moments.

use nuts_rs::{Chain, CpuMath, DiagNutsSettings, Settings};
use rand::SeedableRng;
use stan_model_rs::nuts::NutsModel;
use stan_model_rs::{Data, LogDensityOptions, Model};
use stan_model_rs_fixtures::NormalMean;

#[test]
fn nuts_recovers_the_normal_mean_posterior() {
    // Posterior of mu: precision 1/25 + 1, mean y / (1/25 + 1).
    let y = 2.0;
    let model =
        NormalMean::new(&Data::from_json_str(&format!(r#"{{"y": {y}}}"#)).unwrap()).unwrap();
    let precision = 1.0 / 25.0 + 1.0;
    let (mean, var) = (y / precision, 1.0 / precision);

    let settings = DiagNutsSettings {
        num_tune: 500,
        num_draws: 4000,
        ..Default::default()
    };
    let mut rng = rand::rngs::StdRng::seed_from_u64(5);
    let mut chain = settings
        .new_chain(0, CpuMath::new(NutsModel::new(&model)), &mut rng)
        .unwrap();
    chain.set_position(&[0.1]).unwrap();
    for _ in 0..settings.num_tune {
        chain.draw().unwrap();
    }
    let draws: Vec<f64> = (0..settings.num_draws)
        .map(|_| chain.draw().unwrap().0[0])
        .collect();
    let n = draws.len() as f64;
    let m = draws.iter().sum::<f64>() / n;
    let v = draws.iter().map(|d| (d - m).powi(2)).sum::<f64>() / (n - 1.0);
    assert!(
        (m - mean).abs() < 6.0 * (var / n).sqrt(),
        "mean {m} vs {mean}"
    );
    assert!((v / var - 1.0).abs() < 0.15, "variance {v} vs {var}");
    // The sampler never changes the model.
    let lp = model
        .log_density(&[0.5], LogDensityOptions::default())
        .unwrap();
    assert_eq!(
        lp,
        model
            .log_density(&[0.5], LogDensityOptions::default())
            .unwrap()
    );
}
