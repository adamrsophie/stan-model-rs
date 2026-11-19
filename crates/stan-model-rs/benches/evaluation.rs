//! Cost of the evaluations a sampler repeats: log density, gradient, Hessian-vector product,
//! and parameter constraining, on the regression reference model with growing data.

use criterion::{Criterion, criterion_group, criterion_main};
use stan_model_rs::{Data, LogDensityOptions, Model};
use stan_model_rs_fixtures::LinearRegression;

fn regression(n: usize) -> LinearRegression {
    let x: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|x| 1.0 + 2.0 * x + 0.1 * (x * 37.0).sin())
        .collect();
    let json = serde_json::json!({"N": n, "x": x, "y": y});
    LinearRegression::new(&Data::from_json_value(json).unwrap()).unwrap()
}

fn bench(c: &mut Criterion) {
    let theta = [0.9, 2.1, -1.5];
    for n in [10, 1000] {
        let m = regression(n);
        let opts = LogDensityOptions::SAMPLING;
        let full = LogDensityOptions::default();
        c.bench_function(&format!("regression N={n} log_density"), |b| {
            b.iter(|| m.log_density(&theta, full).unwrap())
        });
        c.bench_function(&format!("regression N={n} log_density propto"), |b| {
            b.iter(|| m.log_density(&theta, opts).unwrap())
        });
        c.bench_function(&format!("regression N={n} gradient"), |b| {
            b.iter(|| m.log_density_gradient(&theta, opts).unwrap())
        });
        c.bench_function(&format!("regression N={n} hessian_vector_product"), |b| {
            b.iter(|| {
                m.log_density_hessian_vector_product(&theta, &[1.0, 0.0, 0.0], opts)
                    .unwrap()
            })
        });
    }
    let m = regression(10);
    c.bench_function("regression param_constrain", |b| {
        b.iter(|| m.param_constrain(&theta).unwrap())
    });
}

criterion_group!(benches, bench);
criterion_main!(benches);
