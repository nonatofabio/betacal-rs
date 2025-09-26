use betacal_inference::{BetaCalModel, CalibrationMethod};
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};

fn create_test_model() -> BetaCalModel {
    BetaCalModel::new(vec![1.2345, -0.6789], 0.0, CalibrationMethod::AB)
}

fn bench_single_prediction(c: &mut Criterion) {
    let model = create_test_model();
    
    c.bench_function("single_prediction", |b| {
        b.iter(|| {
            model.predict_single(black_box(0.7)).unwrap()
        })
    });
}

fn bench_batch_prediction(c: &mut Criterion) {
    let model = create_test_model();
    let mut group = c.benchmark_group("batch_prediction");
    
    for size in [10, 100, 1000, 10000].iter() {
        let data: Vec<f64> = (0..*size).map(|i| i as f64 / *size as f64).collect();
        
        group.bench_with_input(BenchmarkId::new("batch_size", size), size, |b, _| {
            b.iter(|| {
                model.predict(black_box(&data)).unwrap()
            })
        });
    }
    group.finish();
}

fn bench_different_methods(c: &mut Criterion) {
    let mut group = c.benchmark_group("calibration_methods");
    
    let methods = vec![
        (CalibrationMethod::A, vec![1.0]),
        (CalibrationMethod::B, vec![-1.0]),
        (CalibrationMethod::AB, vec![1.0, -0.5]),
    ];
    
    let test_data: Vec<f64> = (0..1000).map(|i| i as f64 / 1000.0).collect();
    
    for (method, weights) in methods {
        let model = BetaCalModel::new(weights, 0.0, method);
        
        group.bench_with_input(
            BenchmarkId::new("method", format!("{:?}", method)),
            &method,
            |b, _| {
                b.iter(|| {
                    model.predict(black_box(&test_data)).unwrap()
                })
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_single_prediction, bench_batch_prediction, bench_different_methods);
criterion_main!(benches);
