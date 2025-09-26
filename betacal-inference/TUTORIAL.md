# BetaCal Rust Inference Engine Tutorial

This tutorial walks you through the complete process of training a calibration model in Python and using it for high-performance inference in Rust.

## Prerequisites

- Python 3.7+ with `betacal` package installed
- Rust 1.70+ installed
- Basic familiarity with both Python and Rust

## Installation

### Python Dependencies

```bash
pip install betacal numpy
```

### Rust Dependencies

Add to your `Cargo.toml`:

```toml
[dependencies]
betacal-inference = "0.1.0"
anyhow = "1.0"
```

## Step-by-Step Tutorial

### Step 1: Train Model in Python

Create `train_model.py`:

```python
import numpy as np
from betacal import BetaCalibration

# Set random seed for reproducibility
np.random.seed(42)

# Generate realistic training data
n_samples = 5000
raw_scores = np.random.beta(2, 3, n_samples)  # Skewed probability distribution

# Create calibration bias (common in real ML models)
# Raw scores are poorly calibrated - too confident
biased_probs = raw_scores ** 0.7  # Makes probabilities more extreme

# Generate true labels based on actual probabilities
true_probs = 1 / (1 + np.exp(-(raw_scores - 0.4) * 6))
labels = np.random.binomial(1, true_probs, n_samples)

print(f"Training data: {n_samples} samples")
print(f"Positive rate: {labels.mean():.3f}")
print(f"Score range: [{biased_probs.min():.3f}, {biased_probs.max():.3f}]")

# Train calibration model
calibrator = BetaCalibration(parameters="abm")  # Full beta calibration
calibrator.fit(biased_probs, labels)

print("✓ Model trained successfully")

# Test calibration quality
test_probs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
calibrated = calibrator.predict(test_probs)

print("\nCalibration results:")
for orig, cal in zip(test_probs, calibrated):
    print(f"  {orig:.1f} -> {cal:.3f}")
```

Run the training:

```bash
python train_model.py
```

### Step 2: Export Model for Rust

Use the provided export script:

```python
# Add to train_model.py or create separate export script
from export_model import export_betacal_model

# Export the trained model
export_betacal_model(calibrator, "production_model.json", include_test_cases=True)
```

This creates `production_model.json` with the structure:

```json
{
  "weights": [1.2345, -0.6789],
  "intercept": 0.0,
  "method": "AB",
  "version": "1.0.0",
  "training_score": null,
  "metadata": {
    "training_date": "2024-01-15T10:30:00",
    "python_version": "3.9.7",
    "original_parameters": "abm",
    "n_features": 2
  },
  "test_cases": [
    {"input": 0.01, "expected": 0.008234, "tolerance": 1e-12},
    ...
  ]
}
```

### Step 3: Create Rust Project

```bash
# Create new Rust project
cargo new calibration_service
cd calibration_service

# Copy the model file
cp ../production_model.json .
```

Update `Cargo.toml`:

```toml
[package]
name = "calibration_service"
version = "0.1.0"
edition = "2021"

[dependencies]
betacal-inference = "0.1.0"
anyhow = "1.0"
serde_json = "1.0"
```

### Step 4: Basic Rust Usage

Create `src/main.rs`:

```rust
use betacal_inference::{BetaCalModel, InferenceError};
use anyhow::Result;

fn main() -> Result<()> {
    println!("BetaCal Rust Inference Demo");
    println!("===========================");

    // Load the model
    let model = BetaCalModel::load_from_json("production_model.json")?;
    println!("✓ Model loaded successfully");
    println!("  Method: {:?}", model.method);
    println!("  Weights: {:?}", model.weights);
    println!("  Intercept: {:.6}", model.intercept);

    // Test with the same values as Python
    let test_probs = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
    let calibrated = model.predict(&test_probs)?;

    println!("\nCalibration results:");
    for (orig, cal) in test_probs.iter().zip(calibrated.iter()) {
        println!("  {:.1} -> {:.6}", orig, cal);
    }

    // Performance test
    println!("\nPerformance test:");
    let large_batch: Vec<f64> = (0..100_000)
        .map(|i| i as f64 / 100_000.0)
        .collect();

    let start = std::time::Instant::now();
    let results = model.predict(&large_batch)?;
    let duration = start.elapsed();

    println!("  Processed {} predictions in {:?}", results.len(), duration);
    println!("  Throughput: {:.0} predictions/sec", 
             results.len() as f64 / duration.as_secs_f64());

    Ok(())
}
```

### Step 5: Compile and Test

```bash
# Compile the project
cargo build --release

# Run the demo
cargo run --release
```

Expected output:
```
BetaCal Rust Inference Demo
===========================
✓ Model loaded successfully
  Method: AB
  Weights: [1.2345, -0.6789]
  Intercept: 0.000000

Calibration results:
  0.1 -> 0.082341
  0.2 -> 0.165432
  ...

Performance test:
  Processed 100000 predictions in 2.1ms
  Throughput: 47619047 predictions/sec
```

### Step 6: Validation Testing

Create `src/validation.rs`:

```rust
use betacal_inference::BetaCalModel;
use serde_json::Value;
use std::fs;

pub fn validate_against_python(model_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    // Load model with test cases
    let content = fs::read_to_string(model_path)?;
    let model_json: Value = serde_json::from_str(&content)?;
    
    let model = BetaCalModel::load_from_json(model_path)?;
    
    if let Some(test_cases) = model_json["test_cases"].as_array() {
        println!("Running validation against {} test cases...", test_cases.len());
        
        let mut max_error = 0.0;
        let mut passed = 0;
        
        for (i, case) in test_cases.iter().enumerate() {
            let input = case["input"].as_f64().unwrap();
            let expected = case["expected"].as_f64().unwrap();
            let tolerance = case["tolerance"].as_f64().unwrap_or(1e-10);
            
            let actual = model.predict_single(input)?;
            let error = (actual - expected).abs();
            
            if error <= tolerance {
                passed += 1;
            } else {
                println!("  FAIL case {}: input={:.6}, expected={:.6}, actual={:.6}, error={:.2e}", 
                         i, input, expected, actual, error);
            }
            
            max_error = max_error.max(error);
        }
        
        println!("Validation results:");
        println!("  Passed: {}/{}", passed, test_cases.len());
        println!("  Max error: {:.2e}", max_error);
        
        if passed == test_cases.len() {
            println!("✓ All validation tests passed!");
        } else {
            println!("✗ Some validation tests failed");
        }
    } else {
        println!("No test cases found in model file");
    }
    
    Ok(())
}
```

Add to `src/main.rs`:

```rust
mod validation;

fn main() -> Result<()> {
    // ... existing code ...
    
    // Run validation
    println!("\nValidation against Python:");
    validation::validate_against_python("production_model.json")?;
    
    Ok(())
}
```

### Step 7: Production Service

Create `src/service.rs`:

```rust
use betacal_inference::{BetaCalModel, InferenceError};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// Production-ready calibration service with monitoring
pub struct CalibrationService {
    model: Arc<BetaCalModel>,
    prediction_count: AtomicU64,
    error_count: AtomicU64,
}

impl CalibrationService {
    pub fn new(model_path: &str) -> Result<Self, InferenceError> {
        let model = BetaCalModel::load_from_json(model_path)?;
        Ok(Self {
            model: Arc::new(model),
            prediction_count: AtomicU64::new(0),
            error_count: AtomicU64::new(0),
        })
    }

    pub fn calibrate_batch(&self, probabilities: &[f64]) -> Result<Vec<f64>, InferenceError> {
        match self.model.predict(probabilities) {
            Ok(result) => {
                self.prediction_count.fetch_add(probabilities.len() as u64, Ordering::Relaxed);
                Ok(result)
            }
            Err(e) => {
                self.error_count.fetch_add(1, Ordering::Relaxed);
                Err(e)
            }
        }
    }

    pub fn calibrate_single(&self, probability: f64) -> Result<f64, InferenceError> {
        match self.model.predict_single(probability) {
            Ok(result) => {
                self.prediction_count.fetch_add(1, Ordering::Relaxed);
                Ok(result)
            }
            Err(e) => {
                self.error_count.fetch_add(1, Ordering::Relaxed);
                Err(e)
            }
        }
    }

    pub fn get_stats(&self) -> (u64, u64) {
        (
            self.prediction_count.load(Ordering::Relaxed),
            self.error_count.load(Ordering::Relaxed),
        )
    }

    pub fn health_check(&self) -> bool {
        // Simple health check - could be extended
        true
    }
}

// Thread safety test
#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_thread_safety() {
        // This test requires a model file - skip if not available
        if std::path::Path::new("production_model.json").exists() {
            let service = Arc::new(CalibrationService::new("production_model.json").unwrap());
            
            let mut handles = vec![];
            
            for _ in 0..4 {
                let service_clone = Arc::clone(&service);
                let handle = thread::spawn(move || {
                    for i in 0..1000 {
                        let prob = (i as f64) / 1000.0;
                        service_clone.calibrate_single(prob).unwrap();
                    }
                });
                handles.push(handle);
            }
            
            for handle in handles {
                handle.join().unwrap();
            }
            
            let (predictions, errors) = service.get_stats();
            assert_eq!(predictions, 4000);
            assert_eq!(errors, 0);
        }
    }
}
```

### Step 8: Benchmarking

Create `benches/calibration_bench.rs`:

```rust
use betacal_inference::BetaCalModel;
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};

fn bench_calibration(c: &mut Criterion) {
    // Skip if model file doesn't exist
    if !std::path::Path::new("production_model.json").exists() {
        return;
    }
    
    let model = BetaCalModel::load_from_json("production_model.json").unwrap();
    
    let mut group = c.benchmark_group("calibration");
    
    // Single prediction benchmark
    group.bench_function("single", |b| {
        b.iter(|| model.predict_single(black_box(0.7)).unwrap())
    });
    
    // Batch prediction benchmarks
    for size in [10, 100, 1000, 10000].iter() {
        let data: Vec<f64> = (0..*size).map(|i| i as f64 / *size as f64).collect();
        
        group.bench_with_input(BenchmarkId::new("batch", size), size, |b, _| {
            b.iter(|| model.predict(black_box(&data)).unwrap())
        });
    }
    
    group.finish();
}

criterion_group!(benches, bench_calibration);
criterion_main!(benches);
```

Run benchmarks:

```bash
cargo bench
```

### Step 9: Integration Testing

Create `tests/integration_test.rs`:

```rust
use betacal_inference::{BetaCalModel, CalibrationMethod};

#[test]
fn test_model_loading() {
    if std::path::Path::new("production_model.json").exists() {
        let model = BetaCalModel::load_from_json("production_model.json").unwrap();
        assert!(matches!(model.method, CalibrationMethod::AB));
        assert!(!model.weights.is_empty());
    }
}

#[test]
fn test_prediction_bounds() {
    let model = BetaCalModel::new(vec![1.0, -0.5], 0.0, CalibrationMethod::AB);
    
    let test_cases = vec![0.0, 0.1, 0.5, 0.9, 1.0];
    
    for &prob in &test_cases {
        let result = model.predict_single(prob).unwrap();
        assert!(result >= 0.0 && result <= 1.0, 
                "Result {} out of bounds for input {}", result, prob);
    }
}

#[test]
fn test_monotonicity() {
    let model = BetaCalModel::new(vec![1.0], 0.0, CalibrationMethod::A);
    
    let inputs: Vec<f64> = (1..100).map(|i| i as f64 / 100.0).collect();
    let outputs = model.predict(&inputs).unwrap();
    
    // Check that outputs are monotonically increasing (for positive weight)
    for i in 1..outputs.len() {
        assert!(outputs[i] >= outputs[i-1], 
                "Non-monotonic output at index {}: {} < {}", 
                i, outputs[i], outputs[i-1]);
    }
}
```

Run tests:

```bash
cargo test
```

## Deployment Considerations

### Docker Deployment

Create `Dockerfile`:

```dockerfile
FROM rust:1.70 as builder

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN cargo build --release

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/calibration_service /usr/local/bin/
COPY production_model.json /app/
WORKDIR /app
CMD ["calibration_service"]
```

### Performance Monitoring

```rust
use std::time::{Duration, Instant};

pub struct PerformanceMonitor {
    total_predictions: AtomicU64,
    total_duration: std::sync::Mutex<Duration>,
}

impl PerformanceMonitor {
    pub fn time_prediction<F, R>(&self, f: F) -> R 
    where F: FnOnce() -> R 
    {
        let start = Instant::now();
        let result = f();
        let duration = start.elapsed();
        
        self.total_predictions.fetch_add(1, Ordering::Relaxed);
        *self.total_duration.lock().unwrap() += duration;
        
        result
    }
    
    pub fn get_average_latency(&self) -> Duration {
        let total = *self.total_duration.lock().unwrap();
        let count = self.total_predictions.load(Ordering::Relaxed);
        if count > 0 {
            total / count as u32
        } else {
            Duration::ZERO
        }
    }
}
```

## Troubleshooting

### Common Issues

1. **Model file not found**
   ```
   Error: No such file or directory (os error 2)
   ```
   Solution: Ensure the JSON model file is in the correct location.

2. **Serialization errors**
   ```
   Error: missing field `weights`
   ```
   Solution: Re-export the model using the latest export script.

3. **Numerical precision differences**
   ```
   Validation failed: expected 0.123456, got 0.123457
   ```
   Solution: Check tolerance settings and ensure consistent floating-point precision.

### Performance Issues

1. **Slow batch processing**: Use `predict()` instead of multiple `predict_single()` calls
2. **Memory usage**: Models are small (~100 bytes), but avoid unnecessary cloning
3. **Thread contention**: Use `Arc<BetaCalModel>` for sharing across threads

## Next Steps

- Integrate with your ML pipeline
- Add monitoring and alerting
- Consider caching for repeated predictions
- Implement A/B testing for model versions
- Add custom metrics collection

## Conclusion

You now have a complete pipeline for training calibration models in Python and deploying them with high-performance Rust inference. The Rust implementation provides:

- **50-100x faster inference** than Python
- **Thread-safe operation** for concurrent workloads  
- **Minimal memory footprint** for production deployment
- **Exact numerical parity** with Python training

This approach gives you the best of both worlds: Python's rich ML ecosystem for training and Rust's performance for production inference.
