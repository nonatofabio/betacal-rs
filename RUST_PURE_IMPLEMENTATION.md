# Pure Rust BetaCal Implementation Strategy

**Version:** 1.0  
**Date:** September 26, 2025  
**Target:** Standalone Rust library without Python dependencies

## Overview

This document outlines the strategy for implementing a pure Rust version of BetaCal that operates independently of Python, providing a high-performance calibration library for the Rust ecosystem.

## Architecture Design

### Core Structure
```
betacal-rs/
├── src/
│   ├── lib.rs              # Public API
│   ├── calibration/        # Core calibration algorithms
│   │   ├── mod.rs
│   │   ├── beta_cal.rs     # Full beta calibration (a, b, m)
│   │   ├── beta_am_cal.rs  # Beta AM calibration (a=b, m)
│   │   ├── beta_ab_cal.rs  # Beta AB calibration (a, b, m=0.5)
│   │   └── beta_a_cal.rs   # Beta A calibration (a=b, m=0.5)
│   ├── optimization/       # Optimization algorithms
│   │   ├── mod.rs
│   │   └── logistic.rs     # Logistic regression implementation
│   ├── utils/              # Utilities
│   │   ├── mod.rs
│   │   ├── validation.rs   # Input validation
│   │   └── math.rs         # Mathematical utilities
│   └── error.rs            # Error handling
├── examples/               # Usage examples
├── benches/               # Performance benchmarks
└── tests/                 # Integration tests
```

## Core Dependencies

```toml
[dependencies]
# Core numerical computing
ndarray = "0.15"
ndarray-linalg = "0.16"
ndarray-rand = "0.14"

# Optimization
argmin = "0.8"
argmin-math = "0.3"

# Statistics and distributions
statrs = "0.16"

# Serialization (optional)
serde = { version = "1.0", features = ["derive"], optional = true }

# Parallel processing
rayon = "1.7"

# Error handling
thiserror = "1.0"

[dev-dependencies]
approx = "0.5"
criterion = "0.5"
proptest = "1.4"
```

## API Design

### Core Traits
```rust
// src/lib.rs
pub trait Calibrator {
    type Error;
    
    /// Fit the calibrator to training data
    fn fit(&mut self, probabilities: &[f64], labels: &[bool]) -> Result<(), Self::Error>;
    
    /// Predict calibrated probabilities
    fn predict(&self, probabilities: &[f64]) -> Result<Vec<f64>, Self::Error>;
    
    /// Check if the calibrator has been fitted
    fn is_fitted(&self) -> bool;
}

/// Main calibration factory
pub enum CalibrationMethod {
    Beta,      // Full beta calibration (a, b, m)
    BetaAM,    // Beta with a=b, variable m
    BetaAB,    // Beta with variable a,b, m=0.5
    BetaA,     // Beta with a=b, m=0.5
}

pub struct BetaCalibration {
    method: CalibrationMethod,
    calibrator: Box<dyn Calibrator<Error = CalibrationError>>,
}

impl BetaCalibration {
    pub fn new(method: CalibrationMethod) -> Self { /* ... */ }
    
    pub fn fit(&mut self, probabilities: &[f64], labels: &[bool]) -> Result<(), CalibrationError> {
        self.calibrator.fit(probabilities, labels)
    }
    
    pub fn predict(&self, probabilities: &[f64]) -> Result<Vec<f64>, CalibrationError> {
        self.calibrator.predict(probabilities)
    }
}
```

### Individual Calibrators
```rust
// src/calibration/beta_cal.rs
pub struct BetaCal {
    parameters: Option<[f64; 3]>, // [a, b, m]
    logistic_model: Option<LogisticRegression>,
}

impl Calibrator for BetaCal {
    type Error = CalibrationError;
    
    fn fit(&mut self, probabilities: &[f64], labels: &[bool]) -> Result<(), Self::Error> {
        // 1. Validate inputs
        validate_inputs(probabilities, labels)?;
        
        // 2. Transform data for beta calibration
        let (features, targets) = prepare_beta_features(probabilities, labels);
        
        // 3. Fit logistic regression
        let mut lr = LogisticRegression::new();
        lr.fit(&features, &targets)?;
        
        // 4. Extract beta parameters
        let coefficients = lr.coefficients();
        self.parameters = Some(extract_beta_parameters(&coefficients));
        self.logistic_model = Some(lr);
        
        Ok(())
    }
    
    fn predict(&self, probabilities: &[f64]) -> Result<Vec<f64>, Self::Error> {
        let params = self.parameters.ok_or(CalibrationError::NotFitted)?;
        let lr = self.logistic_model.as_ref().ok_or(CalibrationError::NotFitted)?;
        
        // Apply beta transformation
        let features = prepare_prediction_features(probabilities, &params);
        lr.predict_proba(&features)
    }
    
    fn is_fitted(&self) -> bool {
        self.parameters.is_some()
    }
}
```

## Implementation Details

### Mathematical Core
```rust
// src/utils/math.rs
use ndarray::Array1;

/// Clip values to avoid numerical issues
pub fn clip_probabilities(probs: &[f64]) -> Vec<f64> {
    const EPS: f64 = f64::EPSILON;
    probs.iter()
        .map(|&p| p.clamp(EPS, 1.0 - EPS))
        .collect()
}

/// Log-odds transformation
pub fn logit(p: f64) -> f64 {
    (p / (1.0 - p)).ln()
}

/// Inverse logit (sigmoid)
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Prepare features for beta calibration
pub fn prepare_beta_features(probs: &[f64], labels: &[bool]) -> (Array2<f64>, Array1<f64>) {
    let clipped = clip_probabilities(probs);
    let n = clipped.len();
    
    // Create feature matrix for beta calibration
    let mut features = Array2::zeros((n, 2));
    let mut targets = Array1::zeros(n);
    
    for (i, (&p, &label)) in clipped.iter().zip(labels.iter()).enumerate() {
        features[[i, 0]] = p.ln();
        features[[i, 1]] = (1.0 - p).ln();
        targets[i] = if label { 1.0 } else { 0.0 };
    }
    
    (features, targets)
}
```

### Logistic Regression Implementation
```rust
// src/optimization/logistic.rs
use argmin::core::{CostFunction, Executor, Gradient};
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::LBFGS;

pub struct LogisticRegression {
    coefficients: Option<Vec<f64>>,
    intercept: Option<f64>,
}

impl LogisticRegression {
    pub fn new() -> Self {
        Self {
            coefficients: None,
            intercept: None,
        }
    }
    
    pub fn fit(&mut self, features: &Array2<f64>, targets: &Array1<f64>) -> Result<(), CalibrationError> {
        let problem = LogisticRegressionProblem::new(features.clone(), targets.clone());
        
        // Initialize parameters
        let init_param = vec![0.0; features.ncols() + 1]; // +1 for intercept
        
        // Set up optimizer
        let linesearch = MoreThuenteLineSearch::new();
        let solver = LBFGS::new(linesearch, 10);
        
        let res = Executor::new(problem, solver)
            .configure(|state| state.param(init_param).max_iters(1000))
            .run()?;
        
        let params = res.state().best_param.as_ref().unwrap();
        self.intercept = Some(params[0]);
        self.coefficients = Some(params[1..].to_vec());
        
        Ok(())
    }
    
    pub fn predict_proba(&self, features: &Array2<f64>) -> Result<Vec<f64>, CalibrationError> {
        let coef = self.coefficients.as_ref().ok_or(CalibrationError::NotFitted)?;
        let intercept = self.intercept.ok_or(CalibrationError::NotFitted)?;
        
        let mut predictions = Vec::with_capacity(features.nrows());
        
        for row in features.rows() {
            let linear_combination = intercept + 
                row.iter().zip(coef.iter()).map(|(x, w)| x * w).sum::<f64>();
            predictions.push(sigmoid(linear_combination));
        }
        
        Ok(predictions)
    }
}

// Cost function for optimization
struct LogisticRegressionProblem {
    features: Array2<f64>,
    targets: Array1<f64>,
}

impl CostFunction for LogisticRegressionProblem {
    type Param = Vec<f64>;
    type Output = f64;
    
    fn cost(&self, param: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
        // Implement logistic loss
        let intercept = param[0];
        let coefficients = &param[1..];
        
        let mut loss = 0.0;
        for (i, row) in self.features.rows().enumerate() {
            let linear_combination = intercept + 
                row.iter().zip(coefficients.iter()).map(|(x, w)| x * w).sum::<f64>();
            let prediction = sigmoid(linear_combination);
            let target = self.targets[i];
            
            // Cross-entropy loss
            loss -= target * prediction.ln() + (1.0 - target) * (1.0 - prediction).ln();
        }
        
        Ok(loss / self.features.nrows() as f64)
    }
}
```

## Usage Examples

### Basic Usage
```rust
use betacal_rs::{BetaCalibration, CalibrationMethod};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Training data
    let probabilities = vec![0.1, 0.3, 0.5, 0.7, 0.9];
    let labels = vec![false, false, true, true, true];
    
    // Create and fit calibrator
    let mut calibrator = BetaCalibration::new(CalibrationMethod::Beta);
    calibrator.fit(&probabilities, &labels)?;
    
    // Make predictions
    let test_probs = vec![0.2, 0.6, 0.8];
    let calibrated = calibrator.predict(&test_probs)?;
    
    println!("Original: {:?}", test_probs);
    println!("Calibrated: {:?}", calibrated);
    
    Ok(())
}
```

### Advanced Usage with Serialization
```rust
use betacal_rs::{BetaCalibration, CalibrationMethod};
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
struct CalibrationModel {
    calibrator: BetaCalibration,
    metadata: ModelMetadata,
}

fn save_and_load_model() -> Result<(), Box<dyn std::error::Error>> {
    // Train model
    let mut calibrator = BetaCalibration::new(CalibrationMethod::BetaAM);
    // ... training code ...
    
    // Save model
    let model = CalibrationModel {
        calibrator,
        metadata: ModelMetadata::new(),
    };
    
    let serialized = serde_json::to_string(&model)?;
    std::fs::write("calibrator.json", serialized)?;
    
    // Load model
    let loaded_data = std::fs::read_to_string("calibrator.json")?;
    let loaded_model: CalibrationModel = serde_json::from_str(&loaded_data)?;
    
    Ok(())
}
```

## Distribution Strategy

### Crate Structure
```toml
[package]
name = "betacal"
version = "0.1.0"
edition = "2021"
authors = ["Your Name <email@example.com>"]
license = "MIT"
description = "High-performance beta calibration for probability calibration"
repository = "https://github.com/username/betacal-rs"
keywords = ["machine-learning", "calibration", "statistics", "probability"]
categories = ["algorithms", "science"]

[features]
default = ["serde"]
serde = ["dep:serde"]
parallel = ["rayon"]
```

### Documentation
- Comprehensive API documentation with `cargo doc`
- Usage examples in `examples/` directory
- Performance benchmarks in `benches/` directory
- Integration with docs.rs for online documentation

### Testing Strategy
```rust
// tests/integration_tests.rs
use betacal::*;
use approx::assert_relative_eq;

#[test]
fn test_calibration_preserves_ordering() {
    let probs = vec![0.1, 0.3, 0.5, 0.7, 0.9];
    let labels = vec![false, false, true, true, true];
    
    let mut cal = BetaCalibration::new(CalibrationMethod::Beta);
    cal.fit(&probs, &labels).unwrap();
    
    let calibrated = cal.predict(&probs).unwrap();
    
    // Check that ordering is preserved
    for i in 1..calibrated.len() {
        assert!(calibrated[i-1] <= calibrated[i]);
    }
}

#[test]
fn test_numerical_stability() {
    let extreme_probs = vec![1e-10, 0.5, 1.0 - 1e-10];
    let labels = vec![false, true, true];
    
    let mut cal = BetaCalibration::new(CalibrationMethod::Beta);
    cal.fit(&extreme_probs, &labels).unwrap();
    
    let result = cal.predict(&extreme_probs).unwrap();
    
    // Should not panic or produce NaN/Inf
    for &val in &result {
        assert!(val.is_finite());
        assert!(val >= 0.0 && val <= 1.0);
    }
}
```

## Advantages of Pure Rust Implementation

### Performance Benefits
- **Zero-copy operations** where possible
- **SIMD optimizations** for vectorized operations
- **Parallel processing** with Rayon
- **Memory efficiency** with stack allocation
- **No GIL limitations** unlike Python

### Ecosystem Integration
- **Native Rust ML ecosystem** integration (candle, linfa, etc.)
- **WebAssembly support** for browser deployment
- **Embedded systems** compatibility
- **Cross-compilation** to multiple targets

### Development Benefits
- **Memory safety** without garbage collection overhead
- **Fearless concurrency** with Rust's ownership system
- **Excellent tooling** (cargo, clippy, rustfmt)
- **Strong type system** prevents many runtime errors

## Migration Path from Python

### Phase 1: Core Implementation (4 weeks)
- [ ] Implement basic beta calibration algorithms
- [ ] Add logistic regression optimization
- [ ] Create comprehensive test suite
- [ ] Basic documentation

### Phase 2: Feature Completeness (2 weeks)
- [ ] All calibration variants (AM, AB, A)
- [ ] Error handling and validation
- [ ] Serialization support
- [ ] Performance optimizations

### Phase 3: Ecosystem Integration (2 weeks)
- [ ] Integration with ndarray ecosystem
- [ ] WebAssembly bindings
- [ ] CLI tool for standalone usage
- [ ] Benchmarking suite

### Phase 4: Distribution (1 week)
- [ ] Crates.io publication
- [ ] Documentation hosting
- [ ] CI/CD pipeline
- [ ] Community outreach

## Conclusion

A pure Rust implementation offers significant advantages:
- **10-50x performance improvements** over Python
- **Memory safety** and **fearless concurrency**
- **Broader deployment options** (WASM, embedded, etc.)
- **Growing Rust ML ecosystem** integration

The implementation maintains the mathematical core while leveraging Rust's strengths for high-performance, safe systems programming.
