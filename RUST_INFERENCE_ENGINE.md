# BetaCal Rust Inference Engine

**Version:** 1.0  
**Date:** September 26, 2025  
**Purpose:** Minimal Rust module for production inference from Python-trained BetaCal models

## Overview

This approach separates training (Python) from inference (Rust), providing:
- **Python training pipeline** - Use existing BetaCal for model fitting
- **Rust inference engine** - High-performance prediction in production
- **Model serialization** - Bridge between Python training and Rust inference

## Architecture

```
Training (Python)          Production (Rust)
┌─────────────────┐        ┌──────────────────┐
│   BetaCal.fit() │───────►│  Model Parameters│
│                 │        │  (JSON/Binary)   │
└─────────────────┘        └──────────────────┘
                                     │
                                     ▼
                           ┌──────────────────┐
                           │ Rust Inference   │
                           │ Engine           │
                           └──────────────────┘
```

## Core Implementation

### Minimal Rust Crate Structure
```
betacal-inference/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── models.rs      # Model parameter structures
│   ├── inference.rs   # Core prediction logic
│   └── serialization.rs # Model loading
└── examples/
    └── basic_usage.rs
```

### Dependencies
```toml
[dependencies]
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
thiserror = "1.0"

[dev-dependencies]
approx = "0.5"
```

### Core Model Structure
```rust
// src/models.rs
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BetaCalModel {
    pub method: CalibrationMethod,
    pub parameters: [f64; 3], // [a, b, m]
    pub logistic_weights: Vec<f64>,
    pub logistic_intercept: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CalibrationMethod {
    Beta,   // Full beta (a, b, m)
    BetaAM, // a=b, variable m
    BetaAB, // variable a,b, m=0.5
    BetaA,  // a=b, m=0.5
}

impl BetaCalModel {
    pub fn predict(&self, probabilities: &[f64]) -> Result<Vec<f64>, InferenceError> {
        // Input validation
        if probabilities.is_empty() {
            return Err(InferenceError::EmptyInput);
        }

        // Clip probabilities to avoid numerical issues
        let clipped: Vec<f64> = probabilities
            .iter()
            .map(|&p| p.clamp(f64::EPSILON, 1.0 - f64::EPSILON))
            .collect();

        // Apply beta transformation based on method
        let features = self.transform_features(&clipped)?;
        
        // Apply logistic regression
        self.apply_logistic_regression(&features)
    }

    fn transform_features(&self, probs: &[f64]) -> Result<Vec<Vec<f64>>, InferenceError> {
        let [a, b, _m] = self.parameters;
        
        match self.method {
            CalibrationMethod::Beta => {
                // Full beta transformation: log(p), log(1-p)
                probs.iter().map(|&p| {
                    let log_p = p.ln();
                    let log_1_minus_p = (1.0 - p).ln();
                    
                    if a == 0.0 {
                        Ok(vec![-log_1_minus_p]) // Only log(1-p) term
                    } else if b == 0.0 {
                        Ok(vec![log_p]) // Only log(p) term
                    } else {
                        Ok(vec![log_p, -log_1_minus_p]) // Both terms
                    }
                }).collect()
            },
            CalibrationMethod::BetaAM => {
                // Beta AM: log(p/(1-p)) - logit transformation
                probs.iter().map(|&p| {
                    Ok(vec![(p / (1.0 - p)).ln()])
                }).collect()
            },
            CalibrationMethod::BetaAB => {
                // Beta AB: log(2*p), log(2*(1-p))
                probs.iter().map(|&p| {
                    Ok(vec![(2.0 * p).ln(), (2.0 * (1.0 - p)).ln()])
                }).collect()
            },
            CalibrationMethod::BetaA => {
                // Beta A: log(p/(1-p)) - same as AM
                probs.iter().map(|&p| {
                    Ok(vec![(p / (1.0 - p)).ln()])
                }).collect()
            },
        }
    }

    fn apply_logistic_regression(&self, features: &[Vec<f64>]) -> Result<Vec<f64>, InferenceError> {
        features.iter().map(|feature_row| {
            // Linear combination: intercept + sum(weights * features)
            let linear_combination = self.logistic_intercept + 
                feature_row.iter()
                    .zip(self.logistic_weights.iter())
                    .map(|(f, w)| f * w)
                    .sum::<f64>();
            
            // Apply sigmoid
            Ok(sigmoid(linear_combination))
        }).collect()
    }
}

#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}
```

### Error Handling
```rust
// src/lib.rs
use thiserror::Error;

#[derive(Error, Debug)]
pub enum InferenceError {
    #[error("Empty input provided")]
    EmptyInput,
    #[error("Invalid probability value: {0}")]
    InvalidProbability(f64),
    #[error("Model serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}
```

### Model Loading
```rust
// src/serialization.rs
use std::fs;
use crate::{BetaCalModel, InferenceError};

impl BetaCalModel {
    pub fn from_json_file(path: &str) -> Result<Self, InferenceError> {
        let content = fs::read_to_string(path)?;
        let model: BetaCalModel = serde_json::from_str(&content)?;
        Ok(model)
    }
    
    pub fn from_json_str(json: &str) -> Result<Self, InferenceError> {
        let model: BetaCalModel = serde_json::from_str(json)?;
        Ok(model)
    }
    
    pub fn to_json_file(&self, path: &str) -> Result<(), InferenceError> {
        let json = serde_json::to_string_pretty(self)?;
        fs::write(path, json)?;
        Ok(())
    }
}
```

## Python Export Script

Create a Python script to export trained models:

```python
# export_model.py
import json
import numpy as np
from betacal import BetaCalibration

def export_betacal_model(calibrator, output_path):
    """Export a trained BetaCal model to JSON for Rust inference."""
    
    # Extract method
    method_map = {
        "abm": "Beta",
        "am": "BetaAM", 
        "ab": "BetaAB",
        "a": "BetaA"
    }
    
    # Get internal calibrator
    internal_cal = calibrator.calibrator_
    
    # Extract parameters and logistic regression weights
    model_data = {
        "method": method_map[calibrator.parameters],
        "parameters": internal_cal.map_.tolist(),
        "logistic_weights": internal_cal.lr_.coef_[0].tolist(),
        "logistic_intercept": float(internal_cal.lr_.intercept_[0])
    }
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(model_data, f, indent=2)
    
    print(f"Model exported to {output_path}")

# Usage example
if __name__ == "__main__":
    # Train model
    probs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    labels = np.array([0, 0, 1, 1, 1])
    
    cal = BetaCalibration(parameters="abm")
    cal.fit(probs, labels)
    
    # Export for Rust
    export_betacal_model(cal, "model.json")
```

## Rust Usage Examples

### Basic Usage
```rust
// examples/basic_usage.rs
use betacal_inference::{BetaCalModel, InferenceError};

fn main() -> Result<(), InferenceError> {
    // Load model exported from Python
    let model = BetaCalModel::from_json_file("model.json")?;
    
    // Make predictions
    let test_probs = vec![0.2, 0.4, 0.6, 0.8];
    let calibrated = model.predict(&test_probs)?;
    
    println!("Original:   {:?}", test_probs);
    println!("Calibrated: {:?}", calibrated);
    
    Ok(())
}
```

### Production Server Integration
```rust
// Production usage with error handling
use betacal_inference::{BetaCalModel, InferenceError};
use std::sync::Arc;

pub struct CalibrationService {
    model: Arc<BetaCalModel>,
}

impl CalibrationService {
    pub fn new(model_path: &str) -> Result<Self, InferenceError> {
        let model = BetaCalModel::from_json_file(model_path)?;
        Ok(Self {
            model: Arc::new(model),
        })
    }
    
    pub fn calibrate_batch(&self, probabilities: &[f64]) -> Result<Vec<f64>, InferenceError> {
        self.model.predict(probabilities)
    }
    
    pub fn calibrate_single(&self, probability: f64) -> Result<f64, InferenceError> {
        let result = self.model.predict(&[probability])?;
        Ok(result[0])
    }
}

// Thread-safe usage
fn production_example() -> Result<(), InferenceError> {
    let service = Arc::new(CalibrationService::new("production_model.json")?);
    
    // Can be shared across threads
    let service_clone = Arc::clone(&service);
    let handle = std::thread::spawn(move || {
        service_clone.calibrate_batch(&[0.1, 0.5, 0.9])
    });
    
    let result = handle.join().unwrap()?;
    println!("Calibrated: {:?}", result);
    
    Ok(())
}
```

## Workflow

### 1. Training Phase (Python)
```python
from betacal import BetaCalibration
import numpy as np

# Train your model
cal = BetaCalibration(parameters="abm")
cal.fit(train_probs, train_labels)

# Export for production
export_betacal_model(cal, "production_model.json")
```

### 2. Production Phase (Rust)
```rust
// Load once at startup
let model = BetaCalModel::from_json_file("production_model.json")?;

// Use for inference
let calibrated = model.predict(&uncalibrated_probs)?;
```

## Performance Benefits

- **No Python runtime** in production
- **Zero-copy operations** for small arrays
- **Memory efficient** - only stores essential parameters
- **Thread-safe** - can be shared across threads
- **Fast startup** - no ML library initialization

## Implementation Timeline

**Week 1: Core Implementation**
- [ ] Basic model structures
- [ ] Core prediction logic
- [ ] JSON serialization
- [ ] Python export script

**Week 2: Production Features**
- [ ] Error handling
- [ ] Performance optimizations
- [ ] Thread safety
- [ ] Documentation

**Total: 2 weeks** for a production-ready inference engine

This approach gives you the best of both worlds: Python's rich ML ecosystem for training and Rust's performance for production inference.
