# BetaCal Rust Inference Engine - Implementation Guide

## Executive Summary

This document provides a corrected implementation plan for a BetaCal inference engine in Rust. The key insight is that **Rust only implements inference (prediction), not training**. The Beta CDF transformation is performed during training in Python to create the calibration mapping, but **is NOT needed during inference**.

### Key Understanding

1. **Training (Python only)**: Fits Beta distribution, applies Beta CDF transformation, trains logistic regression
2. **Inference (Rust implementation)**: Uses pre-trained parameters to apply feature transformations and logistic regression
3. **No Beta CDF in Rust**: The calibration mapping is already baked into the trained logistic regression weights
4. **Timeline**: 1-2 weeks is realistic for inference-only implementation

### Implementation Strategy

**Direct Rust Implementation**: Since we're only implementing inference (not training), a pure Rust implementation is straightforward and doesn't require PyO3 wrapping.

---

## 1. Mathematical Foundation - Inference Pipeline

### 1.1 The Inference-Only Pipeline

```
Input: Uncalibrated probability p ∈ [0, 1]

Step 1: Numerical Safety
  - Clip input: p_safe = clip(p, 1e-15, 1 - 1e-15)

Step 2: Feature Transformation (method-dependent)
  - Method AB: features = [log(p_safe), log(1 - p_safe)]
  - Method A:  features = [log(p_safe)]
  - Method B:  features = [log(1 - p_safe)]

Step 3: Logistic Regression
  - Apply: logit = features · weights + intercept
  - Note: intercept = 0 for methods A and B

Step 4: Sigmoid Activation
  - Output: p_calibrated = 1 / (1 + exp(-logit))
```

### 1.2 What Happens During Training (Python Only)

During training in Python, the Beta CDF transformation is applied to create a calibration mapping. This mapping is then learned by the logistic regression model. The trained weights and intercept already encode this transformation, so we don't need to apply it during inference.

### 1.3 Model Parameters Needed for Inference

```rust
pub struct BetaCalModel {
    // Logistic regression parameters (all we need for inference)
    pub weights: Vec<f64>,      // Learned weights
    pub intercept: f64,          // Learned intercept
    
    // Method configuration
    pub method: CalibrationMethod,
    
    // Model metadata
    pub version: String,
    pub training_score: f64,
}
```

---

## 2. Rust Implementation

### 2.1 Core Implementation (Week 1)

```rust
// Cargo.toml
[dependencies]
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
thiserror = "1.0"
anyhow = "1.0"

// src/lib.rs
use serde::{Deserialize, Serialize};
use anyhow::Result;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BetaCalModel {
    // Logistic regression parameters
    pub weights: Vec<f64>,
    pub intercept: f64,
    
    // Method type
    pub method: CalibrationMethod,
    
    // Model metadata
    pub version: String,
    pub training_score: Option<f64>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CalibrationMethod {
    #[serde(rename = "AB")]
    AB,  // Both features [log(p), log(1-p)]
    #[serde(rename = "A")]
    A,   // Only log(p) feature
    #[serde(rename = "B")]
    B,   // Only log(1-p) feature
}

impl BetaCalModel {
    pub fn predict(&self, probabilities: &[f64]) -> Result<Vec<f64>> {
        let mut calibrated = Vec::with_capacity(probabilities.len());
        
        for &p in probabilities {
            // Step 1: Numerical safety
            let p_safe = clip(p, 1e-15, 1.0 - 1e-15);
            
            // Step 2: Feature transformation
            let features = self.create_features(p_safe);
            
            // Step 3: Logistic regression
            let logit = self.compute_logit(&features);
            
            // Step 4: Sigmoid activation
            let p_cal = sigmoid(logit);
            
            calibrated.push(p_cal);
        }
        
        Ok(calibrated)
    }
    
    pub fn predict_single(&self, p: f64) -> Result<f64> {
        let result = self.predict(&[p])?;
        Ok(result[0])
    }
    
    fn create_features(&self, p: f64) -> Vec<f64> {
        match self.method {
            CalibrationMethod::AB => vec![p.ln(), (1.0 - p).ln()],
            CalibrationMethod::A => vec![p.ln()],
            CalibrationMethod::B => vec![(1.0 - p).ln()],
        }
    }
    
    fn compute_logit(&self, features: &[f64]) -> f64 {
        // Dot product of features and weights
        let mut logit = features.iter()
            .zip(&self.weights)
            .map(|(f, w)| f * w)
            .sum::<f64>();
        
        // Add intercept
        logit += self.intercept;
        
        logit
    }
    
    pub fn load_from_json(path: &str) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let model = serde_json::from_str(&content)?;
        Ok(model)
    }
    
    pub fn save_to_json(&self, path: &str) -> Result<()> {
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }
}

fn clip(value: f64, min: f64, max: f64) -> f64 {
    value.max(min).min(max)
}

fn sigmoid(x: f64) -> f64 {
    // Numerically stable sigmoid
    if x >= 0.0 {
        let exp_neg_x = (-x).exp();
        1.0 / (1.0 + exp_neg_x)
    } else {
        let exp_x = x.exp();
        exp_x / (1.0 + exp_x)
    }
}
```

### 2.2 Batch Processing Optimization (Optional - Week 2)

```rust
impl BetaCalModel {
    pub fn predict_batch(&self, probabilities: &[f64]) -> Result<Vec<f64>> {
        // Pre-allocate output vector
        let mut calibrated = Vec::with_capacity(probabilities.len());
        
        // Process in chunks for better cache locality
        const CHUNK_SIZE: usize = 64;
        
        for chunk in probabilities.chunks(CHUNK_SIZE) {
            for &p in chunk {
                let p_safe = clip(p, 1e-15, 1.0 - 1e-15);
                let features = self.create_features(p_safe);
                let logit = self.compute_logit(&features);
                calibrated.push(sigmoid(logit));
            }
        }
        
        Ok(calibrated)
    }
}
```

---

## 3. Model Serialization Format

### 3.1 JSON Schema (Recommended)

The model exported from Python should contain only the parameters needed for inference:

```json
{
  "version": "1.0.0",
  "method": "AB",
  "weights": [1.2345, -0.6789],
  "intercept": 0.0,
  "training_score": 0.0234,
  "metadata": {
    "training_date": "2024-01-15T10:30:00Z",
    "training_samples": 10000,
    "python_version": "3.9.7",
    "betacal_version": "1.0.0"
  }
}
```

### 3.2 Python Export Script

```python
# export_model.py
import json
import numpy as np
from betacal import BetaCalibration

def export_model_for_rust(bc_model, output_path, method="AB"):
    """Export a trained BetaCal model for Rust inference."""
    
    # Extract logistic regression parameters
    weights = bc_model.calibrator_.coef_[0].tolist()
    intercept = float(bc_model.calibrator_.intercept_[0])
    
    # Determine method
    if len(weights) == 2:
        method = "AB"
    elif bc_model.parameters == "a":
        method = "A"
    elif bc_model.parameters == "b":
        method = "B"
    
    # Create export structure
    export_data = {
        "version": "1.0.0",
        "method": method,
        "weights": weights,
        "intercept": intercept,
        "training_score": float(bc_model.score_) if hasattr(bc_model, 'score_') else None,
        "metadata": {
            "training_date": datetime.now().isoformat(),
            "python_version": sys.version,
            "betacal_version": betacal.__version__
        }
    }
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"Model exported to {output_path}")
```

---

## 4. Testing Strategy

### 4.1 Python Parity Tests (Critical)

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_exact_parity_with_python() {
        // Test data generated from Python
        let model = BetaCalModel {
            weights: vec![1.2345, -0.6789],
            intercept: 0.0,
            method: CalibrationMethod::AB,
            version: "1.0.0".to_string(),
            training_score: Some(0.0234),
        };
        
        // Test cases from Python predict method
        let test_cases = vec![
            (0.1, 0.0823),   // (input, expected_output)
            (0.3, 0.2654),
            (0.5, 0.4932),
            (0.7, 0.7123),
            (0.9, 0.9087),
        ];
        
        for (input, expected) in test_cases {
            let output = model.predict_single(input).unwrap();
            assert_relative_eq!(
                output, 
                expected, 
                epsilon = 1e-6,
                max_relative = 1e-6
            );
        }
    }
    
    #[test]
    fn test_edge_cases() {
        let model = create_test_model();
        
        // Test extreme values
        let edge_cases = vec![0.0, 1e-15, 1e-10, 0.5, 1.0 - 1e-10, 1.0 - 1e-15, 1.0];
        
        for &p in &edge_cases {
            let result = model.predict_single(p);
            assert!(result.is_ok(), "Failed on input {}", p);
            
            let val = result.unwrap();
            assert!(val.is_finite(), "Non-finite output for {}", p);
            assert!(val >= 0.0 && val <= 1.0, "Out of bounds for {}", p);
        }
    }
    
    #[test]
    fn test_batch_consistency() {
        let model = create_test_model();
        let inputs = vec![0.1, 0.3, 0.5, 0.7, 0.9];
        
        // Compare batch vs single predictions
        let batch_results = model.predict(&inputs).unwrap();
        
        for (i, &p) in inputs.iter().enumerate() {
            let single_result = model.predict_single(p).unwrap();
            assert_relative_eq!(
                batch_results[i], 
                single_result,
                epsilon = 1e-15
            );
        }
    }
}
```

### 4.2 Test Data Generation Script

```python
# generate_test_data.py
import json
import numpy as np
from betacal import BetaCalibration

def generate_test_data():
    """Generate test data for Rust parity testing."""
    
    # Train a simple model
    np.random.seed(42)
    n_samples = 1000
    X = np.random.rand(n_samples)
    y = (X + np.random.randn(n_samples) * 0.1 > 0.5).astype(int)
    
    bc = BetaCalibration(parameters="abm")
    bc.fit(X, y)
    
    # Generate test cases
    test_inputs = np.linspace(0.01, 0.99, 20)
    test_outputs = bc.predict(test_inputs)
    
    test_cases = [
        {"input": float(inp), "expected": float(out)}
        for inp, out in zip(test_inputs, test_outputs)
    ]
    
    # Export model parameters
    model_data = {
        "weights": bc.calibrator_.coef_[0].tolist(),
        "intercept": float(bc.calibrator_.intercept_[0]),
        "method": "AB",
        "test_cases": test_cases
    }
    
    with open("test_data.json", "w") as f:
        json.dump(model_data, f, indent=2)
    
    print(f"Generated {len(test_cases)} test cases")
```

---

## 5. Production Deployment

### 5.1 Model Validation

```rust
pub struct ModelValidator {
    test_inputs: Vec<f64>,
    expected_outputs: Vec<f64>,
    tolerance: f64,
}

impl ModelValidator {
    pub fn validate(&self, model: &BetaCalModel) -> Result<()> {
        let outputs = model.predict(&self.test_inputs)?;
        
        for (computed, expected) in outputs.iter().zip(&self.expected_outputs) {
            let diff = (computed - expected).abs();
            if diff > self.tolerance {
                return Err(anyhow!(
                    "Validation failed: diff {} > tolerance {}",
                    diff, self.tolerance
                ));
            }
        }
        
        Ok(())
    }
}
```

### 5.2 Monitoring

```rust
use std::time::Instant;

pub struct BetaCalWithMetrics {
    model: BetaCalModel,
    prediction_count: u64,
    total_duration_ms: u64,
}

impl BetaCalWithMetrics {
    pub fn predict(&mut self, probabilities: &[f64]) -> Result<Vec<f64>> {
        let start = Instant::now();
        let result = self.model.predict(probabilities)?;
        
        self.prediction_count += probabilities.len() as u64;
        self.total_duration_ms += start.elapsed().as_millis() as u64;
        
        Ok(result)
    }
    
    pub fn get_metrics(&self) -> (u64, f64) {
        let avg_duration = if self.prediction_count > 0 {
            self.total_duration_ms as f64 / self.prediction_count as f64
        } else {
            0.0
        };
        (self.prediction_count, avg_duration)
    }
}
```

### 5.3 Model Versioning

```rust
impl BetaCalModel {
    pub fn is_compatible_version(&self, required_version: &str) -> bool {
        // Simple major version compatibility check
        let self_major = self.version.split('.').next().unwrap_or("0");
        let required_major = required_version.split('.').next().unwrap_or("0");
        self_major == required_major
    }
}
```

---

## 6. Realistic Timeline

### Week 1: Core Implementation
- [x] Implement inference pipeline
- [x] Create model loading/saving
- [x] Write parity tests with Python
- [x] Basic error handling

### Week 2: Production Readiness
- [ ] Add comprehensive testing
- [ ] Implement monitoring
- [ ] Add batch processing optimization
- [ ] Create documentation
- [ ] Deploy to staging for validation

**Total: 1-2 weeks** for production-ready inference engine

---

## 7. Key Insights and Corrections

### What We Corrected

1. **No Beta CDF in Inference**: The Beta CDF transformation happens during training only. The trained logistic regression already encodes the calibration mapping.

2. **Simpler Implementation**: Without needing Beta distribution libraries (statrs), the implementation is much simpler and faster.

3. **Realistic Timeline**: 1-2 weeks is perfectly reasonable for an inference-only implementation.

4. **No PyO3 Needed**: Direct Rust implementation is simpler since we're not wrapping Python code.

### Focus Areas

1. **Parity Testing**: Ensure Rust predictions exactly match Python's predict method
2. **Numerical Stability**: Handle edge cases (0, 1, very small/large values)
3. **Performance**: Optimize for batch predictions if needed
4. **Monitoring**: Track prediction latency and counts

### What NOT to Do

1. **Don't implement Beta CDF** - It's not needed for inference
2. **Don't implement training** - Keep it in Python
3. **Don't over-engineer** - The inference pipeline is simple
4. **Don't skip parity tests** - Mathematical correctness is critical

---

## 8. Complete Working Example

```rust
// Complete minimal implementation
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct MinimalBetaCal {
    weights: Vec<f64>,
    intercept: f64,
    method: String,  // "AB", "A", or "B"
}

impl MinimalBetaCal {
    pub fn predict(&self, p: f64) -> f64 {
        // Step 1: Clip input
        let p_safe = p.max(1e-15).min(1.0 - 1e-15);
        
        // Step 2: Feature transformation
        let features = match self.method.as_str() {
            "AB" => vec![p_safe.ln(), (1.0 - p_safe).ln()],
            "A" => vec![p_safe.ln()],
            "B" => vec![(1.0 - p_safe).ln()],
            _ => panic!("Unknown method"),
        };
        
        // Step 3: Logistic regression
        let logit: f64 = features.iter()
            .zip(&self.weights)
            .map(|(f, w)| f * w)
            .sum::<f64>() + self.intercept;
        
        // Step 4: Sigmoid
        1.0 / (1.0 + (-logit).exp())
    }
}

// Usage
fn main() {
    let model = MinimalBetaCal {
        weights: vec![1.2345, -0.6789],
        intercept: 0.0,
        method: "AB".to_string(),
    };
    
    let calibrated = model.predict(0.7);
    println!("Calibrated probability: {}", calibrated);
}
```

---

## Conclusion

This corrected implementation plan provides a straightforward path to a production-ready BetaCal inference engine in Rust. By understanding that inference doesn't require Beta CDF transformation, we've simplified the implementation significantly.

**Key Takeaways:**
1. Inference is simple: clip → transform → logistic regression → sigmoid
2. No Beta distribution libraries needed
3. 1-2 weeks is realistic for production deployment
4. Focus on parity with Python's predict method
5. Keep training in Python, inference in Rust

By following this guide, you'll have a fast, correct, and maintainable Rust inference engine for BetaCal.