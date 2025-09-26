//! # BetaCal Inference Engine
//!
//! A high-performance Rust library for inference with BetaCal probability calibration models.
//! This library provides fast, thread-safe inference from models trained with the Python BetaCal package.
//!
//! ## Quick Start
//!
//! ```rust
//! use betacal_inference::{BetaCalModel, CalibrationMethod};
//!
//! // Create model manually
//! let model = BetaCalModel::new(vec![1.0, -0.5], 0.0, CalibrationMethod::AB);
//!
//! // Make predictions
//! let uncalibrated = vec![0.1, 0.5, 0.9];
//! let calibrated = model.predict(&uncalibrated)?;
//! println!("Calibrated: {:?}", calibrated);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Features
//!
//! - **Fast**: Pure Rust implementation with minimal overhead
//! - **Thread-safe**: Models can be shared across threads
//! - **Memory efficient**: Only stores essential parameters
//! - **Numerically stable**: Handles edge cases gracefully

use serde::{Deserialize, Serialize};
use std::fs;
use thiserror::Error;

/// Errors that can occur during inference
#[derive(Error, Debug)]
pub enum InferenceError {
    #[error("Empty input provided")]
    EmptyInput,
    #[error("Invalid probability value: {0} (must be in [0, 1])")]
    InvalidProbability(f64),
    #[error("Model serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Invalid model: {0}")]
    InvalidModel(String),
}

/// Calibration method used by the model
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum CalibrationMethod {
    /// Both log(p) and log(1-p) features
    #[serde(rename = "AB")]
    AB,
    /// Only log(p) feature
    #[serde(rename = "A")]
    A,
    /// Only log(1-p) feature
    #[serde(rename = "B")]
    B,
}

/// BetaCal model for inference
///
/// This struct contains the minimal parameters needed for fast inference
/// from a model trained with the Python BetaCal package.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BetaCalModel {
    /// Logistic regression weights
    pub weights: Vec<f64>,
    /// Logistic regression intercept
    pub intercept: f64,
    /// Calibration method
    pub method: CalibrationMethod,
    /// Model version
    pub version: String,
    /// Training score (optional)
    pub training_score: Option<f64>,
}

impl BetaCalModel {
    /// Create a new BetaCal model
    ///
    /// # Arguments
    ///
    /// * `weights` - Logistic regression weights
    /// * `intercept` - Logistic regression intercept
    /// * `method` - Calibration method
    ///
    /// # Example
    ///
    /// ```rust
    /// use betacal_inference::{BetaCalModel, CalibrationMethod};
    ///
    /// let model = BetaCalModel::new(
    ///     vec![1.2, -0.8],
    ///     0.0,
    ///     CalibrationMethod::AB
    /// );
    /// ```
    pub fn new(weights: Vec<f64>, intercept: f64, method: CalibrationMethod) -> Self {
        Self {
            weights,
            intercept,
            method,
            version: "1.0.0".to_string(),
            training_score: None,
        }
    }

    /// Predict calibrated probabilities for multiple inputs
    ///
    /// # Arguments
    ///
    /// * `probabilities` - Slice of uncalibrated probabilities in [0, 1]
    ///
    /// # Returns
    ///
    /// Vector of calibrated probabilities
    ///
    /// # Example
    ///
    /// ```rust
    /// use betacal_inference::{BetaCalModel, CalibrationMethod};
    ///
    /// let model = BetaCalModel::new(vec![1.0], 0.0, CalibrationMethod::A);
    /// let result = model.predict(&[0.1, 0.5, 0.9]).unwrap();
    /// assert_eq!(result.len(), 3);
    /// ```
    pub fn predict(&self, probabilities: &[f64]) -> Result<Vec<f64>, InferenceError> {
        if probabilities.is_empty() {
            return Err(InferenceError::EmptyInput);
        }

        let mut calibrated = Vec::with_capacity(probabilities.len());

        for &p in probabilities {
            // Validate input
            if !p.is_finite() || p < 0.0 || p > 1.0 {
                return Err(InferenceError::InvalidProbability(p));
            }

            // Step 1: Numerical safety - clip to avoid log(0)
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

    /// Predict calibrated probability for a single input
    ///
    /// # Arguments
    ///
    /// * `probability` - Uncalibrated probability in [0, 1]
    ///
    /// # Returns
    ///
    /// Calibrated probability
    ///
    /// # Example
    ///
    /// ```rust
    /// use betacal_inference::{BetaCalModel, CalibrationMethod};
    ///
    /// let model = BetaCalModel::new(vec![1.0], 0.0, CalibrationMethod::A);
    /// let result = model.predict_single(0.7).unwrap();
    /// assert!(result > 0.0 && result < 1.0);
    /// ```
    pub fn predict_single(&self, probability: f64) -> Result<f64, InferenceError> {
        let result = self.predict(&[probability])?;
        Ok(result[0])
    }

    /// Load model from JSON file
    ///
    /// # Arguments
    ///
    /// * `path` - Path to JSON file containing model parameters
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use betacal_inference::BetaCalModel;
    ///
    /// let model = BetaCalModel::load_from_json("model.json").unwrap();
    /// ```
    pub fn load_from_json(path: &str) -> Result<Self, InferenceError> {
        let content = fs::read_to_string(path)?;
        let model: BetaCalModel = serde_json::from_str(&content)?;
        model.validate()?;
        Ok(model)
    }

    /// Load model from JSON string
    ///
    /// # Arguments
    ///
    /// * `json` - JSON string containing model parameters
    pub fn load_from_json_str(json: &str) -> Result<Self, InferenceError> {
        let model: BetaCalModel = serde_json::from_str(json)?;
        model.validate()?;
        Ok(model)
    }

    /// Save model to JSON file
    ///
    /// # Arguments
    ///
    /// * `path` - Path where to save the model
    pub fn save_to_json(&self, path: &str) -> Result<(), InferenceError> {
        let content = serde_json::to_string_pretty(self)?;
        fs::write(path, content)?;
        Ok(())
    }

    /// Validate model parameters
    fn validate(&self) -> Result<(), InferenceError> {
        let expected_features = match self.method {
            CalibrationMethod::AB => 2,
            CalibrationMethod::A | CalibrationMethod::B => 1,
        };

        if self.weights.len() != expected_features {
            return Err(InferenceError::InvalidModel(format!(
                "Method {:?} expects {} weights, got {}",
                self.method,
                expected_features,
                self.weights.len()
            )));
        }

        for &w in &self.weights {
            if !w.is_finite() {
                return Err(InferenceError::InvalidModel("Non-finite weight".to_string()));
            }
        }

        if !self.intercept.is_finite() {
            return Err(InferenceError::InvalidModel("Non-finite intercept".to_string()));
        }

        Ok(())
    }

    /// Create feature vector based on calibration method
    /// This matches the exact transformation from Python BetaCal:
    /// x = np.hstack((df, 1. - df))
    /// x = np.log(x)  
    /// x[:, 1] *= -1  # Negate the second feature
    fn create_features(&self, p: f64) -> Vec<f64> {
        match self.method {
            CalibrationMethod::AB => vec![p.ln(), -(1.0 - p).ln()],  // [log(p), -log(1-p)]
            CalibrationMethod::A => vec![p.ln()],                     // [log(p)]
            CalibrationMethod::B => vec![-(1.0 - p).ln()],            // [-log(1-p)]
        }
    }

    /// Compute logit value from features
    fn compute_logit(&self, features: &[f64]) -> f64 {
        // Dot product of features and weights
        let mut logit = features
            .iter()
            .zip(&self.weights)
            .map(|(f, w)| f * w)
            .sum::<f64>();

        // Add intercept
        logit += self.intercept;

        logit
    }
}

/// Clip value to range [min, max]
#[inline]
fn clip(value: f64, min: f64, max: f64) -> f64 {
    value.max(min).min(max)
}

/// Numerically stable sigmoid function
#[inline]
fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        let exp_neg_x = (-x).exp();
        1.0 / (1.0 + exp_neg_x)
    } else {
        let exp_x = x.exp();
        exp_x / (1.0 + exp_x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn create_test_model() -> BetaCalModel {
        BetaCalModel::new(vec![1.2345, -0.6789], 0.0, CalibrationMethod::AB)
    }

    #[test]
    fn test_basic_prediction() {
        let model = create_test_model();
        let result = model.predict_single(0.5).unwrap();
        assert!(result > 0.0 && result < 1.0);
    }

    #[test]
    fn test_batch_prediction() {
        let model = create_test_model();
        let inputs = vec![0.1, 0.3, 0.5, 0.7, 0.9];
        let results = model.predict(&inputs).unwrap();
        assert_eq!(results.len(), 5);
        
        for result in results {
            assert!(result > 0.0 && result < 1.0);
        }
    }

    #[test]
    fn test_batch_vs_single_consistency() {
        let model = create_test_model();
        let inputs = vec![0.1, 0.3, 0.5, 0.7, 0.9];
        
        let batch_results = model.predict(&inputs).unwrap();
        
        for (i, &p) in inputs.iter().enumerate() {
            let single_result = model.predict_single(p).unwrap();
            assert_relative_eq!(batch_results[i], single_result, epsilon = 1e-15);
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
    fn test_invalid_inputs() {
        let model = create_test_model();
        
        // Test invalid probabilities
        assert!(model.predict_single(-0.1).is_err());
        assert!(model.predict_single(1.1).is_err());
        assert!(model.predict_single(f64::NAN).is_err());
        assert!(model.predict_single(f64::INFINITY).is_err());
        
        // Test empty input
        assert!(model.predict(&[]).is_err());
    }

    #[test]
    fn test_serialization() {
        let model = create_test_model();
        let json = serde_json::to_string(&model).unwrap();
        let deserialized: BetaCalModel = serde_json::from_str(&json).unwrap();
        
        assert_eq!(model.weights, deserialized.weights);
        assert_eq!(model.intercept, deserialized.intercept);
        assert_eq!(model.method, deserialized.method);
    }

    #[test]
    fn test_different_methods() {
        let test_cases = vec![
            (CalibrationMethod::A, vec![1.0]),
            (CalibrationMethod::B, vec![-1.0]),
            (CalibrationMethod::AB, vec![1.0, -0.5]),
        ];

        for (method, weights) in test_cases {
            let model = BetaCalModel::new(weights, 0.0, method);
            let result = model.predict_single(0.5).unwrap();
            assert!(result > 0.0 && result < 1.0);
        }
    }
}
