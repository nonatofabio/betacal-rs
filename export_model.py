#!/usr/bin/env python3
"""
BetaCal Model Export Script

This script exports trained BetaCal models to JSON format for use with the Rust inference engine.
It extracts the essential parameters needed for fast inference without requiring the full Python environment.

Usage:
    python export_model.py

Or import and use programmatically:
    from export_model import export_betacal_model
    export_betacal_model(trained_calibrator, "my_model.json")
"""

import json
import sys
from datetime import datetime
import numpy as np


def export_betacal_model(calibrator, output_path, include_test_cases=True):
    """
    Export a trained BetaCal model for Rust inference.
    
    Args:
        calibrator: Trained BetaCalibration instance
        output_path: Path to save the JSON model file
        include_test_cases: Whether to generate test cases for validation
    
    Returns:
        dict: The exported model data
    
    Raises:
        ValueError: If model is not fitted or has invalid parameters
        AttributeError: If model structure is unexpected
    """
    
    # Validate input
    if not hasattr(calibrator, 'calibrator_'):
        raise ValueError("Calibrator not fitted - call fit() first")
    
    internal_cal = calibrator.calibrator_
    
    # Extract logistic regression parameters
    if not hasattr(internal_cal, 'lr_') or internal_cal.lr_ is None:
        raise ValueError("Model not fitted or missing logistic regression component")
    
    # Get weights and intercept from sklearn LogisticRegression
    lr = internal_cal.lr_
    weights = lr.coef_[0].tolist()
    intercept = float(lr.intercept_[0])
    
    # Validate parameters are finite
    if not all(np.isfinite(weights)) or not np.isfinite(intercept):
        raise ValueError("Model contains non-finite parameters")
    
    # Determine method based on calibrator parameters and weights
    method = determine_method(calibrator.parameters, len(weights))
    
    # Create export structure
    model_data = {
        "weights": weights,
        "intercept": intercept,
        "method": method,
        "version": "1.0.0",
        "training_score": getattr(internal_cal, 'score_', None),
        "metadata": {
            "training_date": datetime.now().isoformat(),
            "python_version": sys.version.split()[0],
            "original_parameters": calibrator.parameters,
            "n_features": len(weights),
            "export_script_version": "1.0.0"
        }
    }
    
    # Add test cases for validation if requested
    if include_test_cases:
        test_cases = generate_test_cases(calibrator)
        model_data["test_cases"] = test_cases
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(model_data, f, indent=2)
    
    # Print summary
    print(f"✓ Model exported to {output_path}")
    print(f"  Method: {method}")
    print(f"  Features: {len(weights)}")
    print(f"  Weights: {weights}")
    print(f"  Intercept: {intercept:.6f}")
    if include_test_cases:
        print(f"  Test cases: {len(model_data['test_cases'])}")
    
    return model_data


def determine_method(parameters, n_weights):
    """
    Determine the Rust calibration method based on Python parameters and weight count.
    
    Args:
        parameters: Original BetaCal parameters string
        n_weights: Number of logistic regression weights
    
    Returns:
        str: Method identifier for Rust ("A", "B", or "AB")
    """
    
    # Map based on parameter string and weight count
    if parameters == "a" or (parameters in ["am"] and n_weights == 1):
        return "A"  # Only log(p) feature
    elif parameters == "b" or n_weights == 1:
        # Note: Pure "b" method is rare, usually combined with others
        return "B"  # Only log(1-p) feature  
    elif parameters in ["abm", "ab"] or n_weights == 2:
        return "AB"  # Both log(p) and log(1-p) features
    else:
        # Default to AB for safety
        print(f"Warning: Unknown parameter combination '{parameters}' with {n_weights} weights, defaulting to AB")
        return "AB"


def generate_test_cases(calibrator, n_cases=20):
    """
    Generate test cases for validating Rust implementation against Python.
    
    Args:
        calibrator: Trained BetaCalibration instance
        n_cases: Number of test cases to generate
    
    Returns:
        list: Test cases with input/expected output pairs
    """
    
    # Generate test inputs covering the probability range
    test_inputs = np.linspace(0.01, 0.99, n_cases)
    
    # Add some edge cases
    edge_cases = [1e-10, 1e-6, 0.001, 0.999, 1.0 - 1e-6, 1.0 - 1e-10]
    test_inputs = np.concatenate([test_inputs, edge_cases])
    
    # Get expected outputs from Python
    test_outputs = calibrator.predict(test_inputs)
    
    # Create test case structure
    test_cases = []
    for inp, out in zip(test_inputs, test_outputs):
        test_cases.append({
            "input": float(inp),
            "expected": float(out),
            "tolerance": 1e-12  # Very strict tolerance for exact parity
        })
    
    return test_cases


def validate_export(model_path):
    """
    Validate an exported model file.
    
    Args:
        model_path: Path to the exported JSON model
    
    Returns:
        bool: True if valid, False otherwise
    """
    
    try:
        with open(model_path, 'r') as f:
            model_data = json.load(f)
        
        # Check required fields
        required_fields = ["weights", "intercept", "method", "version"]
        for field in required_fields:
            if field not in model_data:
                print(f"✗ Missing required field: {field}")
                return False
        
        # Validate weights
        weights = model_data["weights"]
        if not isinstance(weights, list) or len(weights) == 0:
            print("✗ Invalid weights")
            return False
        
        if not all(isinstance(w, (int, float)) and np.isfinite(w) for w in weights):
            print("✗ Non-finite weights")
            return False
        
        # Validate intercept
        intercept = model_data["intercept"]
        if not isinstance(intercept, (int, float)) or not np.isfinite(intercept):
            print("✗ Invalid intercept")
            return False
        
        # Validate method
        method = model_data["method"]
        if method not in ["A", "B", "AB"]:
            print(f"✗ Invalid method: {method}")
            return False
        
        # Check method consistency with weights
        expected_weights = {"A": 1, "B": 1, "AB": 2}
        if len(weights) != expected_weights[method]:
            print(f"✗ Method {method} expects {expected_weights[method]} weights, got {len(weights)}")
            return False
        
        print(f"✓ Model validation passed: {model_path}")
        return True
        
    except Exception as e:
        print(f"✗ Validation error: {e}")
        return False


def create_example_model():
    """
    Create an example model for demonstration purposes.
    
    Returns:
        BetaCalibration: A trained calibrator
    """
    
    try:
        from betacal import BetaCalibration
    except ImportError:
        print("Error: betacal package not found. Install with: pip install betacal")
        return None
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Create realistic probability data
    probabilities = np.random.beta(2, 2, n_samples)  # Beta distribution for realistic probabilities
    
    # Create labels with some calibration bias
    true_probs = 1 / (1 + np.exp(-(probabilities - 0.5) * 4))  # Sigmoid transformation
    labels = np.random.binomial(1, true_probs, n_samples)
    
    # Train calibration model
    calibrator = BetaCalibration(parameters="abm")
    calibrator.fit(probabilities, labels)
    
    print("✓ Example model created and trained")
    print(f"  Training samples: {n_samples}")
    print(f"  Parameters: {calibrator.parameters}")
    
    return calibrator


def main():
    """Main function for command-line usage."""
    
    print("BetaCal Model Export Script")
    print("==========================")
    
    # Create example model
    print("\n1. Creating example model...")
    calibrator = create_example_model()
    
    if calibrator is None:
        return 1
    
    # Export model
    print("\n2. Exporting model...")
    try:
        model_data = export_betacal_model(calibrator, "example_model.json")
        
        # Validate export
        print("\n3. Validating export...")
        if validate_export("example_model.json"):
            print("\n✓ Export completed successfully!")
            print("\nNext steps:")
            print("1. Copy example_model.json to your Rust project")
            print("2. Use BetaCalModel::load_from_json(\"example_model.json\") in Rust")
            print("3. Run cargo test to validate parity")
        else:
            print("\n✗ Export validation failed")
            return 1
            
    except Exception as e:
        print(f"\n✗ Export failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
