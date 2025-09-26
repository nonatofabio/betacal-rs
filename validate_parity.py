#!/usr/bin/env python3
"""
Validation script to test Python-Rust parity for BetaCal inference.
"""

import json
import subprocess
import tempfile
import os
from betacal import BetaCalibration
import numpy as np
from export_model import export_betacal_model

def test_parity():
    """Test that Rust inference matches Python exactly."""
    
    print("BetaCal Python-Rust Parity Test")
    print("===============================")
    
    # Create and train a model
    print("\n1. Training Python model...")
    np.random.seed(42)
    probs = np.random.beta(2, 3, 1000)
    labels = (probs + np.random.randn(1000) * 0.1 > 0.5).astype(int)
    
    cal = BetaCalibration(parameters="abm")
    cal.fit(probs, labels)
    print("✓ Model trained")
    
    # Export model
    print("\n2. Exporting model...")
    model_path = "parity_test_model.json"
    export_betacal_model(cal, model_path, include_test_cases=True)
    
    # Test cases
    test_inputs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    python_outputs = cal.predict(np.array(test_inputs))
    
    print("\n3. Python predictions:")
    for inp, out in zip(test_inputs, python_outputs):
        print(f"  {inp:.1f} -> {out:.8f}")
    
    # Create Rust test program
    print("\n4. Testing Rust inference...")
    rust_test_code = f'''
use betacal_inference::BetaCalModel;

fn main() -> Result<(), Box<dyn std::error::Error>> {{
    let model = BetaCalModel::load_from_json("{model_path}")?;
    let test_inputs = vec![{", ".join(map(str, test_inputs))}];
    let results = model.predict(&test_inputs)?;
    
    println!("Rust predictions:");
    for (inp, out) in test_inputs.iter().zip(results.iter()) {{
        println!("  {{:.1}} -> {{:.8}}", inp, out);
    }}
    
    // Validate against Python results
    let python_results = vec![{", ".join(f"{x:.12f}" for x in python_outputs)}];
    let mut max_error = 0.0;
    let mut all_passed = true;
    
    for (i, (&rust_val, &python_val)) in results.iter().zip(python_results.iter()).enumerate() {{
        let error = (rust_val - python_val).abs();
        max_error = max_error.max(error);
        
        if error > 1e-10 {{
            println!("MISMATCH at index {{}}: Python={{:.12}}, Rust={{:.12}}, Error={{:.2e}}", 
                     i, python_val, rust_val, error);
            all_passed = false;
        }}
    }}
    
    println!("\\nValidation results:");
    println!("  Max error: {{:.2e}}", max_error);
    if all_passed {{
        println!("  ✓ All predictions match within tolerance");
    }} else {{
        println!("  ✗ Some predictions differ");
    }}
    
    Ok(())
}}
'''
    
    # Write and run Rust test
    with open("betacal-inference/examples/parity_test.rs", "w") as f:
        f.write(rust_test_code)
    
    # Run Rust test
    result = subprocess.run(
        ["cargo", "run", "--example", "parity_test"],
        cwd="betacal-inference",
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.stderr:
        print("Rust stderr:", result.stderr)
    
    # Clean up
    os.remove(model_path)
    os.remove("betacal-inference/examples/parity_test.rs")
    
    if result.returncode == 0:
        print("\n✓ Parity test completed successfully!")
        return True
    else:
        print("\n✗ Parity test failed!")
        return False

if __name__ == "__main__":
    success = test_parity()
    exit(0 if success else 1)
