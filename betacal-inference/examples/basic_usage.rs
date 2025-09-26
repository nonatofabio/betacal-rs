use betacal_inference::{BetaCalModel, CalibrationMethod, InferenceError};

fn main() -> Result<(), InferenceError> {
    println!("BetaCal Inference Engine - Basic Usage Example");
    println!("==============================================");

    // Example 1: Create model manually
    println!("\n1. Creating model manually:");
    let model = BetaCalModel::new(
        vec![1.2345, -0.6789], // weights from Python training
        0.0,                   // intercept
        CalibrationMethod::AB, // method
    );

    // Make predictions
    let test_probs = vec![0.1, 0.3, 0.5, 0.7, 0.9];
    let calibrated = model.predict(&test_probs)?;

    println!("Original:   {:?}", test_probs);
    println!("Calibrated: {:?}", calibrated);

    // Example 2: Single prediction
    println!("\n2. Single prediction:");
    let single_prob = 0.75;
    let single_calibrated = model.predict_single(single_prob)?;
    println!("Original: {:.3} -> Calibrated: {:.3}", single_prob, single_calibrated);

    // Example 3: Different calibration methods
    println!("\n3. Different calibration methods:");
    
    let methods = vec![
        (CalibrationMethod::A, vec![1.0], "Method A (log(p) only)"),
        (CalibrationMethod::B, vec![-1.0], "Method B (log(1-p) only)"),
        (CalibrationMethod::AB, vec![1.0, -0.5], "Method AB (both features)"),
    ];

    for (method, weights, description) in methods {
        let model = BetaCalModel::new(weights, 0.0, method);
        let result = model.predict_single(0.6)?;
        println!("{}: 0.6 -> {:.3}", description, result);
    }

    // Example 4: Save and load model
    println!("\n4. Save and load model:");
    let model = BetaCalModel::new(vec![0.8, -1.2], 0.1, CalibrationMethod::AB);
    
    // Save to JSON
    model.save_to_json("example_model.json")?;
    println!("Model saved to example_model.json");
    
    // Load from JSON
    let loaded_model = BetaCalModel::load_from_json("example_model.json")?;
    let original_result = model.predict_single(0.4)?;
    let loaded_result = loaded_model.predict_single(0.4)?;
    
    println!("Original model result: {:.6}", original_result);
    println!("Loaded model result:   {:.6}", loaded_result);
    println!("Results match: {}", (original_result - loaded_result).abs() < 1e-10);

    // Clean up
    std::fs::remove_file("example_model.json").ok();

    println!("\n✓ All examples completed successfully!");
    Ok(())
}
