use betacal_inference::{BetaCalModel, InferenceError};
use std::sync::Arc;
use std::thread;
use std::time::Instant;

/// Production-ready calibration service
pub struct CalibrationService {
    model: Arc<BetaCalModel>,
    prediction_count: std::sync::atomic::AtomicU64,
}

impl CalibrationService {
    pub fn new(model_path: &str) -> Result<Self, InferenceError> {
        let model = BetaCalModel::load_from_json(model_path)?;
        Ok(Self {
            model: Arc::new(model),
            prediction_count: std::sync::atomic::AtomicU64::new(0),
        })
    }

    pub fn calibrate_batch(&self, probabilities: &[f64]) -> Result<Vec<f64>, InferenceError> {
        let result = self.model.predict(probabilities)?;
        self.prediction_count.fetch_add(
            probabilities.len() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        Ok(result)
    }

    pub fn calibrate_single(&self, probability: f64) -> Result<f64, InferenceError> {
        let result = self.model.predict_single(probability)?;
        self.prediction_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(result)
    }

    pub fn get_prediction_count(&self) -> u64 {
        self.prediction_count.load(std::sync::atomic::Ordering::Relaxed)
    }
}

fn main() -> Result<(), InferenceError> {
    println!("BetaCal Inference Engine - Production Usage Example");
    println!("===================================================");

    // Create a test model for demonstration
    let model = BetaCalModel::new(vec![1.5, -0.8], 0.1, betacal_inference::CalibrationMethod::AB);
    model.save_to_json("production_model.json")?;

    // Example 1: Thread-safe service
    println!("\n1. Thread-safe calibration service:");
    let service = Arc::new(CalibrationService::new("production_model.json")?);

    // Spawn multiple threads
    let mut handles = vec![];
    for thread_id in 0..4 {
        let service_clone = Arc::clone(&service);
        let handle = thread::spawn(move || {
            let test_data: Vec<f64> = (0..100)
                .map(|i| (i as f64 + thread_id as f64 * 100.0) / 1000.0)
                .collect();
            
            service_clone.calibrate_batch(&test_data).unwrap()
        });
        handles.push(handle);
    }

    // Wait for all threads to complete
    let mut all_results = vec![];
    for handle in handles {
        let results = handle.join().unwrap();
        all_results.extend(results);
    }

    println!("Processed {} predictions across 4 threads", all_results.len());
    println!("Total service predictions: {}", service.get_prediction_count());

    // Example 2: Performance benchmarking
    println!("\n2. Performance benchmarking:");
    let test_data: Vec<f64> = (0..10000).map(|i| i as f64 / 10000.0).collect();
    
    let start = Instant::now();
    let results = service.calibrate_batch(&test_data)?;
    let duration = start.elapsed();
    
    println!("Batch prediction of {} samples:", test_data.len());
    println!("  Duration: {:?}", duration);
    println!("  Throughput: {:.0} predictions/sec", 
             test_data.len() as f64 / duration.as_secs_f64());
    println!("  Latency: {:.2} μs/prediction", 
             duration.as_micros() as f64 / test_data.len() as f64);

    // Example 3: Model validation
    println!("\n3. Model validation:");
    let validation_cases = vec![
        (0.0, "boundary case"),
        (1.0, "boundary case"),
        (0.5, "midpoint"),
        (1e-10, "near zero"),
        (1.0 - 1e-10, "near one"),
    ];

    for (input, description) in validation_cases {
        match service.calibrate_single(input) {
            Ok(output) => {
                println!("  {} ({:.2e}): {:.6} ✓", description, input, output);
            }
            Err(e) => {
                println!("  {} ({:.2e}): Error - {} ✗", description, input, e);
            }
        }
    }

    // Example 4: Error handling
    println!("\n4. Error handling:");
    let invalid_inputs = vec![-0.1, 1.1, f64::NAN, f64::INFINITY];
    
    for &input in &invalid_inputs {
        match service.calibrate_single(input) {
            Ok(_) => println!("  {}: Unexpected success ✗", input),
            Err(e) => println!("  {}: {} ✓", input, e),
        }
    }

    // Clean up
    std::fs::remove_file("production_model.json").ok();

    println!("\n✓ Production example completed successfully!");
    Ok(())
}
