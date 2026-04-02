use turboquant::{TurboQuantMse, TurboQuantProd};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let vector = vec![0.25, -0.7, 1.1, 0.4, -0.2, 0.9, -1.0, 0.3];
    let batch = vec![
        vector.clone(),
        vec![-0.6, 0.3, 0.9, -0.4, 0.2, -1.1, 0.7, 0.5],
    ];
    let query = vec![0.1, -0.3, 0.5, -0.2, 0.7, -0.4, 0.2, 0.6];

    let mse = TurboQuantMse::new(vector.len(), 2, 7)?;
    let mse_code = mse.quantize(&vector)?;
    let mse_decoded = mse.dequantize(&mse_code)?;
    let batch_codes = mse.quantize_batch(&batch)?;

    let prod = TurboQuantProd::new(vector.len(), 3, 7)?;
    let prod_code = prod.quantize(&vector)?;
    let prod_ip = prod.estimate_inner_product(&prod_code, &query)?;

    println!("rotation backend: {:?}", mse.rotation_backend());
    println!("mse decoded: {mse_decoded:?}");
    println!("batch encoded vectors: {}", batch_codes.len());
    println!("mse code logical bytes: {}", mse_code.storage_bytes());
    println!("prod code logical bytes: {}", prod_code.storage_bytes());
    println!("approx inner product: {prod_ip:.6}");
    Ok(())
}
