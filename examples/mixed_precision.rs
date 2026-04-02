use turboquant::{OutlierSplitPlan, TurboQuantMixedMse, TurboQuantMixedProd};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let vector = vec![0.25, -0.7, 1.1, 0.4, -0.2, 0.9, -1.0, 0.3];
    let query = vec![0.1, -0.3, 0.5, -0.2, 0.7, -0.4, 0.2, 0.6];

    // 4 outlier channels at 3 bits + 4 regular channels at 2 bits => 2.5 bits average.
    let plan = OutlierSplitPlan::new(vector.len(), vec![0, 2, 4, 6], 3, 2)?;

    let mse = TurboQuantMixedMse::new(plan.clone(), 7)?;
    let mse_code = mse.quantize(&vector)?;
    let mse_decoded = mse.dequantize(&mse_code)?;

    let prod = TurboQuantMixedProd::new(plan, 7)?;
    let prod_code = prod.quantize(&vector)?;
    let prod_ip = prod.estimate_inner_product(&prod_code, &query)?;

    println!("effective bits: {:.2}", mse.effective_bit_width());
    println!("mixed mse decoded: {mse_decoded:?}");
    println!("mixed mse code logical bytes: {}", mse_code.storage_bytes());
    println!(
        "mixed prod code logical bytes: {}",
        prod_code.storage_bytes()
    );
    println!("mixed approx inner product: {prod_ip:.6}");
    Ok(())
}
