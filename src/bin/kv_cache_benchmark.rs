use std::env;
use std::time::Instant;

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};
use turboquant::{
    KvQuantizerSpec, KvTensor, KvTensorShape, TurboQuantKvCache, TurboQuantKvCacheConfig,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    if args.iter().any(|arg| arg == "--help") {
        print_usage();
        return Ok(());
    }

    let mode = find_arg(&args, "--mode").unwrap_or("mse");
    let num_layers = parse_usize(&args, "--layers", 32);
    let batch_size = parse_usize(&args, "--batch", 1);
    let kv_heads = parse_usize(&args, "--heads", 8);
    let head_dim = parse_usize(&args, "--head-dim", 128);
    let prefill = parse_usize(&args, "--prefill", 8_192);
    let decode_tokens = parse_usize(&args, "--decode-tokens", 128);
    let residual_length = parse_usize(&args, "--residual-length", 128);
    let bits = parse_u8(&args, "--bits", 4);
    let seed = parse_u64(&args, "--seed", 7);
    let skip_layers = parse_usize_list(&args, "--skip-layers");

    let (key_spec, value_spec) = match mode {
        "mse" => (
            KvQuantizerSpec::Mse { bit_width: bits },
            KvQuantizerSpec::Mse { bit_width: bits },
        ),
        "prod_mse" => (
            KvQuantizerSpec::Prod { bit_width: bits },
            KvQuantizerSpec::Mse { bit_width: bits },
        ),
        "fast_mse" => (
            KvQuantizerSpec::FastMse { bit_width: bits },
            KvQuantizerSpec::FastMse { bit_width: bits },
        ),
        "fast_prod_mse" => (
            KvQuantizerSpec::FastProd { bit_width: bits },
            KvQuantizerSpec::FastMse { bit_width: bits },
        ),
        other => return Err(format!("unsupported --mode value: {other}").into()),
    };

    let mut cache = TurboQuantKvCache::new(TurboQuantKvCacheConfig {
        num_layers,
        batch_size,
        kv_heads,
        head_dim,
        residual_length,
        key_spec,
        value_spec,
        seed,
        skip_layers: skip_layers.clone(),
    })?;

    let mut rng = StdRng::seed_from_u64(seed ^ 0xDEAD_BEEF_CAFE_BABE);
    let prefill_shape = KvTensorShape::new(batch_size, kv_heads, prefill, head_dim)?;
    let prefill_start = Instant::now();
    for layer_index in 0..num_layers {
        let key_tensor = random_tensor(prefill_shape, &mut rng)?;
        let value_tensor = random_tensor(prefill_shape, &mut rng)?;
        cache.update(layer_index, &key_tensor, &value_tensor)?;
    }
    let prefill_seconds = prefill_start.elapsed().as_secs_f64();

    let step_shape = KvTensorShape::new(batch_size, kv_heads, 1, head_dim)?;
    let decode_start = Instant::now();
    for _ in 0..decode_tokens {
        for layer_index in 0..num_layers {
            let key_tensor = random_tensor(step_shape, &mut rng)?;
            let value_tensor = random_tensor(step_shape, &mut rng)?;
            cache.update(layer_index, &key_tensor, &value_tensor)?;
        }
    }
    let decode_seconds = decode_start.elapsed().as_secs_f64();

    let total_seq = prefill + decode_tokens;
    let full_precision_bytes =
        num_layers * 2 * batch_size * kv_heads * total_seq * head_dim * std::mem::size_of::<f64>();
    let quantized_bytes = cache.storage_bytes();
    let compression_ratio = if quantized_bytes == 0 {
        f64::INFINITY
    } else {
        full_precision_bytes as f64 / quantized_bytes as f64
    };

    println!(
        "mode,layers,batch,heads,head_dim,prefill,decode_tokens,residual_length,bits,skip_layers,full_precision_bytes,quantized_bytes,compression_ratio,prefill_seconds,decode_seconds,total_seconds"
    );
    println!(
        "{mode},{num_layers},{batch_size},{kv_heads},{head_dim},{prefill},{decode_tokens},{residual_length},{bits},\"{}\",{full_precision_bytes},{quantized_bytes},{compression_ratio:.6},{prefill_seconds:.6},{decode_seconds:.6},{:.6}",
        join_indices(&skip_layers),
        prefill_seconds + decode_seconds
    );

    Ok(())
}

fn random_tensor(
    shape: KvTensorShape,
    rng: &mut StdRng,
) -> Result<KvTensor, Box<dyn std::error::Error>> {
    let values = (0..shape.element_count())
        .map(|_| StandardNormal.sample(rng))
        .collect::<Vec<f64>>();
    Ok(KvTensor::new(shape, values)?)
}

fn parse_usize(args: &[String], flag: &str, default: usize) -> usize {
    find_arg(args, flag)
        .map(|value| value.parse::<usize>().expect("invalid usize flag"))
        .unwrap_or(default)
}

fn parse_u8(args: &[String], flag: &str, default: u8) -> u8 {
    find_arg(args, flag)
        .map(|value| value.parse::<u8>().expect("invalid u8 flag"))
        .unwrap_or(default)
}

fn parse_u64(args: &[String], flag: &str, default: u64) -> u64 {
    find_arg(args, flag)
        .map(|value| value.parse::<u64>().expect("invalid u64 flag"))
        .unwrap_or(default)
}

fn parse_usize_list(args: &[String], flag: &str) -> Vec<usize> {
    find_arg(args, flag)
        .map(|value| {
            value
                .split([',', ' ', '\t'])
                .filter(|item| !item.is_empty())
                .map(|item| item.parse::<usize>().expect("invalid usize list"))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default()
}

fn find_arg<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.windows(2)
        .find(|window| window[0] == flag)
        .map(|window| window[1].as_str())
}

fn join_indices(indices: &[usize]) -> String {
    indices
        .iter()
        .map(|index| index.to_string())
        .collect::<Vec<_>>()
        .join(" ")
}

fn print_usage() {
    eprintln!("usage:");
    eprintln!(
        "  kv_cache_benchmark --mode mse --layers 32 --batch 1 --heads 8 --head-dim 128 --prefill 8192 --decode-tokens 128 --residual-length 128 --bits 4 --skip-layers 0,31 --seed 7"
    );
    eprintln!(
        "  kv_cache_benchmark --mode prod_mse --layers 32 --batch 1 --heads 8 --head-dim 128 --prefill 8192 --decode-tokens 128 --residual-length 128 --bits 4 --seed 7"
    );
    eprintln!(
        "  kv_cache_benchmark --mode fast_prod_mse --layers 32 --batch 1 --heads 8 --head-dim 128 --prefill 8192 --decode-tokens 128 --residual-length 128 --bits 4 --seed 7"
    );
}
