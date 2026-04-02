use std::env;

use turboquant::experiment::{
    evaluate_mse, evaluate_mse_bounds, evaluate_prod, evaluate_prod_bounds, evaluate_recall,
    evaluate_recall_curve,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    if args.is_empty() {
        print_usage();
        return Ok(());
    }

    let mode = args[0].as_str();
    match mode {
        "mse" => run_mse(&args[1..])?,
        "prod" => run_prod(&args[1..])?,
        "recall" => run_recall(&args[1..])?,
        "recall_curve" => run_recall_curve(&args[1..])?,
        "bounds" => run_bounds(&args[1..])?,
        _ => print_usage(),
    }

    Ok(())
}

fn run_mse(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let dimension = parse_usize(args, "--dimension", 1536);
    let samples = parse_usize(args, "--samples", 1000);
    let seed = parse_u64(args, "--seed", 7);
    let bits = parse_bits(args);

    println!("mode,dimension,bits,samples,seed,average_mse");
    for bit_width in bits {
        let metrics = evaluate_mse(dimension, bit_width, samples, seed)?;
        println!(
            "mse,{dimension},{bit_width},{samples},{seed},{}",
            metrics.average_mse
        );
    }
    Ok(())
}

fn run_prod(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let dimension = parse_usize(args, "--dimension", 1536);
    let samples = parse_usize(args, "--samples", 1000);
    let seed = parse_u64(args, "--seed", 7);
    let bits = parse_bits(args);

    println!("mode,dimension,bits,samples,seed,bias,variance,mse");
    for bit_width in bits {
        let metrics = evaluate_prod(dimension, bit_width, samples, seed)?;
        println!(
            "prod,{dimension},{bit_width},{samples},{seed},{},{},{}",
            metrics.bias, metrics.variance, metrics.mse
        );
    }
    Ok(())
}

fn run_recall(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let dimension = parse_usize(args, "--dimension", 256);
    let dataset_size = parse_usize(args, "--dataset-size", 10_000);
    let queries = parse_usize(args, "--queries", 100);
    let top_k = parse_usize(args, "--top-k", 10);
    let seed = parse_u64(args, "--seed", 7);
    let bits = parse_bits(args);

    println!("mode,dimension,bits,dataset_size,queries,top_k,seed,recall_at_k");
    for bit_width in bits {
        let metrics = evaluate_recall(dimension, bit_width, dataset_size, queries, top_k, seed)?;
        println!(
            "recall,{dimension},{bit_width},{dataset_size},{queries},{top_k},{seed},{}",
            metrics.recall_at_k
        );
    }
    Ok(())
}

fn run_recall_curve(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let dimension = parse_usize(args, "--dimension", 256);
    let dataset_size = parse_usize(args, "--dataset-size", 10_000);
    let queries = parse_usize(args, "--queries", 100);
    let seed = parse_u64(args, "--seed", 7);
    let bits = parse_bits(args);
    let ks = parse_usizes(args, "--ks", &[1, 2, 4, 8, 16, 32, 64]);

    println!(
        "mode,dimension,bits,dataset_size,queries,k,seed,recall_at_1,indexing_seconds,query_seconds"
    );
    for bit_width in bits {
        let metrics =
            evaluate_recall_curve(dimension, bit_width, dataset_size, queries, &ks, seed)?;
        for point in metrics.points {
            println!(
                "recall_curve,{dimension},{bit_width},{dataset_size},{queries},{},{seed},{},{},{}",
                point.k, point.recall_at_1, metrics.indexing_seconds, metrics.query_seconds
            );
        }
    }
    Ok(())
}

fn run_bounds(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let dimension = parse_usize(args, "--dimension", 1536);
    let samples = parse_usize(args, "--samples", 1000);
    let seed = parse_u64(args, "--seed", 7);
    let bits = parse_bits(args);
    let metric = find_arg(args, "--metric").unwrap_or("both");

    println!(
        "metric,dimension,bits,samples,seed,measured,lower_bound,upper_bound,small_bit_reference"
    );
    match metric {
        "mse" => {
            for bit_width in bits {
                let metrics = evaluate_mse_bounds(dimension, bit_width, samples, seed)?;
                println!(
                    "mse,{dimension},{bit_width},{samples},{seed},{},{},{},{}",
                    metrics.measured,
                    metrics.lower_bound,
                    metrics.upper_bound,
                    metrics
                        .small_bit_reference
                        .map(|value| value.to_string())
                        .unwrap_or_default()
                );
            }
        }
        "prod" => {
            for bit_width in bits {
                let metrics = evaluate_prod_bounds(dimension, bit_width, samples, seed)?;
                println!(
                    "prod,{dimension},{bit_width},{samples},{seed},{},{},{},{}",
                    metrics.measured,
                    metrics.lower_bound,
                    metrics.upper_bound,
                    metrics
                        .small_bit_reference
                        .map(|value| value.to_string())
                        .unwrap_or_default()
                );
            }
        }
        "both" => {
            for bit_width in &bits {
                let metrics = evaluate_mse_bounds(dimension, *bit_width, samples, seed)?;
                println!(
                    "mse,{dimension},{},{samples},{seed},{},{},{},{}",
                    bit_width,
                    metrics.measured,
                    metrics.lower_bound,
                    metrics.upper_bound,
                    metrics
                        .small_bit_reference
                        .map(|value| value.to_string())
                        .unwrap_or_default()
                );
            }
            for bit_width in &bits {
                let metrics = evaluate_prod_bounds(dimension, *bit_width, samples, seed)?;
                println!(
                    "prod,{dimension},{},{samples},{seed},{},{},{},{}",
                    bit_width,
                    metrics.measured,
                    metrics.lower_bound,
                    metrics.upper_bound,
                    metrics
                        .small_bit_reference
                        .map(|value| value.to_string())
                        .unwrap_or_default()
                );
            }
        }
        other => return Err(format!("unsupported --metric value: {other}").into()),
    }

    Ok(())
}

fn parse_bits(args: &[String]) -> Vec<u8> {
    find_arg(args, "--bits")
        .map(|value| {
            value
                .split([',', ' ', '\t'])
                .filter(|item| !item.is_empty())
                .map(|item| item.parse::<u8>().expect("invalid bit width"))
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|| vec![1, 2, 3, 4])
}

fn parse_usize(args: &[String], flag: &str, default: usize) -> usize {
    find_arg(args, flag)
        .map(|value| value.parse::<usize>().expect("invalid usize flag"))
        .unwrap_or(default)
}

fn parse_usizes(args: &[String], flag: &str, default: &[usize]) -> Vec<usize> {
    find_arg(args, flag)
        .map(|value| {
            value
                .split([',', ' ', '\t'])
                .filter(|item| !item.is_empty())
                .map(|item| item.parse::<usize>().expect("invalid usize list"))
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|| default.to_vec())
}

fn parse_u64(args: &[String], flag: &str, default: u64) -> u64 {
    find_arg(args, flag)
        .map(|value| value.parse::<u64>().expect("invalid u64 flag"))
        .unwrap_or(default)
}

fn find_arg<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.windows(2)
        .find(|window| window[0] == flag)
        .map(|window| window[1].as_str())
}

fn print_usage() {
    eprintln!("usage:");
    eprintln!("  paper_validation mse --dimension 1536 --samples 1000 --bits 1,2,3,4 --seed 7");
    eprintln!("  paper_validation prod --dimension 1536 --samples 1000 --bits 1,2,3,4 --seed 7");
    eprintln!(
        "  paper_validation recall --dimension 256 --dataset-size 10000 --queries 100 --top-k 10 --bits 2,3 --seed 7"
    );
    eprintln!(
        "  paper_validation recall_curve --dimension 256 --dataset-size 10000 --queries 100 --ks 1,2,4,8,16,32,64 --bits 2,4 --seed 7"
    );
    eprintln!(
        "  paper_validation bounds --metric both --dimension 1536 --samples 1000 --bits 1,2,3,4 --seed 7"
    );
}
