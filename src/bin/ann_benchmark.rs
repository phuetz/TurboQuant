use std::env;

use turboquant::OutlierSplitPlan;
use turboquant::data::{load_text_embeddings, split_train_and_queries};
use turboquant::experiment::{
    evaluate_mixed_recall_curve_dataset, evaluate_pq_recall_curve_dataset,
    evaluate_rabitq_recall_curve_dataset, evaluate_recall_curve_dataset,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    if args.is_empty() {
        print_usage();
        return Ok(());
    }

    let data_path = find_arg(&args, "--data").ok_or("missing --data")?;
    let query_path = find_arg(&args, "--queries");
    let train_size = parse_usize(&args, "--train-size", 100_000);
    let query_size = parse_usize(&args, "--query-size", 1_000);
    let seed = parse_u64(&args, "--seed", 7);
    let method = find_arg(&args, "--method").unwrap_or("turboquant");
    let bits = parse_u8_list(&args, "--bits", &[2, 4]);
    let ks = parse_usize_list(&args, "--ks", &[1, 2, 4, 8, 16, 32, 64]);

    let dataset_rows = load_text_embeddings(data_path)?;
    let (train, queries) = if let Some(query_path) = query_path {
        let queries = load_text_embeddings(query_path)?;
        let train = dataset_rows
            .into_iter()
            .take(train_size)
            .collect::<Vec<_>>();
        (
            train,
            queries.into_iter().take(query_size).collect::<Vec<_>>(),
        )
    } else {
        split_train_and_queries(&dataset_rows, train_size, query_size)?
    };

    println!(
        "method,code_spec,k,train_size,query_size,seed,recall_at_1,indexing_seconds,query_seconds"
    );
    match method {
        "turboquant" => {
            for bit_width in bits {
                let metrics =
                    evaluate_recall_curve_dataset(&train, &queries, bit_width, &ks, seed)?;
                let code_spec = format!("turboquant_b{bit_width}");
                for point in metrics.points {
                    println!(
                        "{method},{code_spec},{},{train_size},{query_size},{seed},{},{},{}",
                        point.k, point.recall_at_1, metrics.indexing_seconds, metrics.query_seconds
                    );
                }
            }
        }
        "pq" => {
            let dimension = train
                .first()
                .map(|row| row.len())
                .ok_or("empty train set")?;
            let subspaces = parse_usize(&args, "--subspaces", dimension.min(16));
            let iterations = parse_usize(&args, "--pq-iters", 12);
            for bit_width in bits {
                let metrics = evaluate_pq_recall_curve_dataset(
                    &train, &queries, subspaces, bit_width, iterations, &ks, seed,
                )?;
                let code_spec = format!("pq_m{subspaces}_b{bit_width}");
                for point in metrics.points {
                    println!(
                        "{method},{code_spec},{},{train_size},{query_size},{seed},{},{},{}",
                        point.k, point.recall_at_1, metrics.indexing_seconds, metrics.query_seconds
                    );
                }
            }
        }
        "rabitq" => {
            for bit_width in bits {
                let metrics =
                    evaluate_rabitq_recall_curve_dataset(&train, &queries, bit_width, &ks, seed)?;
                let code_spec = format!("rabitq_b{bit_width}");
                for point in metrics.points {
                    println!(
                        "{method},{code_spec},{},{train_size},{query_size},{seed},{},{},{}",
                        point.k, point.recall_at_1, metrics.indexing_seconds, metrics.query_seconds
                    );
                }
            }
        }
        "mixed_turboquant" => {
            let dimension = train
                .first()
                .map(|row| row.len())
                .ok_or("empty train set")?;
            let outlier_bits = parse_u8(&args, "--outlier-bits", 4);
            let regular_bits = parse_u8(&args, "--regular-bits", 3);
            let outlier_count =
                parse_usize(&args, "--outlier-count", default_outlier_count(dimension));
            let calibration_size = parse_usize(&args, "--calibration-size", train.len().min(4096));
            let calibration_size = calibration_size.min(train.len());
            let plan = OutlierSplitPlan::from_channel_rms(
                &train[..calibration_size],
                outlier_count,
                outlier_bits,
                regular_bits,
            )?;
            let metrics = evaluate_mixed_recall_curve_dataset(&train, &queries, &plan, &ks, seed)?;
            let code_spec = format!(
                "mixed_out{}_b{}_reg{}_b{}_eff{:.3}",
                plan.outlier_indices().len(),
                outlier_bits,
                plan.regular_indices().len(),
                regular_bits,
                plan.effective_bit_width()
            );
            for point in metrics.points {
                println!(
                    "{method},{code_spec},{},{train_size},{query_size},{seed},{},{},{}",
                    point.k, point.recall_at_1, metrics.indexing_seconds, metrics.query_seconds
                );
            }
        }
        other => return Err(format!("unsupported --method value: {other}").into()),
    }

    Ok(())
}

fn parse_u8_list(args: &[String], flag: &str, default: &[u8]) -> Vec<u8> {
    find_arg(args, flag)
        .map(|value| {
            value
                .split([',', ' ', '\t'])
                .filter(|item| !item.is_empty())
                .map(|item| item.parse::<u8>().expect("invalid u8 list"))
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|| default.to_vec())
}

fn parse_usize_list(args: &[String], flag: &str, default: &[usize]) -> Vec<usize> {
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

fn find_arg<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.windows(2)
        .find(|window| window[0] == flag)
        .map(|window| window[1].as_str())
}

fn print_usage() {
    eprintln!("usage:");
    eprintln!(
        "  ann_benchmark --method turboquant --data embeddings.txt --train-size 100000 --query-size 1000 --bits 2,4 --ks 1,2,4,8,16,32,64 --seed 7"
    );
    eprintln!(
        "  ann_benchmark --method pq --data embeddings.txt --train-size 100000 --query-size 1000 --bits 8 --subspaces 24 --pq-iters 12 --ks 1,2,4,8,16,32,64 --seed 7"
    );
    eprintln!(
        "  ann_benchmark --method rabitq --data embeddings.txt --train-size 100000 --query-size 1000 --bits 2,4 --ks 1,2,4,8,16,32,64 --seed 7"
    );
    eprintln!(
        "  ann_benchmark --method mixed_turboquant --data embeddings.txt --train-size 100000 --query-size 1000 --outlier-count 32 --outlier-bits 4 --regular-bits 3 --ks 1,2,4,8,16,32,64 --seed 7"
    );
}

fn default_outlier_count(dimension: usize) -> usize {
    if dimension >= 8 {
        (dimension / 4).min(dimension.saturating_sub(2))
    } else {
        0
    }
}
