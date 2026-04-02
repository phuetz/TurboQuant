use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use crate::error::{Result, TurboQuantError};

pub fn load_text_embeddings<P: AsRef<Path>>(path: P) -> Result<Vec<Vec<f64>>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut rows = Vec::new();
    let mut expected_dimension = None;

    for (line_index, line) in reader.lines().enumerate() {
        let line_number = line_index + 1;
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        let values = trimmed
            .split([',', ' ', '\t'])
            .filter(|token| !token.is_empty())
            .enumerate()
            .map(|(column_index, token)| {
                token
                    .parse::<f64>()
                    .map_err(|_| TurboQuantError::ParseFloat {
                        line: line_number,
                        column: column_index + 1,
                        value: token.to_string(),
                    })
            })
            .collect::<Result<Vec<_>>>()?;

        if values.is_empty() {
            continue;
        }

        if let Some(dimension) = expected_dimension {
            if values.len() != dimension {
                return Err(TurboQuantError::InconsistentRowDimension {
                    line: line_number,
                    expected: dimension,
                    actual: values.len(),
                });
            }
        } else {
            expected_dimension = Some(values.len());
        }

        rows.push(values);
    }

    if rows.is_empty() {
        return Err(TurboQuantError::EmptyDataset);
    }

    Ok(rows)
}

pub fn split_train_and_queries(
    rows: &[Vec<f64>],
    train_size: usize,
    query_size: usize,
) -> Result<(Vec<Vec<f64>>, Vec<Vec<f64>>)> {
    let requested = train_size + query_size;
    if rows.len() < requested {
        return Err(TurboQuantError::InsufficientDatasetSize {
            available: rows.len(),
            requested,
        });
    }

    let train = rows[..train_size].to_vec();
    let queries = rows[train_size..train_size + query_size].to_vec();
    Ok((train, queries))
}

#[cfg(test)]
mod tests {
    use super::{load_text_embeddings, split_train_and_queries};
    use crate::TurboQuantError;
    use std::fs;

    #[test]
    fn split_reports_when_dataset_is_too_small() {
        let rows = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let err = split_train_and_queries(&rows, 2, 1).unwrap_err();
        assert!(matches!(
            err,
            TurboQuantError::InsufficientDatasetSize {
                available: 2,
                requested: 3
            }
        ));
    }

    #[test]
    fn text_embeddings_loader_parses_mixed_separators() {
        let path = std::env::temp_dir().join("turboquant_load_text_embeddings_test.txt");
        fs::write(&path, "# comment\n1,2,3\n4 5 6\n7\t8\t9\n").unwrap();

        let rows = load_text_embeddings(&path).unwrap();
        fs::remove_file(&path).unwrap();

        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(rows[1], vec![4.0, 5.0, 6.0]);
        assert_eq!(rows[2], vec![7.0, 8.0, 9.0]);
    }
}
