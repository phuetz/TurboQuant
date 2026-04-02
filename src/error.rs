use thiserror::Error;

pub type Result<T> = std::result::Result<T, TurboQuantError>;

#[derive(Debug, Error)]
pub enum TurboQuantError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("dimension must be at least 2, got {0}")]
    InvalidDimension(usize),
    #[error("bit width {0} is too large for the current implementation")]
    UnsupportedBitWidth(u8),
    #[error("expected vector of length {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error("quantized index count mismatch: expected {expected}, got {actual}")]
    InvalidIndexCount { expected: usize, actual: usize },
    #[error("QJL sign count mismatch: expected {expected}, got {actual}")]
    InvalidSignCount { expected: usize, actual: usize },
    #[error("codebook index {index} is out of range for {codebook_len} centroids")]
    InvalidCodebookIndex { index: usize, codebook_len: usize },
    #[error("bit width must be at least 1 for TurboQuantProd")]
    ProdBitWidthTooSmall,
    #[error("batch length mismatch: expected {expected}, got {actual}")]
    BatchLengthMismatch { expected: usize, actual: usize },
    #[error("packed value {value} does not fit in {bits_per_symbol} bits")]
    PackedValueOutOfRange { value: u64, bits_per_symbol: u8 },
    #[error("packed symbol index {index} is out of range for {symbol_count} symbols")]
    PackedIndexOutOfRange { index: usize, symbol_count: usize },
    #[error(
        "packed layout mismatch: expected {expected_symbols} symbols at {expected_bits_per_symbol} bits, got {actual_symbols} symbols at {actual_bits_per_symbol} bits"
    )]
    PackedLayoutMismatch {
        expected_symbols: usize,
        actual_symbols: usize,
        expected_bits_per_symbol: u8,
        actual_bits_per_symbol: u8,
    },
    #[error("invalid experiment parameter `{parameter}`: {detail}")]
    InvalidExperimentParameter {
        parameter: &'static str,
        detail: &'static str,
    },
    #[error("channel index {index} is out of range for dimension {dimension}")]
    InvalidChannelIndex { index: usize, dimension: usize },
    #[error("channel index {index} appears more than once in the split plan")]
    DuplicateChannelIndex { index: usize },
    #[error(
        "invalid channel partition: outliers={outliers}, regular={regular}; each non-empty partition must have at least 2 channels"
    )]
    InvalidChannelPartition { outliers: usize, regular: usize },
    #[error("calibration sample set is empty")]
    EmptyCalibrationSet,
    #[error("embedding dataset is empty")]
    EmptyDataset,
    #[error("could not parse float `{value}` at line {line}, column {column}")]
    ParseFloat {
        line: usize,
        column: usize,
        value: String,
    },
    #[error("inconsistent embedding dimension at line {line}: expected {expected}, got {actual}")]
    InconsistentRowDimension {
        line: usize,
        expected: usize,
        actual: usize,
    },
    #[error("dataset is too small: available={available}, requested={requested}")]
    InsufficientDatasetSize { available: usize, requested: usize },
    #[error("invalid PQ subspace count {subspaces} for dimension {dimension}")]
    InvalidSubspaceCount { dimension: usize, subspaces: usize },
    #[error("invalid tensor shape axis `{axis}` with value {value}")]
    InvalidTensorShape { axis: &'static str, value: usize },
    #[error("tensor element count mismatch: expected {expected}, got {actual}")]
    TensorElementCountMismatch { expected: usize, actual: usize },
    #[error("tensor row count mismatch: expected {expected}, got {actual}")]
    TensorRowCountMismatch { expected: usize, actual: usize },
    #[error(
        "KV tensor layout mismatch: expected batch={expected_batch}, heads={expected_heads}, dim={expected_dim}; got batch={actual_batch}, heads={actual_heads}, dim={actual_dim}"
    )]
    KvTensorLayoutMismatch {
        expected_batch: usize,
        expected_heads: usize,
        expected_dim: usize,
        actual_batch: usize,
        actual_heads: usize,
        actual_dim: usize,
    },
    #[error(
        "KV tensor shape mismatch: expected batch={expected_batch}, heads={expected_heads}, seq={expected_seq}, dim={expected_dim}; got batch={actual_batch}, heads={actual_heads}, seq={actual_seq}, dim={actual_dim}"
    )]
    KvTensorShapeMismatch {
        expected_batch: usize,
        expected_heads: usize,
        expected_seq: usize,
        expected_dim: usize,
        actual_batch: usize,
        actual_heads: usize,
        actual_seq: usize,
        actual_dim: usize,
    },
    #[error("invalid sequence range [{start}, {end}) for sequence length {sequence_length}")]
    InvalidSequenceRange {
        start: usize,
        end: usize,
        sequence_length: usize,
    },
    #[error(
        "KV tensor code variant `{actual}` does not match the expected quantizer/code kind `{expected}`"
    )]
    KvQuantizerMismatch {
        expected: &'static str,
        actual: &'static str,
    },
    #[error("layer index {layer_index} is out of range for {layer_count} layers")]
    LayerIndexOutOfRange {
        layer_index: usize,
        layer_count: usize,
    },
    #[error("layer index {layer_index} appears more than once")]
    DuplicateLayerIndex { layer_index: usize },
    #[error("layer set is empty")]
    EmptyLayerSet,
}
