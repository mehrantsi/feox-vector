use serde_json::Value;

use crate::{Result, VectorError};

const MAX_SEGMENT_BYTES: usize = 512;
const MAX_RECORDS_PER_UPSERT: usize = 128;
const MAX_VECTOR_DIMENSIONS: usize = 4_096;
const MAX_METADATA_BYTES: usize = 16 * 1024;
const MAX_QUERY_RESULTS: usize = 100;
const MAX_ANN_CANDIDATES: usize = 10_000;

pub(crate) fn validate_segment(value: &str, field: &str) -> Result<()> {
    if value.trim().is_empty() || value.contains('/') || value.len() > MAX_SEGMENT_BYTES {
        return Err(VectorError::Invalid(format!(
            "{field} cannot be empty, contain /, or exceed {MAX_SEGMENT_BYTES} bytes"
        )));
    }
    Ok(())
}

pub(crate) fn validate_upsert_len(len: usize) -> Result<()> {
    if len == 0 || len > MAX_RECORDS_PER_UPSERT {
        return Err(VectorError::Invalid(format!(
            "upsert must contain 1 to {MAX_RECORDS_PER_UPSERT} records"
        )));
    }
    Ok(())
}

pub(crate) fn validate_values(values: &[f32]) -> Result<()> {
    if values.is_empty() || values.len() > MAX_VECTOR_DIMENSIONS {
        return Err(VectorError::Invalid(format!(
            "vector must contain 1 to {MAX_VECTOR_DIMENSIONS} dimensions"
        )));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(VectorError::Invalid(
            "vector values must be finite numbers".to_string(),
        ));
    }
    Ok(())
}

pub(crate) fn validate_metadata(value: &Value) -> Result<()> {
    let len = serde_json::to_vec(value)?.len();
    if len > MAX_METADATA_BYTES {
        return Err(VectorError::Invalid(format!(
            "metadata cannot exceed {MAX_METADATA_BYTES} bytes"
        )));
    }
    Ok(())
}

pub(crate) fn query_limit(top_k: Option<usize>) -> Result<usize> {
    let value = top_k.unwrap_or(10);
    if value == 0 || value > MAX_QUERY_RESULTS {
        return Err(VectorError::Invalid(format!(
            "top_k must be between 1 and {MAX_QUERY_RESULTS}"
        )));
    }
    Ok(value)
}

pub(crate) fn ann_candidate_limit(candidate_limit: Option<usize>, top_k: usize) -> Result<usize> {
    let value = candidate_limit.unwrap_or((top_k * 8).max(64));
    if value < top_k || value > MAX_ANN_CANDIDATES {
        return Err(VectorError::Invalid(format!(
            "candidate_limit must be between top_k and {MAX_ANN_CANDIDATES}"
        )));
    }
    Ok(value)
}

pub(crate) fn ann_ef_search(ef_search: Option<usize>, candidate_limit: usize) -> Result<usize> {
    let value = ef_search.unwrap_or(candidate_limit.max(64));
    if value < candidate_limit || value > MAX_ANN_CANDIDATES {
        return Err(VectorError::Invalid(format!(
            "ef_search must be between candidate_limit and {MAX_ANN_CANDIDATES}"
        )));
    }
    Ok(value)
}
