use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

pub type VectorMetadata = BTreeMap<String, Value>;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VectorRecord {
    pub id: String,
    pub values: Vec<f32>,
    #[serde(default)]
    pub metadata: VectorMetadata,
    pub created_at_ms: u64,
    pub updated_at_ms: u64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VectorUpsertRecord {
    pub id: String,
    pub values: Vec<f32>,
    #[serde(default)]
    pub metadata: VectorMetadata,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VectorUpsertInput {
    pub records: Vec<VectorUpsertRecord>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VectorUpsertResult {
    pub upserted: usize,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VectorQueryInput {
    pub vector: Vec<f32>,
    pub top_k: Option<usize>,
    #[serde(default)]
    pub filter: VectorMetadata,
    pub min_score: Option<f32>,
    pub mode: Option<VectorQueryMode>,
    pub ef_search: Option<usize>,
    pub candidate_limit: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum VectorQueryMode {
    Exact,
    Ann,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VectorMatch {
    pub id: String,
    pub score: f32,
    #[serde(default)]
    pub metadata: VectorMetadata,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VectorQueryResult {
    pub matches: Vec<VectorMatch>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VectorDeleteInput {
    pub ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VectorDeleteResult {
    pub deleted: usize,
}

#[derive(Debug, thiserror::Error)]
pub enum VectorError {
    #[error("store error: {0}")]
    Store(#[from] crate::storage::StoreError),
    #[error("ann error: {0}")]
    Ann(#[from] feox_ann::AnnError),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("invalid vector operation: {0}")]
    Invalid(String),
}
