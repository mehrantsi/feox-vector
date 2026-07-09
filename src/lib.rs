//! # FeOx Vector
//!
//! An embedded vector store for Rust, built on [FeOxDB](https://crates.io/crates/feoxdb)
//! and [feox-ann](https://crates.io/crates/feox-ann). Exact and approximate cosine
//! similarity search with metadata filtering, in your process, with no server to run.
//!
//! ## Highlights
//!
//! - **Exact and ANN queries**: brute-force scans for small or heavily filtered
//!   collections, HNSW via `feox-ann` for large ones. Selectable per query.
//! - **Metadata filtering**: `$eq`, `$in`, `$nin` operators compiled against an
//!   inverted facet index. Small filtered candidate sets are re-ranked exactly
//!   instead of traversing the ANN graph, so selective filters return exact
//!   results.
//! - **Lock-free index refresh**: ANN snapshots are published through `arc-swap`.
//!   Writes mark a scope dirty and a background thread rebuilds without blocking
//!   queries. Queries fall back to exact scans until a fresh snapshot is ready.
//! - **Deterministic ranking**: results order by score, then recency, then id.
//!   Identical data returns identical output.
//! - **Namespaced collections**: records are scoped by `namespace / index /
//!   partition`, so one store serves many collections.
//! - **Memory or disk**: run fully in memory, or give FeOxDB a file path for
//!   persistence with write-behind buffering and explicit commit-point flushes.
//!
//! ## Quick start
//!
//! ```
//! use feox_vector::{Store, StoreConfig, VectorStore, VectorQueryInput, VectorUpsertInput, VectorUpsertRecord};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let store = Store::open(StoreConfig::default())?;
//! let vectors = VectorStore::new(store);
//!
//! vectors.upsert("app", "docs", "main", VectorUpsertInput {
//!     records: vec![VectorUpsertRecord {
//!         id: "doc-1".to_string(),
//!         values: vec![0.9, 0.1, 0.0],
//!         metadata: Default::default(),
//!     }],
//! })?;
//!
//! let result = vectors.query("app", "docs", "main", VectorQueryInput {
//!     vector: vec![1.0, 0.0, 0.0],
//!     top_k: Some(5),
//!     filter: Default::default(),
//!     min_score: None,
//!     mode: None,
//!     ef_search: None,
//!     candidate_limit: None,
//! })?;
//! assert_eq!(result.matches[0].id, "doc-1");
//! # Ok(())
//! # }
//! ```
//!
//! ## ANN mode
//!
//! Pass `mode: Some(VectorQueryMode::Ann)` to use the HNSW index. The first ANN
//! query on a scope schedules a background build and serves the query exactly in
//! the meantime. Dirty or rebuilding scopes also serve exact results, preserving
//! read-after-write semantics. Rebuilds are triggered automatically after writes,
//! or explicitly via
//! [`VectorStore::rebuild_ann`] / [`VectorStore::rebuild_ann_with_config`].

mod ann;
mod codec;
mod filter;
mod keys;
mod model;
mod storage;
mod store;
mod validation;

pub use feox_ann::AnnConfig;
pub use model::{
    VectorDeleteInput, VectorDeleteResult, VectorError, VectorMatch, VectorQueryInput,
    VectorQueryMode, VectorQueryResult, VectorRecord, VectorUpsertInput, VectorUpsertRecord,
    VectorUpsertResult,
};
pub use storage::{KeyValue, Store, StoreConfig, StoreError, StoreStats};
pub use store::VectorStore;

pub type Result<T> = std::result::Result<T, VectorError>;

#[cfg(test)]
mod tests;
