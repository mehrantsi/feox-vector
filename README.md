<div align="center">

# FeOx Vector

Embedded vector store for Rust, built on [FeOxDB](https://github.com/mehrantsi/feoxdb). Exact and approximate cosine search with metadata filtering, in your process, with no server to run.

</div>

[Documentation](https://docs.rs/feox-vector) | [FeOxDB](https://feoxdb.com) | [feox-ann](https://github.com/mehrantsi/feox-ann) | [Issues](https://github.com/mehrantsi/feox-vector/issues)

## Features

- **Exact and ANN Queries**: Brute-force scans for small or heavily filtered collections, HNSW (via [feox-ann](https://github.com/mehrantsi/feox-ann)) for large ones. Selectable per query.
- **SIMD Scoring**: Cosine similarity runs on feox-ann's NEON (aarch64) and AVX2+FMA (x86_64, runtime-detected) dot-product kernels, on both the exact and ANN paths.
- **Metadata Filtering**: `$eq`, `$in`, `$nin` operators compiled against an inverted facet index. Small filtered candidate sets are re-ranked exactly instead of traversing the ANN graph, so selective filters return exact results.
- **Lock-Free Index Refresh**: ANN snapshots publish through `arc-swap`. Writes mark a scope dirty and a background thread rebuilds while queries keep flowing. Until a fresh snapshot is ready, ANN queries fall back to exact scans so completed writes and deletes are immediately visible.
- **Deterministic Output**: feox-ann builds reproducible graphs, and results rank by score, then recency, then id. Identical data returns identical output.
- **Namespaced Collections**: Records scope by `namespace / index / partition`. One store serves many collections.
- **Memory or Disk**: Run fully in memory, or point FeOxDB at a file for persistence with write-behind buffering.
- **Compact Records**: Vectors are stored in a binary format (magic `FXV1`) with little-endian f32 values and JSON metadata. Vector data is decoded without JSON parsing.

## Quick Start

```toml
[dependencies]
feox-vector = "0.1"
```

```rust
use feox_vector::{Store, StoreConfig, VectorStore, VectorQueryInput, VectorUpsertInput, VectorUpsertRecord};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let store = Store::open(StoreConfig::default())?;   // in-memory
    let vectors = VectorStore::new(store);

    vectors.upsert("app", "docs", "main", VectorUpsertInput {
        records: vec![VectorUpsertRecord {
            id: "doc-1".to_string(),
            values: embedding,
            metadata: Default::default(),
        }],
    })?;

    let result = vectors.query("app", "docs", "main", VectorQueryInput {
        vector: query_embedding,
        top_k: Some(10),
        filter: Default::default(),
        min_score: Some(0.5),
        mode: None,                 // exact scan
        ef_search: None,
        candidate_limit: None,
    })?;
    Ok(())
}
```

Persistent mode:

```rust
let store = Store::open(StoreConfig {
    device_path: Some("/var/lib/myapp/vectors.feox".into()),
    ..Default::default()
})?;
let vectors = VectorStore::new(store);

// Flush at application commit points that must survive abrupt process exit.
vectors.flush()?;
```

## Metadata Filters

Filters are a map from metadata key to either a literal (equality) or an operator object:

```rust
use serde_json::json;

let mut filter = std::collections::BTreeMap::new();
filter.insert("kind".to_string(), json!("article"));                          // equality
filter.insert("lang".to_string(), json!({ "$in": ["en", "da"] }));            // any of
filter.insert("status".to_string(), json!({ "$nin": ["draft", "hidden"] }));  // none of
```

In ANN mode, filters compile against an inverted facet index built alongside the graph. When the filtered candidate set is small (at most 4,096 ids), the query skips the graph entirely and re-ranks those candidates exactly. Selective filters therefore return exact results.

## ANN Mode

```rust
use feox_vector::{VectorQueryMode, AnnConfig};

// Explicit build (blocking), with default or custom HNSW parameters:
vectors.rebuild_ann("app", "docs", "main", 384)?;
vectors.rebuild_ann_with_config("app", "docs", "main", AnnConfig {
    ef_search: 128,
    ..AnnConfig::for_dimensions(384)
})?;

// Query:
let result = vectors.query("app", "docs", "main", VectorQueryInput {
    mode: Some(VectorQueryMode::Ann),
    ..input
})?;
```

The first ANN query on a scope schedules a background build and answers exactly in the meantime. Writes mark the scope dirty and trigger a fresh rebuild on the next ANN query. Dirty and rebuilding scopes answer exactly until the fresh snapshot publishes, preserving read-after-write semantics without locking query execution.

**Note**: ANN snapshots are in-memory and rebuilt from stored records on process start (first ANN query per scope). Records themselves persist via FeOxDB.

## Benchmarks

```bash
cargo bench
```

Criterion benches populate 10,000 x 128d records from a seeded generator and measure batched upsert throughput plus top-10 query latency for exact, ANN, and filtered-ANN modes. Deterministic inputs make numbers comparable across machines and commits.

Reference numbers on Apple Silicon, single-threaded, 10k x 128d with metadata:

| Operation | Result |
|---|---|
| Batched upsert | ~13.5 ms for 10k records (~740k records/s) |
| ANN query top-10 | ~102 us |
| Exact scan top-10 (full 10k) | ~3.8 ms |
| Filtered ANN (filter matches 2,500 ids, exact re-rank path) | ~1.6 ms, exact results |

## Limits

Inputs are validated with practical ceilings: at most 128 records per upsert, 4,096 dimensions, 16 KB metadata per record, and 100 results per query. Segment names (`namespace`, `index`, `partition`, ids) must be non-empty, `/`-free, and at most 512 bytes.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).
