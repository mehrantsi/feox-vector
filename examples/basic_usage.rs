use std::collections::BTreeMap;

use feox_vector::{
    Store, StoreConfig, VectorQueryInput, VectorQueryMode, VectorStore, VectorUpsertInput,
    VectorUpsertRecord,
};
use serde_json::json;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let store = Store::open(StoreConfig::default())?;
    let vectors = VectorStore::new(store);

    let records = vec![
        record("rust-book", vec![0.9, 0.1, 0.0], "book"),
        record("iron-guide", vec![0.8, 0.2, 0.1], "guide"),
        record("oxide-paper", vec![0.7, 0.3, 0.2], "paper"),
        record("feather-notes", vec![0.0, 0.2, 0.9], "notes"),
    ];
    vectors.upsert("app", "docs", "main", VectorUpsertInput { records })?;

    let result = vectors.query(
        "app",
        "docs",
        "main",
        VectorQueryInput {
            vector: vec![1.0, 0.0, 0.0],
            top_k: Some(3),
            filter: BTreeMap::new(),
            min_score: None,
            mode: None,
            ef_search: None,
            candidate_limit: None,
        },
    )?;
    println!("exact query:");
    for item in &result.matches {
        println!("  {} (score {:.4})", item.id, item.score);
    }

    let mut filter = BTreeMap::new();
    filter.insert("kind".to_string(), json!({ "$in": ["book", "guide"] }));
    let filtered = vectors.query(
        "app",
        "docs",
        "main",
        VectorQueryInput {
            vector: vec![1.0, 0.0, 0.0],
            top_k: Some(3),
            filter,
            min_score: None,
            mode: None,
            ef_search: None,
            candidate_limit: None,
        },
    )?;
    println!("filtered to books and guides:");
    for item in &filtered.matches {
        println!("  {} (score {:.4})", item.id, item.score);
    }

    let indexed = vectors.rebuild_ann("app", "docs", "main", 3)?;
    println!("ann index built over {indexed} records");

    let ann = vectors.query(
        "app",
        "docs",
        "main",
        VectorQueryInput {
            vector: vec![1.0, 0.0, 0.0],
            top_k: Some(3),
            filter: BTreeMap::new(),
            min_score: None,
            mode: Some(VectorQueryMode::Ann),
            ef_search: None,
            candidate_limit: None,
        },
    )?;
    println!("ann query:");
    for item in &ann.matches {
        println!("  {} (score {:.4})", item.id, item.score);
    }

    Ok(())
}

fn record(id: &str, values: Vec<f32>, kind: &str) -> VectorUpsertRecord {
    let mut metadata = BTreeMap::new();
    metadata.insert("kind".to_string(), json!(kind));
    VectorUpsertRecord {
        id: id.to_string(),
        values,
        metadata,
    }
}
