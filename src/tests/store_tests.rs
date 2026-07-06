use crate::storage::{Store, StoreConfig};
use serde_json::json;

use crate::{VectorDeleteInput, VectorQueryInput, VectorQueryMode, VectorStore, VectorUpsertInput};

#[test]
fn queries_partitioned_vectors_by_cosine_score() {
    let store = VectorStore::new(Store::open(StoreConfig::default()).unwrap());
    store
        .upsert(
            "ns-a",
            "memory",
            "user-a",
            VectorUpsertInput {
                records: vec![
                    crate::VectorUpsertRecord {
                        id: "north".to_string(),
                        values: vec![1.0, 0.0],
                        metadata: serde_json::from_value(json!({ "userId": "user-a" })).unwrap(),
                    },
                    crate::VectorUpsertRecord {
                        id: "east".to_string(),
                        values: vec![0.0, 1.0],
                        metadata: serde_json::from_value(json!({ "userId": "user-a" })).unwrap(),
                    },
                ],
            },
        )
        .unwrap();

    let result = store
        .query(
            "ns-a",
            "memory",
            "user-a",
            VectorQueryInput {
                vector: vec![0.9, 0.1],
                top_k: Some(1),
                filter: serde_json::from_value(json!({ "userId": { "$eq": "user-a" } })).unwrap(),
                min_score: Some(0.0),
                mode: None,
                ef_search: None,
                candidate_limit: None,
            },
        )
        .unwrap();

    assert_eq!(result.matches.len(), 1);
    assert_eq!(result.matches[0].id, "north");
}

#[test]
fn deletes_records() {
    let store = VectorStore::new(Store::open(StoreConfig::default()).unwrap());
    store
        .upsert(
            "ns-a",
            "memory",
            "user-a",
            VectorUpsertInput {
                records: vec![crate::VectorUpsertRecord {
                    id: "one".to_string(),
                    values: vec![1.0],
                    metadata: Default::default(),
                }],
            },
        )
        .unwrap();

    let deleted = store
        .delete(
            "ns-a",
            "memory",
            "user-a",
            VectorDeleteInput {
                ids: vec!["one".to_string()],
            },
        )
        .unwrap();

    assert_eq!(deleted.deleted, 1);
    let result = store
        .query(
            "ns-a",
            "memory",
            "user-a",
            VectorQueryInput {
                vector: vec![1.0],
                top_k: Some(10),
                filter: Default::default(),
                min_score: None,
                mode: None,
                ef_search: None,
                candidate_limit: None,
            },
        )
        .unwrap();
    assert!(result.matches.is_empty());
}

#[test]
fn ann_query_uses_rebuilt_snapshot() {
    let raw_store = Store::open(StoreConfig::default()).unwrap();
    let writer = VectorStore::new(raw_store.clone());
    writer
        .upsert(
            "ns-a",
            "knowledge",
            "org-a",
            VectorUpsertInput {
                records: vec![
                    crate::VectorUpsertRecord {
                        id: "north".to_string(),
                        values: vec![1.0, 0.0],
                        metadata: Default::default(),
                    },
                    crate::VectorUpsertRecord {
                        id: "east".to_string(),
                        values: vec![0.0, 1.0],
                        metadata: Default::default(),
                    },
                    crate::VectorUpsertRecord {
                        id: "south".to_string(),
                        values: vec![-1.0, 0.0],
                        metadata: Default::default(),
                    },
                ],
            },
        )
        .unwrap();

    let reader = VectorStore::new(raw_store);
    let indexed = reader.rebuild_ann("ns-a", "knowledge", "org-a", 2).unwrap();
    assert_eq!(indexed, 3);

    let result = reader
        .query(
            "ns-a",
            "knowledge",
            "org-a",
            VectorQueryInput {
                vector: vec![0.95, 0.05],
                top_k: Some(1),
                filter: Default::default(),
                min_score: Some(0.0),
                mode: Some(VectorQueryMode::Ann),
                ef_search: Some(16),
                candidate_limit: Some(8),
            },
        )
        .unwrap();

    assert_eq!(result.matches.len(), 1);
    assert_eq!(result.matches[0].id, "north");
}

#[test]
fn ann_query_applies_metadata_filter_from_snapshot() {
    let raw_store = Store::open(StoreConfig::default()).unwrap();
    let store = VectorStore::new(raw_store);
    store
        .upsert(
            "ns-a",
            "knowledge",
            "org-a",
            VectorUpsertInput {
                records: vec![
                    crate::VectorUpsertRecord {
                        id: "runbook-north".to_string(),
                        values: vec![1.0, 0.0],
                        metadata: serde_json::from_value(json!({ "tag": "runbook" })).unwrap(),
                    },
                    crate::VectorUpsertRecord {
                        id: "guide-north".to_string(),
                        values: vec![0.95, 0.05],
                        metadata: serde_json::from_value(json!({ "tag": "guide" })).unwrap(),
                    },
                    crate::VectorUpsertRecord {
                        id: "guide-east".to_string(),
                        values: vec![0.0, 1.0],
                        metadata: serde_json::from_value(json!({ "tag": "guide" })).unwrap(),
                    },
                ],
            },
        )
        .unwrap();

    assert_eq!(
        store.rebuild_ann("ns-a", "knowledge", "org-a", 2).unwrap(),
        3
    );

    let result = store
        .query(
            "ns-a",
            "knowledge",
            "org-a",
            VectorQueryInput {
                vector: vec![1.0, 0.0],
                top_k: Some(2),
                filter: serde_json::from_value(json!({ "tag": { "$eq": "guide" } })).unwrap(),
                min_score: Some(0.0),
                mode: Some(VectorQueryMode::Ann),
                ef_search: Some(16),
                candidate_limit: Some(8),
            },
        )
        .unwrap();

    assert_eq!(result.matches.len(), 2);
    assert_eq!(result.matches[0].id, "guide-north");
    assert_eq!(result.matches[1].id, "guide-east");
}

#[test]
fn queries_metadata_filter_in_and_not_in() {
    let store = VectorStore::new(Store::open(StoreConfig::default()).unwrap());
    store
        .upsert(
            "ns-a",
            "knowledge",
            "org-a",
            VectorUpsertInput {
                records: vec![
                    crate::VectorUpsertRecord {
                        id: "runbook".to_string(),
                        values: vec![1.0, 0.0],
                        metadata: serde_json::from_value(json!({ "tag": "runbook" })).unwrap(),
                    },
                    crate::VectorUpsertRecord {
                        id: "guide".to_string(),
                        values: vec![0.9, 0.1],
                        metadata: serde_json::from_value(json!({ "tag": "guide" })).unwrap(),
                    },
                    crate::VectorUpsertRecord {
                        id: "changelog".to_string(),
                        values: vec![0.8, 0.2],
                        metadata: serde_json::from_value(json!({ "tag": "changelog" })).unwrap(),
                    },
                ],
            },
        )
        .unwrap();

    let included = store
        .query(
            "ns-a",
            "knowledge",
            "org-a",
            VectorQueryInput {
                vector: vec![1.0, 0.0],
                top_k: Some(3),
                filter: serde_json::from_value(json!({ "tag": { "$in": ["guide", "changelog"] } }))
                    .unwrap(),
                min_score: None,
                mode: None,
                ef_search: None,
                candidate_limit: None,
            },
        )
        .unwrap();
    assert_eq!(
        included
            .matches
            .iter()
            .map(|item| item.id.as_str())
            .collect::<Vec<_>>(),
        vec!["guide", "changelog"]
    );

    let excluded = store
        .query(
            "ns-a",
            "knowledge",
            "org-a",
            VectorQueryInput {
                vector: vec![1.0, 0.0],
                top_k: Some(3),
                filter: serde_json::from_value(json!({ "tag": { "$nin": ["runbook"] } })).unwrap(),
                min_score: None,
                mode: None,
                ef_search: None,
                candidate_limit: None,
            },
        )
        .unwrap();
    assert_eq!(
        excluded
            .matches
            .iter()
            .map(|item| item.id.as_str())
            .collect::<Vec<_>>(),
        vec!["guide", "changelog"]
    );
}

#[test]
fn ann_query_without_snapshot_falls_back_to_exact() {
    let raw_store = Store::open(StoreConfig::default()).unwrap();
    let store = VectorStore::new(raw_store);
    store
        .upsert(
            "ns-a",
            "knowledge",
            "org-a",
            VectorUpsertInput {
                records: vec![
                    crate::VectorUpsertRecord {
                        id: "north".to_string(),
                        values: vec![1.0, 0.0],
                        metadata: Default::default(),
                    },
                    crate::VectorUpsertRecord {
                        id: "east".to_string(),
                        values: vec![0.0, 1.0],
                        metadata: Default::default(),
                    },
                ],
            },
        )
        .unwrap();

    let result = store
        .query(
            "ns-a",
            "knowledge",
            "org-a",
            VectorQueryInput {
                vector: vec![1.0, 0.0],
                top_k: Some(1),
                filter: Default::default(),
                min_score: None,
                mode: Some(VectorQueryMode::Ann),
                ef_search: Some(16),
                candidate_limit: Some(8),
            },
        )
        .unwrap();

    assert_eq!(result.matches.len(), 1);
    assert_eq!(result.matches[0].id, "north");
}
