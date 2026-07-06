use serde_json::{json, Value};

use crate::filter::{metadata_matches_filter, MetadataFilterIndex};
use crate::model::VectorMetadata;

#[test]
fn candidate_ids_use_eq_in_and_array_metadata() {
    let mut index = MetadataFilterIndex::default();
    let first = metadata(json!({
        "user_id": "user-a",
        "tags": ["runbook", "guide"]
    }));
    let second = metadata(json!({
        "user_id": "user-b",
        "tags": ["guide"]
    }));
    let third = metadata(json!({
        "user_id": "user-a",
        "tags": ["changelog"]
    }));
    index.insert("first", &first);
    index.insert("second", &second);
    index.insert("third", &third);

    let filter = metadata(json!({
        "user_id": { "$eq": "user-a" },
        "tags": { "$in": ["guide"] }
    }));
    let compiled = index.compile(&filter);
    let mut ids = compiled.candidate_ids(16).unwrap();
    ids.sort_unstable();

    assert_eq!(ids, vec!["first"]);
    assert!(metadata_matches_filter(&first, &filter));
    assert!(!metadata_matches_filter(&second, &filter));
    assert!(!metadata_matches_filter(&third, &filter));
}

#[test]
fn candidate_ids_respect_limit() {
    let mut index = MetadataFilterIndex::default();
    let first = metadata(json!({ "user_id": "user-a" }));
    let second = metadata(json!({ "user_id": "user-a" }));
    index.insert("first", &first);
    index.insert("second", &second);

    let filter = metadata(json!({ "user_id": { "$eq": "user-a" } }));
    let compiled = index.compile(&filter);

    assert!(compiled.candidate_ids(1).is_none());
    assert_eq!(compiled.candidate_ids(2).unwrap().len(), 2);
}

#[test]
fn nin_matches_absent_values_and_requires_array() {
    let record = metadata(json!({ "tag": "guide" }));
    let missing_ok = metadata(json!({ "source": { "$nin": ["archived"] } }));
    let scalar_nin = metadata(json!({ "tag": { "$nin": "guide" } }));

    assert!(metadata_matches_filter(&record, &missing_ok));
    assert!(!metadata_matches_filter(&record, &scalar_nin));

    let mut index = MetadataFilterIndex::default();
    index.insert("guide", &record);
    assert!(index.compile(&scalar_nin).candidate_ids(16).is_none());
}

fn metadata(value: Value) -> VectorMetadata {
    serde_json::from_value(value).unwrap()
}
