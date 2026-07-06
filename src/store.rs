use std::cmp::Ordering;
use std::sync::Arc;
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::storage::Store;
use feox_ann::{AnnConfig, AnnQuery};
use serde_json::{Map, Value};

use crate::ann::{AnnRegistry, AnnSnapshot};
use crate::codec::{decode_ann_record, decode_record, encode_record};
use crate::filter::{metadata_matches_filter, MetadataFilterIndex};
use crate::keys::{record_key, record_prefix};
use crate::validation::{
    ann_candidate_limit, ann_ef_search, query_limit, validate_metadata, validate_segment,
    validate_upsert_len, validate_values,
};
use crate::{
    Result, VectorDeleteInput, VectorDeleteResult, VectorMatch, VectorQueryInput, VectorQueryMode,
    VectorQueryResult, VectorRecord, VectorUpsertInput, VectorUpsertResult,
};

const QUERY_SCAN_BATCH: usize = 512;
const FILTER_EXACT_MAX_CANDIDATES: usize = 4_096;

#[derive(Clone)]
pub struct VectorStore {
    store: Store,
    indexes: Arc<AnnRegistry>,
}

impl VectorStore {
    pub fn new(store: Store) -> Self {
        Self {
            store,
            indexes: Arc::new(AnnRegistry::default()),
        }
    }

    pub fn upsert(
        &self,
        namespace: &str,
        index: &str,
        partition: &str,
        input: VectorUpsertInput,
    ) -> Result<VectorUpsertResult> {
        validate_scope(namespace, index, partition)?;
        validate_upsert_len(input.records.len())?;

        let now = now_ms();
        let scope = scope_key(namespace, index, partition);
        let mut upserted = 0;
        for record in input.records {
            validate_segment(&record.id, "vector_id")?;
            validate_values(&record.values)?;
            validate_metadata(&Value::Object(Map::from_iter(record.metadata.clone())))?;

            let key = record_key(namespace, index, partition, &record.id);
            let existing = self.get_record(&key)?;
            let created_at_ms = existing
                .as_ref()
                .map(|record| record.created_at_ms)
                .unwrap_or(now);
            let encoded = encode_record(&record.values, &record.metadata, created_at_ms, now)?;
            self.store.put_bytes(&key, &encoded)?;
            self.indexes.mark_dirty_if_present(&scope);
            upserted += 1;
        }

        Ok(VectorUpsertResult { upserted })
    }

    pub fn query(
        &self,
        namespace: &str,
        index: &str,
        partition: &str,
        input: VectorQueryInput,
    ) -> Result<VectorQueryResult> {
        validate_scope(namespace, index, partition)?;
        validate_values(&input.vector)?;
        validate_metadata(&Value::Object(Map::from_iter(input.filter.clone())))?;

        let limit = query_limit(input.top_k)?;
        if input
            .min_score
            .map(|score| !score.is_finite())
            .unwrap_or(false)
        {
            return Err(crate::VectorError::Invalid(
                "min_score must be a finite number".to_string(),
            ));
        }
        let min_score = input.min_score.unwrap_or(f32::NEG_INFINITY);

        if input.mode == Some(VectorQueryMode::Ann) {
            if let Some(result) =
                self.query_ann(namespace, index, partition, &input, limit, min_score)?
            {
                return Ok(result);
            }
        }

        self.query_exact(namespace, index, partition, &input, limit, min_score)
    }

    pub fn delete(
        &self,
        namespace: &str,
        index: &str,
        partition: &str,
        input: VectorDeleteInput,
    ) -> Result<VectorDeleteResult> {
        validate_scope(namespace, index, partition)?;

        let scope = scope_key(namespace, index, partition);
        let mut deleted = 0;
        for id in input.ids {
            validate_segment(&id, "vector_id")?;
            if self
                .store
                .delete(&record_key(namespace, index, partition, &id))?
            {
                self.indexes.mark_dirty_if_present(&scope);
                deleted += 1;
            }
        }

        Ok(VectorDeleteResult { deleted })
    }

    fn query_exact(
        &self,
        namespace: &str,
        index: &str,
        partition: &str,
        input: &VectorQueryInput,
        limit: usize,
        min_score: f32,
    ) -> Result<VectorQueryResult> {
        let mut matches = Vec::new();
        let prefix = record_prefix(namespace, index, partition);
        let mut after = None;
        loop {
            let rows = self
                .store
                .list_prefix_after(&prefix, after.as_deref(), QUERY_SCAN_BATCH)?;
            if rows.is_empty() {
                break;
            }
            after = rows.last().map(|row| row.key.clone());

            for row in rows {
                let record = decode_record(&row.key, &row.value)?;
                if record.values.len() != input.vector.len() {
                    continue;
                }
                if !metadata_matches_filter(&record.metadata, &input.filter) {
                    continue;
                }

                let score = cosine_similarity(&input.vector, &record.values);
                if score < min_score {
                    continue;
                }

                matches.push((
                    record.updated_at_ms,
                    VectorMatch {
                        id: record.id,
                        score,
                        metadata: record.metadata,
                    },
                ));
            }
        }

        Ok(rank_matches(matches, limit))
    }

    #[allow(clippy::too_many_arguments)]
    fn query_exact_ids(
        &self,
        namespace: &str,
        index: &str,
        partition: &str,
        ids: &[&str],
        input: &VectorQueryInput,
        limit: usize,
        min_score: f32,
    ) -> Result<VectorQueryResult> {
        let mut matches = Vec::new();
        for id in ids {
            let key = record_key(namespace, index, partition, id);
            let Some(record) = self.get_record(&key)? else {
                continue;
            };
            if record.values.len() != input.vector.len() {
                continue;
            }
            if !metadata_matches_filter(&record.metadata, &input.filter) {
                continue;
            }

            let score = cosine_similarity(&input.vector, &record.values);
            if score < min_score {
                continue;
            }

            matches.push((
                record.updated_at_ms,
                VectorMatch {
                    id: record.id,
                    score,
                    metadata: record.metadata,
                },
            ));
        }

        Ok(rank_matches(matches, limit))
    }

    fn query_ann(
        &self,
        namespace: &str,
        index: &str,
        partition: &str,
        input: &VectorQueryInput,
        limit: usize,
        min_score: f32,
    ) -> Result<Option<VectorQueryResult>> {
        let requested_candidate_limit = input
            .candidate_limit
            .or_else(|| (!input.filter.is_empty()).then_some((limit * 32).max(256)));
        let candidate_limit = ann_candidate_limit(requested_candidate_limit, limit)?;
        let ef_search = ann_ef_search(input.ef_search, candidate_limit)?;
        let scope = scope_key(namespace, index, partition);
        let ann_scope = self.indexes.scope(&scope);
        let Some(snapshot) = ann_scope.snapshot() else {
            self.schedule_ann_rebuild(
                namespace.to_string(),
                index.to_string(),
                partition.to_string(),
                scope,
                input.vector.len(),
            );
            return Ok(None);
        };
        if snapshot.index.dimensions() != input.vector.len() {
            self.schedule_ann_rebuild(
                namespace.to_string(),
                index.to_string(),
                partition.to_string(),
                scope,
                input.vector.len(),
            );
            return Ok(None);
        }
        if ann_scope.is_dirty() {
            self.schedule_ann_rebuild(
                namespace.to_string(),
                index.to_string(),
                partition.to_string(),
                scope.clone(),
                input.vector.len(),
            );
        }

        let compiled_filter = snapshot.filters.compile(&input.filter);
        if compiled_filter.is_none() {
            return Ok(Some(VectorQueryResult {
                matches: Vec::new(),
            }));
        }
        if !input.filter.is_empty() {
            if let Some(ids) = compiled_filter.candidate_ids(FILTER_EXACT_MAX_CANDIDATES) {
                return Ok(Some(self.query_exact_ids(
                    namespace, index, partition, &ids, input, limit, min_score,
                )?));
            }
        }
        let filter = |id: &str| compiled_filter.accepts(id);
        let query_filter = if input.filter.is_empty() || compiled_filter.is_all() {
            None
        } else {
            Some(&filter as &dyn feox_ann::AnnFilter)
        };

        let candidates = snapshot.index.query(AnnQuery {
            vector: &input.vector,
            top_k: candidate_limit,
            ef_search: Some(ef_search),
            filter: query_filter,
        })?;

        let mut matches = Vec::new();
        for candidate in candidates {
            let key = record_key(namespace, index, partition, &candidate.id);
            let Some(record) = self.get_record(&key)? else {
                continue;
            };
            if record.values.len() != input.vector.len() {
                continue;
            }
            if !metadata_matches_filter(&record.metadata, &input.filter) {
                continue;
            }
            let score = cosine_similarity(&input.vector, &record.values);
            if score < min_score {
                continue;
            }
            matches.push((
                record.updated_at_ms,
                VectorMatch {
                    id: record.id,
                    score,
                    metadata: record.metadata,
                },
            ));
        }

        Ok(Some(rank_matches(matches, limit)))
    }

    pub fn rebuild_ann(
        &self,
        namespace: &str,
        index: &str,
        partition: &str,
        dimensions: usize,
    ) -> Result<usize> {
        self.rebuild_ann_with_config(
            namespace,
            index,
            partition,
            AnnConfig::for_dimensions(dimensions),
        )
    }

    pub fn rebuild_ann_with_config(
        &self,
        namespace: &str,
        index: &str,
        partition: &str,
        config: AnnConfig,
    ) -> Result<usize> {
        validate_scope(namespace, index, partition)?;
        let scope = scope_key(namespace, index, partition);
        self.rebuild_ann_for_scope(namespace, index, partition, &scope, config)
    }

    fn rebuild_ann_index(
        &self,
        namespace: &str,
        index: &str,
        partition: &str,
        config: AnnConfig,
    ) -> Result<AnnSnapshot> {
        let dimensions = config.dimensions;
        let mut ann = feox_ann::AnnIndex::new(config)?;
        let mut filters = MetadataFilterIndex::default();
        let prefix = record_prefix(namespace, index, partition);
        let mut after = None;
        let mut cursor = ann.insert_cursor();
        loop {
            let rows = self
                .store
                .list_prefix_after(&prefix, after.as_deref(), QUERY_SCAN_BATCH)?;
            if rows.is_empty() {
                break;
            }
            after = rows.last().map(|row| row.key.clone());
            cursor.reserve(rows.len());
            for row in rows {
                if let Some(record) = decode_ann_record(&row.key, &row.value, dimensions)? {
                    filters.insert(&record.id, &record.metadata);
                    cursor.upsert_owned(record.id, record.values)?;
                }
            }
        }
        drop(cursor);
        Ok(AnnSnapshot {
            index: ann,
            filters,
        })
    }

    fn rebuild_ann_for_scope(
        &self,
        namespace: &str,
        index: &str,
        partition: &str,
        scope_key: &str,
        config: AnnConfig,
    ) -> Result<usize> {
        let scope = self.indexes.scope(scope_key);
        if !scope.begin_rebuild() {
            return Ok(scope
                .snapshot()
                .map(|snapshot| snapshot.index.len())
                .unwrap_or(0));
        }

        match self.rebuild_ann_index(namespace, index, partition, config) {
            Ok(snapshot) => {
                let len = snapshot.index.len();
                scope.publish(snapshot);
                Ok(len)
            }
            Err(error) => {
                scope.finish_failed_rebuild();
                Err(error)
            }
        }
    }

    fn schedule_ann_rebuild(
        &self,
        namespace: String,
        index: String,
        partition: String,
        scope_key: String,
        dimensions: usize,
    ) {
        let scope = self.indexes.scope(&scope_key);
        if !scope.begin_rebuild() {
            return;
        }

        let store = self.clone();
        let _ = thread::Builder::new()
            .name("feox-ann-rebuild".to_string())
            .spawn(move || {
                let result = store.rebuild_ann_index(
                    &namespace,
                    &index,
                    &partition,
                    AnnConfig::for_dimensions(dimensions),
                );
                match result {
                    Ok(index) => scope.publish(index),
                    Err(error) => {
                        scope.finish_failed_rebuild();
                        eprintln!("[feox-vector] ANN rebuild failed for {scope_key}: {error}");
                    }
                }
            });
    }

    fn get_record(&self, key: &str) -> Result<Option<VectorRecord>> {
        let Some(bytes) = self.store.get_bytes(key)? else {
            return Ok(None);
        };
        Ok(Some(decode_record(key, &bytes)?))
    }
}

fn validate_scope(namespace: &str, index: &str, partition: &str) -> Result<()> {
    validate_segment(namespace, "namespace")?;
    validate_segment(index, "index")?;
    validate_segment(partition, "partition")?;
    Ok(())
}

fn cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
    let len = left.len().min(right.len());
    let (left, right) = (&left[..len], &right[..len]);
    let left_norm = feox_ann::dot(left, left);
    let right_norm = feox_ann::dot(right, right);
    if left_norm == 0.0 || right_norm == 0.0 {
        return 0.0;
    }
    feox_ann::dot(left, right) / (left_norm.sqrt() * right_norm.sqrt())
}

fn rank_matches(mut matches: Vec<(u64, VectorMatch)>, limit: usize) -> VectorQueryResult {
    matches.sort_by(|left, right| {
        right
            .1
            .score
            .partial_cmp(&left.1.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| right.0.cmp(&left.0))
            .then_with(|| left.1.id.cmp(&right.1.id))
    });
    matches.truncate(limit);

    VectorQueryResult {
        matches: matches.into_iter().map(|(_, item)| item).collect(),
    }
}

fn scope_key(namespace: &str, index: &str, partition: &str) -> String {
    format!("{namespace}/{index}/{partition}")
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before unix epoch")
        .as_millis() as u64
}
