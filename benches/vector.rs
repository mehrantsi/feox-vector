use std::collections::BTreeMap;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use feox_vector::{
    Store, StoreConfig, VectorQueryInput, VectorQueryMode, VectorStore, VectorUpsertInput,
    VectorUpsertRecord,
};
use serde_json::json;

const DIMENSIONS: usize = 128;
const RECORDS: usize = 10_000;
const UPSERT_BATCH: usize = 100;

struct XorShift(u64);

impl XorShift {
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / (1 << 24) as f32) * 2.0 - 1.0
    }

    fn vector(&mut self, dimensions: usize) -> Vec<f32> {
        (0..dimensions).map(|_| self.next_f32()).collect()
    }
}

fn populated_store(records: usize) -> VectorStore {
    let store = Store::open(StoreConfig::default()).unwrap();
    let vectors = VectorStore::new(store);
    let mut rng = XorShift(0x5eed_f0f5 ^ records as u64);

    let mut batch = Vec::with_capacity(UPSERT_BATCH);
    for i in 0..records {
        let mut metadata = BTreeMap::new();
        metadata.insert("shard".to_string(), json!(format!("shard-{}", i % 8)));
        batch.push(VectorUpsertRecord {
            id: format!("record-{i:06}"),
            values: rng.vector(DIMENSIONS),
            metadata,
        });
        if batch.len() == UPSERT_BATCH {
            vectors
                .upsert(
                    "bench",
                    "docs",
                    "main",
                    VectorUpsertInput {
                        records: std::mem::take(&mut batch),
                    },
                )
                .unwrap();
        }
    }
    vectors
}

fn query_input(vector: Vec<f32>, mode: Option<VectorQueryMode>) -> VectorQueryInput {
    VectorQueryInput {
        vector,
        top_k: Some(10),
        filter: BTreeMap::new(),
        min_score: None,
        mode,
        ef_search: None,
        candidate_limit: None,
    }
}

fn bench_upsert(c: &mut Criterion) {
    let mut group = c.benchmark_group("upsert");
    group.sample_size(10);
    group.throughput(Throughput::Elements(RECORDS as u64));
    group.bench_function(BenchmarkId::new("batched", RECORDS), |b| {
        b.iter(|| populated_store(RECORDS));
    });
    group.finish();
}

fn bench_query(c: &mut Criterion) {
    let vectors = populated_store(RECORDS);
    vectors
        .rebuild_ann("bench", "docs", "main", DIMENSIONS)
        .unwrap();
    let mut rng = XorShift(0xfeed_beef);
    let queries: Vec<Vec<f32>> = (0..256).map(|_| rng.vector(DIMENSIONS)).collect();

    let mut group = c.benchmark_group("query");
    group.throughput(Throughput::Elements(1));
    group.bench_function("exact_top10", |b| {
        let mut cursor = 0;
        b.iter(|| {
            let query = queries[cursor % queries.len()].clone();
            cursor += 1;
            vectors
                .query("bench", "docs", "main", query_input(query, None))
                .unwrap()
        });
    });
    group.bench_function("ann_top10", |b| {
        let mut cursor = 0;
        b.iter(|| {
            let query = queries[cursor % queries.len()].clone();
            cursor += 1;
            vectors
                .query(
                    "bench",
                    "docs",
                    "main",
                    query_input(query, Some(VectorQueryMode::Ann)),
                )
                .unwrap()
        });
    });
    group.bench_function("ann_top10_filtered", |b| {
        let mut cursor = 0;
        b.iter(|| {
            let query = queries[cursor % queries.len()].clone();
            cursor += 1;
            let mut input = query_input(query, Some(VectorQueryMode::Ann));
            input.filter.insert(
                "shard".to_string(),
                json!({ "$in": ["shard-0", "shard-1"] }),
            );
            vectors.query("bench", "docs", "main", input).unwrap()
        });
    });
    group.finish();
}

criterion_group!(benches, bench_upsert, bench_query);
criterion_main!(benches);
