use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use serde_json::Value;

use crate::model::VectorMetadata;

#[derive(Default)]
pub(crate) struct MetadataFilterIndex {
    ids: HashSet<Arc<str>>,
    facets: HashMap<String, HashSet<Arc<str>>>,
}

pub(crate) enum CompiledMetadataFilter<'a> {
    All,
    None,
    Plan(MetadataFilterPlan<'a>),
}

pub(crate) struct MetadataFilterPlan<'a> {
    ids: &'a HashSet<Arc<str>>,
    includes: Vec<AnySet<'a>>,
    excludes: Vec<&'a HashSet<Arc<str>>>,
}

struct AnySet<'a> {
    sets: Vec<&'a HashSet<Arc<str>>>,
}

impl MetadataFilterIndex {
    pub(crate) fn insert(&mut self, id: &str, metadata: &VectorMetadata) {
        let id = self.intern_id(id);
        for (key, value) in metadata {
            insert_value_tokens(&mut self.facets, &id, key, value);
        }
    }

    fn intern_id(&mut self, id: &str) -> Arc<str> {
        if let Some(existing) = self.ids.get(id) {
            return Arc::clone(existing);
        }
        let id = Arc::<str>::from(id);
        self.ids.insert(Arc::clone(&id));
        id
    }

    pub(crate) fn compile<'a>(&'a self, filter: &VectorMetadata) -> CompiledMetadataFilter<'a> {
        if filter.is_empty() {
            return CompiledMetadataFilter::All;
        }

        let mut includes = Vec::new();
        let mut excludes = Vec::new();
        for (key, expected) in filter {
            if !self.compile_filter_value(key, expected, &mut includes, &mut excludes) {
                return CompiledMetadataFilter::None;
            }
        }

        if includes.is_empty() && excludes.is_empty() {
            CompiledMetadataFilter::All
        } else {
            CompiledMetadataFilter::Plan(MetadataFilterPlan {
                ids: &self.ids,
                includes,
                excludes,
            })
        }
    }

    fn compile_filter_value<'a>(
        &'a self,
        key: &str,
        expected: &Value,
        includes: &mut Vec<AnySet<'a>>,
        excludes: &mut Vec<&'a HashSet<Arc<str>>>,
    ) -> bool {
        let Some(object) = expected.as_object() else {
            return self.push_include(key, expected, includes);
        };
        if object.is_empty() {
            return false;
        }

        for (operator, value) in object {
            match operator.as_str() {
                "$eq" => {
                    if !self.push_include(key, value, includes) {
                        return false;
                    }
                }
                "$in" => {
                    if !self.push_include_any(key, value, includes) {
                        return false;
                    }
                }
                "$nin" => {
                    if !self.push_exclude_any(key, value, excludes) {
                        return false;
                    }
                }
                _ => return false,
            }
        }
        true
    }

    fn push_include<'a>(
        &'a self,
        key: &str,
        value: &Value,
        includes: &mut Vec<AnySet<'a>>,
    ) -> bool {
        let mut sets = Vec::new();
        for token in value_tokens(key, value) {
            if let Some(set) = self.facets.get(&token) {
                sets.push(set);
            }
        }
        if sets.is_empty() {
            return false;
        }
        includes.push(AnySet { sets });
        true
    }

    fn push_include_any<'a>(
        &'a self,
        key: &str,
        value: &Value,
        includes: &mut Vec<AnySet<'a>>,
    ) -> bool {
        let Some(values) = value.as_array() else {
            return false;
        };
        let mut sets = Vec::new();
        for value in values {
            for token in value_tokens(key, value) {
                if let Some(set) = self.facets.get(&token) {
                    sets.push(set);
                }
            }
        }
        if sets.is_empty() {
            return false;
        }
        includes.push(AnySet { sets });
        true
    }

    fn push_exclude_any<'a>(
        &'a self,
        key: &str,
        value: &Value,
        excludes: &mut Vec<&'a HashSet<Arc<str>>>,
    ) -> bool {
        let Some(values) = value.as_array() else {
            return false;
        };
        for value in values {
            for token in value_tokens(key, value) {
                if let Some(set) = self.facets.get(&token) {
                    excludes.push(set);
                }
            }
        }
        true
    }
}

impl CompiledMetadataFilter<'_> {
    pub(crate) fn accepts(&self, id: &str) -> bool {
        match self {
            Self::All => true,
            Self::None => false,
            Self::Plan(plan) => plan.accepts(id),
        }
    }

    pub(crate) fn is_none(&self) -> bool {
        matches!(self, Self::None)
    }

    pub(crate) fn is_all(&self) -> bool {
        matches!(self, Self::All)
    }

    pub(crate) fn candidate_ids(&self, max: usize) -> Option<Vec<&str>> {
        match self {
            Self::All | Self::None => None,
            Self::Plan(plan) => plan.candidate_ids(max),
        }
    }
}

impl MetadataFilterPlan<'_> {
    fn accepts(&self, id: &str) -> bool {
        for include in &self.includes {
            if !include.sets.iter().any(|set| set.contains(id)) {
                return false;
            }
        }
        for exclude in &self.excludes {
            if exclude.contains(id) {
                return false;
            }
        }
        true
    }

    fn candidate_ids(&self, max: usize) -> Option<Vec<&str>> {
        let estimate = self.estimated_len();
        if estimate > max {
            return None;
        }

        let mut ids = Vec::with_capacity(estimate);
        let mut seen = HashSet::new();
        if let Some(base) = self.base_include() {
            for set in &base.sets {
                for id in *set {
                    let id = id.as_ref();
                    if seen.insert(id) && self.accepts(id) {
                        ids.push(id);
                        if ids.len() > max {
                            return None;
                        }
                    }
                }
            }
            return Some(ids);
        }

        for id in self.ids {
            let id = id.as_ref();
            if self.accepts(id) {
                ids.push(id);
                if ids.len() > max {
                    return None;
                }
            }
        }
        Some(ids)
    }

    fn estimated_len(&self) -> usize {
        self.base_include()
            .map(AnySet::estimated_len)
            .unwrap_or(self.ids.len())
    }

    fn base_include(&self) -> Option<&AnySet<'_>> {
        self.includes
            .iter()
            .min_by_key(|include| include.estimated_len())
    }
}

impl AnySet<'_> {
    fn estimated_len(&self) -> usize {
        self.sets.iter().map(|set| set.len()).sum()
    }
}

pub(crate) fn metadata_matches_filter(metadata: &VectorMetadata, filter: &VectorMetadata) -> bool {
    filter
        .iter()
        .all(|(key, expected)| value_matches_filter(metadata.get(key), expected))
}

fn value_matches_filter(actual: Option<&Value>, expected: &Value) -> bool {
    let Some(object) = expected.as_object() else {
        return actual
            .map(|actual| value_contains(actual, expected))
            .unwrap_or(false);
    };
    if object.is_empty() {
        return false;
    }

    object
        .iter()
        .all(|(operator, value)| match operator.as_str() {
            "$eq" => actual
                .map(|actual| value_contains(actual, value))
                .unwrap_or(false),
            "$in" => value
                .as_array()
                .map(|values| {
                    actual
                        .map(|actual| values.iter().any(|value| value_contains(actual, value)))
                        .unwrap_or(false)
                })
                .unwrap_or(false),
            "$nin" => value
                .as_array()
                .map(|values| {
                    actual
                        .map(|actual| !values.iter().any(|value| value_contains(actual, value)))
                        .unwrap_or(true)
                })
                .unwrap_or(false),
            _ => false,
        })
}

fn value_contains(actual: &Value, expected: &Value) -> bool {
    actual == expected
        || actual
            .as_array()
            .map(|values| values.iter().any(|value| value == expected))
            .unwrap_or(false)
}

fn insert_value_tokens(
    facets: &mut HashMap<String, HashSet<Arc<str>>>,
    id: &Arc<str>,
    key: &str,
    value: &Value,
) {
    for token in value_tokens(key, value) {
        facets.entry(token).or_default().insert(Arc::clone(id));
    }
}

fn value_tokens(key: &str, value: &Value) -> Vec<String> {
    match value {
        Value::Array(values) => values
            .iter()
            .filter_map(|value| scalar_token(key, value))
            .collect(),
        _ => scalar_token(key, value).into_iter().collect(),
    }
}

fn scalar_token(key: &str, value: &Value) -> Option<String> {
    match value {
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => {
            serde_json::to_string(value)
                .ok()
                .map(|encoded| format!("{key}\u{1f}{encoded}"))
        }
        Value::Array(_) | Value::Object(_) => None,
    }
}
