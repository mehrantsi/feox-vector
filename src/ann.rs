use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use arc_swap::ArcSwapOption;
use feox_ann::AnnIndex;
use scc::HashIndex;

use crate::filter::MetadataFilterIndex;

pub(crate) struct AnnSnapshot {
    pub(crate) index: AnnIndex,
    pub(crate) filters: MetadataFilterIndex,
}

#[derive(Default)]
pub(crate) struct AnnRegistry {
    scopes: HashIndex<String, Arc<AnnScope>>,
}

impl AnnRegistry {
    pub(crate) fn scope(&self, key: &str) -> Arc<AnnScope> {
        if let Some(scope) = self.scopes.peek_with(key, |_, scope| scope.clone()) {
            return scope;
        }
        self.scopes
            .entry(key.to_string())
            .or_insert_with(|| Arc::new(AnnScope::default()))
            .get()
            .clone()
    }

    pub(crate) fn mark_dirty_if_present(&self, key: &str) {
        if let Some(scope) = self.scopes.peek_with(key, |_, scope| scope.clone()) {
            scope.mark_dirty();
        }
    }
}

pub(crate) struct AnnScope {
    snapshot: ArcSwapOption<AnnSnapshot>,
    dirty: AtomicBool,
    rebuilding: AtomicBool,
}

impl Default for AnnScope {
    fn default() -> Self {
        Self {
            snapshot: ArcSwapOption::from(None),
            dirty: AtomicBool::new(false),
            rebuilding: AtomicBool::new(false),
        }
    }
}

impl AnnScope {
    pub(crate) fn snapshot(&self) -> Option<Arc<AnnSnapshot>> {
        self.snapshot.load_full()
    }

    pub(crate) fn query_snapshot(&self) -> Option<Arc<AnnSnapshot>> {
        if self.requires_exact_query() {
            return None;
        }
        let snapshot = self.snapshot.load_full();
        if self.requires_exact_query() {
            return None;
        }
        snapshot
    }

    pub(crate) fn mark_dirty(&self) {
        self.dirty.store(true, Ordering::Release);
    }

    fn requires_exact_query(&self) -> bool {
        self.dirty.load(Ordering::Acquire) || self.rebuilding.load(Ordering::Acquire)
    }

    pub(crate) fn begin_rebuild(&self) -> bool {
        if self
            .rebuilding
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            self.dirty.store(false, Ordering::Release);
            return true;
        }
        false
    }

    pub(crate) fn publish(&self, snapshot: AnnSnapshot) {
        self.snapshot.store(Some(Arc::new(snapshot)));
        self.rebuilding.store(false, Ordering::Release);
    }

    pub(crate) fn finish_failed_rebuild(&self) {
        self.dirty.store(true, Ordering::Release);
        self.rebuilding.store(false, Ordering::Release);
    }
}
