use std::path::PathBuf;
use std::sync::Arc;

use feoxdb::{FeoxError, FeoxStore};

#[derive(Debug, Clone)]
pub struct StoreConfig {
    pub device_path: Option<PathBuf>,
    pub file_size: Option<u64>,
    pub max_memory: Option<usize>,
}

impl Default for StoreConfig {
    fn default() -> Self {
        Self {
            device_path: None,
            file_size: None,
            max_memory: Some(512 * 1024 * 1024),
        }
    }
}

#[derive(Debug, Clone)]
pub struct KeyValue {
    pub key: String,
    pub value: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct StoreStats {
    pub records: usize,
    pub memory_bytes: usize,
}

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("feox error: {0}")]
    Feox(#[from] FeoxError),
    #[error("stored key is not utf-8")]
    InvalidKey,
}

pub type Result<T> = std::result::Result<T, StoreError>;

#[derive(Clone)]
pub struct Store {
    inner: Arc<FeoxStore>,
}

impl Store {
    pub fn open(config: StoreConfig) -> Result<Self> {
        let mut builder = FeoxStore::builder();

        if let Some(path) = config.device_path {
            builder = builder.device_path(path.to_string_lossy().to_string());
        }
        if let Some(file_size) = config.file_size {
            builder = builder.file_size(file_size);
        }
        if let Some(max_memory) = config.max_memory {
            builder = builder.max_memory(max_memory);
        } else {
            builder = builder.no_memory_limit();
        }

        Ok(Self {
            inner: Arc::new(builder.build()?),
        })
    }

    pub fn put_bytes(&self, key: &str, value: &[u8]) -> Result<bool> {
        Ok(self.inner.insert(key.as_bytes(), value)?)
    }

    pub fn get_bytes(&self, key: &str) -> Result<Option<Vec<u8>>> {
        match self.inner.get(key.as_bytes()) {
            Ok(value) => Ok(Some(value)),
            Err(FeoxError::KeyNotFound) => Ok(None),
            Err(error) => Err(StoreError::Feox(error)),
        }
    }

    pub fn delete(&self, key: &str) -> Result<bool> {
        match self.inner.delete(key.as_bytes()) {
            Ok(()) => Ok(true),
            Err(FeoxError::KeyNotFound) => Ok(false),
            Err(error) => Err(StoreError::Feox(error)),
        }
    }

    pub fn list_prefix(&self, prefix: &str, limit: usize) -> Result<Vec<KeyValue>> {
        self.list_prefix_after(prefix, None, limit)
    }

    pub fn list_prefix_after(
        &self,
        prefix: &str,
        start_after: Option<&str>,
        limit: usize,
    ) -> Result<Vec<KeyValue>> {
        if limit == 0 {
            return Ok(Vec::new());
        }

        let start = match start_after {
            Some(key) => {
                let mut value = key.as_bytes().to_vec();
                value.push(0);
                value
            }
            None => prefix.as_bytes().to_vec(),
        };
        let mut end = prefix.as_bytes().to_vec();
        end.push(0xff);

        let pairs = self
            .inner
            .range_query(&start, &end, limit)?
            .into_iter()
            .filter(|(key, _)| key.starts_with(prefix.as_bytes()))
            .map(|(key, value)| {
                let key = String::from_utf8(key).map_err(|_| StoreError::InvalidKey)?;
                Ok(KeyValue { key, value })
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(pairs)
    }

    pub fn stats(&self) -> StoreStats {
        StoreStats {
            records: self.inner.len(),
            memory_bytes: self.inner.memory_usage(),
        }
    }
}
