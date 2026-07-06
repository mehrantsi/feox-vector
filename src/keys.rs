pub(crate) fn record_prefix(namespace: &str, index: &str, partition: &str) -> String {
    format!("vector/{}/{}/{}/record/", namespace, index, partition)
}

pub(crate) fn record_key(namespace: &str, index: &str, partition: &str, id: &str) -> String {
    format!("{}{}", record_prefix(namespace, index, partition), id)
}

pub(crate) fn record_id_from_key(key: &str) -> String {
    key.rsplit('/').next().unwrap_or_default().to_string()
}
