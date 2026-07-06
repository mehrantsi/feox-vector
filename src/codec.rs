use crate::keys::record_id_from_key;
use crate::model::VectorMetadata;
use crate::{Result, VectorError, VectorRecord};

const MAGIC: &[u8; 4] = b"FXV1";
const HEADER_BYTES: usize = 4 + 2 + 2 + 8 + 8 + 4;

pub(crate) struct VectorAnnRecord {
    pub(crate) id: String,
    pub(crate) values: Vec<f32>,
    pub(crate) metadata: VectorMetadata,
}

pub(crate) fn encode_record(
    values: &[f32],
    metadata: &VectorMetadata,
    created_at_ms: u64,
    updated_at_ms: u64,
) -> Result<Vec<u8>> {
    let metadata_bytes = serde_json::to_vec(metadata)?;
    if metadata_bytes.len() > u32::MAX as usize {
        return Err(VectorError::Invalid(
            "metadata is too large to encode".to_string(),
        ));
    }

    let mut bytes = Vec::with_capacity(HEADER_BYTES + values.len() * 4 + metadata_bytes.len());
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&(values.len() as u16).to_le_bytes());
    bytes.extend_from_slice(&0_u16.to_le_bytes());
    bytes.extend_from_slice(&created_at_ms.to_le_bytes());
    bytes.extend_from_slice(&updated_at_ms.to_le_bytes());
    bytes.extend_from_slice(&(metadata_bytes.len() as u32).to_le_bytes());
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes.extend_from_slice(&metadata_bytes);

    Ok(bytes)
}

pub(crate) fn decode_record(key: &str, bytes: &[u8]) -> Result<VectorRecord> {
    let header = decode_header(bytes)?;

    let mut values = Vec::with_capacity(header.dimensions);
    let (chunks, _) = bytes[header.values_start..header.metadata_start].as_chunks::<4>();
    for chunk in chunks {
        values.push(f32::from_le_bytes(*chunk));
    }
    let metadata = serde_json::from_slice(&bytes[header.metadata_start..])?;

    Ok(VectorRecord {
        id: record_id_from_key(key),
        values,
        metadata,
        created_at_ms: header.created_at_ms,
        updated_at_ms: header.updated_at_ms,
    })
}

pub(crate) fn decode_ann_record(
    key: &str,
    bytes: &[u8],
    expected_dimensions: usize,
) -> Result<Option<VectorAnnRecord>> {
    let header = decode_header(bytes)?;
    if header.dimensions != expected_dimensions {
        return Ok(None);
    }

    let mut values = Vec::with_capacity(header.dimensions);
    let (chunks, _) = bytes[header.values_start..header.metadata_start].as_chunks::<4>();
    for chunk in chunks {
        values.push(f32::from_le_bytes(*chunk));
    }

    let metadata_bytes = &bytes[header.metadata_start..];
    let metadata = if metadata_bytes == b"{}" {
        VectorMetadata::default()
    } else {
        serde_json::from_slice(metadata_bytes)?
    };

    Ok(Some(VectorAnnRecord {
        id: record_id_from_key(key),
        values,
        metadata,
    }))
}

struct RecordHeader {
    dimensions: usize,
    created_at_ms: u64,
    updated_at_ms: u64,
    values_start: usize,
    metadata_start: usize,
}

fn decode_header(bytes: &[u8]) -> Result<RecordHeader> {
    if bytes.len() < HEADER_BYTES || &bytes[..4] != MAGIC {
        return Err(VectorError::Invalid(
            "stored vector record has an unsupported encoding".to_string(),
        ));
    }

    let dimensions = u16::from_le_bytes([bytes[4], bytes[5]]) as usize;
    let created_at_ms = read_u64(bytes, 8)?;
    let updated_at_ms = read_u64(bytes, 16)?;
    let metadata_len = read_u32(bytes, 24)? as usize;
    let values_start = HEADER_BYTES;
    let values_len = dimensions
        .checked_mul(4)
        .ok_or_else(|| VectorError::Invalid("stored vector dimensions overflow".to_string()))?;
    let metadata_start = values_start + values_len;
    let total_len = metadata_start
        .checked_add(metadata_len)
        .ok_or_else(|| VectorError::Invalid("stored vector metadata overflows".to_string()))?;
    if bytes.len() != total_len {
        return Err(VectorError::Invalid(
            "stored vector record has an invalid length".to_string(),
        ));
    }

    Ok(RecordHeader {
        dimensions,
        created_at_ms,
        updated_at_ms,
        values_start,
        metadata_start,
    })
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64> {
    let slice = bytes.get(offset..offset + 8).ok_or_else(|| {
        VectorError::Invalid("stored vector record is missing u64 field".to_string())
    })?;
    Ok(u64::from_le_bytes(
        slice
            .try_into()
            .expect("stored vector u64 slice has fixed length"),
    ))
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32> {
    let slice = bytes.get(offset..offset + 4).ok_or_else(|| {
        VectorError::Invalid("stored vector record is missing u32 field".to_string())
    })?;
    Ok(u32::from_le_bytes(
        slice
            .try_into()
            .expect("stored vector u32 slice has fixed length"),
    ))
}
