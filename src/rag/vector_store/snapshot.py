from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag.vector_store.store import QdrantStore


async def trigger_snapshot(store: QdrantStore) -> str:
    """Trigger a Qdrant collection snapshot and return the snapshot name."""
    return await store.snapshot()


async def upload_to_s3(
    snapshot_name: str,
    bucket: str,
    prefix: str = "qdrant-snapshots/",
    region: str = "us-east-1",
) -> None:
    """Upload a Qdrant snapshot to S3. Requires boto3."""
    raise NotImplementedError(
        "S3 upload requires boto3. Install it and implement this function "
        "with: boto3.client('s3').upload_file(snapshot_path, bucket, prefix + snapshot_name)"
    )


async def upload_to_gcs(
    snapshot_name: str,
    bucket: str,
    prefix: str = "qdrant-snapshots/",
) -> None:
    """Upload a Qdrant snapshot to GCS. Requires google-cloud-storage."""
    raise NotImplementedError(
        "GCS upload requires google-cloud-storage. Install it and implement this function "
        "with: storage.Client().bucket(bucket).blob(prefix + snapshot_name).upload_from_filename(...)"
    )
