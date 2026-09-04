"""Backend-agnostic access to the raw MCAP source dataset: HF Hub, S3, or GCS.

training/config.py's DataConfig.source_uri identifies the dataset by URI:
    hf://datasets/<repo_id>   (Hugging Face Hub -- gated repos need HF_TOKEN)
    s3://<bucket>/<prefix>
    gs://<bucket>/<prefix>

Listing and download route hf:// sources through huggingface_hub's own
HfApi/hf_hub_download rather than the generic fsspec path used for
streaming reads (data/decode.py) and for S3/GCS -- proven fast and correct
against the real XDOF/ABC-130k repo. S3/GCS are not independently verified
against a real bucket (no credentials available in this environment).
"""
from __future__ import annotations

import os


def is_hf_uri(source_uri: str) -> bool:
    return source_uri.startswith("hf://")


def parse_hf_uri(source_uri: str) -> tuple[str, str]:
    """"hf://datasets/XDOF/ABC-130k" -> ("dataset", "XDOF/ABC-130k").
    "hf://some/model" -> ("model", "some/model")."""
    if not is_hf_uri(source_uri):
        raise ValueError(f"not an hf:// URI: {source_uri!r}")
    rest = source_uri[len("hf://"):]
    for prefix, repo_type in (("datasets/", "dataset"), ("spaces/", "space")):
        if rest.startswith(prefix):
            return repo_type, rest[len(prefix):]
    return "model", rest


def requires_hf_token(source_uri: str) -> bool:
    return is_hf_uri(source_uri)


def open_fs(source_uri: str, hf_token: str | None = None, revision: str | None = None):
    """Returns (fs, fs_root) via fsspec. hf_token is forwarded only for
    hf:// URIs; s3://gs:// sources rely on ambient credentials (the standard
    boto3/gcloud credential chain). revision pins an hf:// URI to a specific
    commit/branch/tag via the "hf://datasets/org/repo@revision" syntax."""
    import fsspec
    if is_hf_uri(source_uri) and revision:
        source_uri = f"{source_uri}@{revision}"
    if is_hf_uri(source_uri) and hf_token:
        return fsspec.core.url_to_fs(source_uri, token=hf_token)
    return fsspec.core.url_to_fs(source_uri)


def default_local_cache_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "..", ".cache", "downloads")


def _reject_unsafe_rel_path(rel_path: str) -> None:
    """rel_path comes from a remote listing -- guard against path traversal."""
    if os.path.isabs(rel_path):
        raise ValueError(f"refusing to download an absolute rel_path: {rel_path!r}")
    normalized = os.path.normpath(rel_path)
    if normalized == os.pardir or normalized.startswith(os.pardir + os.sep):
        raise ValueError(f"rel_path {rel_path!r} escapes its intended directory -- refusing")


def download_one(
    source_uri: str, rel_path: str, token: str | None = None, cache_dir: str | None = None,
    revision: str | None = None,
) -> str:
    """Download rel_path from source_uri's backend to local disk, returning
    the local path. hf:// uses hf_hub_download (proven local caching, reused
    automatically on a repeat call). s3://gs:// use fsspec's fs.get() into
    cache_dir, skipped if the target file already exists there."""
    _reject_unsafe_rel_path(rel_path)

    if is_hf_uri(source_uri):
        from huggingface_hub import hf_hub_download
        repo_type, repo_id = parse_hf_uri(source_uri)
        return hf_hub_download(
            repo_id=repo_id, repo_type=repo_type, filename=rel_path, token=token, cache_dir=cache_dir,
            revision=revision,
        )

    fs, fs_root = open_fs(source_uri)
    base_dir = cache_dir or default_local_cache_dir()
    local_path = os.path.join(base_dir, rel_path)
    resolved_base = os.path.realpath(base_dir)
    resolved_target = os.path.realpath(local_path)
    if os.path.commonpath([resolved_base, resolved_target]) != resolved_base:
        raise ValueError(f"rel_path {rel_path!r} would resolve outside {base_dir!r} -- refusing")
    if not os.path.exists(local_path):
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        fs.get(f"{fs_root}/{rel_path}", local_path)
    return local_path
