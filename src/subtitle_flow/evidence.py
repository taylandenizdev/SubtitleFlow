"""Cloud-neutral durable raw-evidence helpers shared by the whole product.

A grouped provider response may archive one raw body per item and a review
re-run archives one body per region; the primary group/transcript reference only
points at the *last* of them. Every body is independently registered by
:meth:`subtitle_flow.storage.JobStore.archive_raw` and must be independently
re-verified on every reuse, export and comparison path.
:func:`verify_supplementary_raw` is that read-only check; it fails closed on a
deleted, tampered, fabricated or cross-attempt body.

The module is import-light (no ``httpx``), opens no socket, reads no secret and
loads no model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from subtitle_flow.schemas import RawArtifactRef, StageKind
from subtitle_flow.storage import JobStore, StorageCorruptionError

__all__ = [
    "collect_supplementary_refs",
    "verify_supplementary_raw",
]


def _reference_from_entry(entry: object) -> RawArtifactRef | None:
    if not isinstance(entry, Mapping):
        return None
    raw = entry.get("raw_reference")
    if isinstance(raw, Mapping):
        try:
            return RawArtifactRef.model_validate(dict(raw))
        except ValueError as exc:  # pragma: no cover - defensive
            raise StorageCorruptionError(
                "a supplementary raw reference is not a valid RawArtifactRef"
            ) from exc
    if entry.get("raw_locator") is not None:
        raise StorageCorruptionError(
            "a supplementary raw entry records a locator but no verifiable "
            "reference; refusing to trust unchecked metadata"
        )
    return None


def collect_supplementary_refs(
    stage: StageKind, metadata: object
) -> tuple[RawArtifactRef, ...]:
    """Extract every supplementary raw reference recorded in adapter metadata.

    For MT this is one reference per individual provider response in the group
    (``request_metadata["requests"]``); for STT it is one reference per review
    re-run body (``request_metadata["review_reruns"]``). An entry that claims a
    raw body but cannot produce a verifiable reference is corruption.
    """

    if not isinstance(metadata, Mapping):
        return ()
    if stage is StageKind.mt:
        entries = metadata.get("requests")
    elif stage is StageKind.stt:
        entries = metadata.get("review_reruns")
    else:  # pragma: no cover - StageKind is exhaustive
        return ()
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        return ()
    refs: list[RawArtifactRef] = []
    for entry in entries:
        reference = _reference_from_entry(entry)
        if reference is not None:
            refs.append(reference)
    return tuple(refs)


def verify_supplementary_raw(
    store: JobStore,
    references: Sequence[RawArtifactRef],
    *,
    stage: StageKind,
    group_id: str | None,
    attempt: int | None,
) -> None:
    """Read-only proof that every supplementary body is registered and intact.

    Fails closed (``StorageCorruptionError``) on a deleted, tampered,
    fabricated or cross-attempt/-group body. The primary ``raw_reference`` is
    not enough: every earlier grouped response and every review re-run must be
    independently proven on reuse, export and comparison.
    """

    for reference in references:
        store.verify_raw(
            reference, stage=stage, group_id=group_id, attempt=attempt
        )
