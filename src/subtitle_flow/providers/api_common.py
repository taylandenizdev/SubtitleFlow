"""Shared infrastructure for the real (paid) provider adapters.

This module is imported at package import time by the concrete adapters, so it
must stay dependency-free: ``httpx`` is resolved lazily, only when an adapter is
actually called with no injected client. Importing :mod:`subtitle_flow` never
opens a socket, reads a secret or loads a model.

Two concerns live here:

* **Call context and raw sink.** The pipeline binds an :class:`ApiContext`
  (durable store, validated audio, immutable config, stage/group/attempt) and a
  raw sink before dispatch and clears both on exit. Adapters refuse to run
  without them and refuse overlapping bindings, so one adapter instance can
  never apply one job's context, budget or token to another.
* **Durable paid-call budget.** Every HTTP attempt reserves a caller-supplied
  conservative upper bound in the append-only
  ``manifests/api.trace.jsonl`` ledger *before* sending. The reservation stays
  for an uncertain or successful attempt and is only released when the request
  provably never reached the provider. Remaining budget is therefore derived
  from durable history, survives new adapter objects and process resumes, and a
  corrupt ledger halts all new sends. No price is hard-coded and a local
  reservation never claims to predict the provider invoice.
"""

from __future__ import annotations

import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from email.utils import parsedate_to_datetime
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

from pydantic import ValidationError

from subtitle_flow.config import ApiLimits, ApiSettings, PaidApiPolicy
from subtitle_flow.media import AudioInfo
from subtitle_flow.providers.errors import ProviderCallError
from subtitle_flow.schemas import RawArtifactRef, StageError, StageKind
from subtitle_flow.storage import (
    RELEASE_PROOF_OUTCOME,
    ApiTraceKind,
    ApiTraceRecord,
    JobStore,
    StorageCorruptionError,
    exact_decimal_sum,
    outcome_proves_predispatch,
    utc_now,
)

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from subtitle_flow.config import PipelineConfig

__all__ = [
    "ApiBindingError",
    "ApiContext",
    "ApiExtraMissingError",
    "AttemptOutcomeLabel",
    "ExpectedSegment",
    "budget_preflight",
    "build_client",
    "canonical_endpoint",
    "classify_predispatch_transport_error",
    "parse_retry_after",
    "record_outcome",
    "refused_error",
    "release_reservation",
    "require_httpx",
    "reserve_attempt",
    "reserved_total",
    "safe_response_headers",
]


class ApiExtraMissingError(ImportError):
    """Raised when the optional ``api`` extra is not installed."""


class ApiBindingError(RuntimeError):
    """Raised when an adapter is called without its pipeline-bound context."""


class AttemptOutcomeLabel:
    """String labels stored in the ``outcome`` field of the API trace."""

    complete = "complete"
    http_error = "http_error"
    invalid_response = "invalid_response"
    remote_unknown = "remote_unknown"
    predispatch_error = RELEASE_PROOF_OUTCOME
    refused = "refused"


#: Response headers that are safe to persist: they never carry credentials and
#: are useful for provider-side correlation and retry policy.
_SAFE_RESPONSE_HEADERS = frozenset(
    {
        "request-id",
        "x-request-id",
        "x-trace-id",
        "x-goog-request-id",
        "retry-after",
        "content-type",
        "date",
    }
)


def safe_response_headers(response: Any) -> dict[str, str]:
    """Return the allow-listed response headers, never credentials/cookies."""

    headers = getattr(response, "headers", None)
    if headers is None:
        return {}
    selected: dict[str, str] = {}
    for name in _SAFE_RESPONSE_HEADERS:
        try:
            value = headers.get(name)
        except Exception:  # pragma: no cover - defensive against odd mappings
            value = None
        if value:
            selected[name] = str(value)
    return selected


def canonical_endpoint(base_url: str, path: str) -> str:
    """Join a validated canonical base URL with a fixed official path."""

    return f"{base_url.rstrip('/')}{path}"


@dataclass(frozen=True)
class ExpectedSegment:
    """The exact segment identity a bound MT dispatch is allowed to send."""

    segment_id: str
    source_text: str
    translation_input: str | None = None


@dataclass(frozen=True)
class ApiContext:
    """Explicit, per-dispatch context bound by the pipeline.

    ``config`` is the immutable job snapshot; adapters compare their typed
    adapter settings against ``config.api`` before every call so a job can never
    run with a settings object that disagrees with its stored fingerprint.
    ``expected_segments`` records the exact source segment identity (id, text and
    translation input) the request arguments must match, so a caller-supplied
    argument list can never be translated against a different bound job.
    """

    store: JobStore
    stage: StageKind
    config: "PipelineConfig"
    audio: AudioInfo | None = None
    group_id: str | None = None
    pipeline_attempt: int | None = None
    expected_segments: tuple[ExpectedSegment, ...] = ()
    source_language: str | None = None
    target_language: str | None = None

    @property
    def segment_ids(self) -> tuple[str, ...]:
        return tuple(segment.segment_id for segment in self.expected_segments)


def require_httpx() -> Any:
    """Import and return ``httpx`` or raise an actionable missing-extra error."""

    try:  # pragma: no cover - exercised via the missing-extra test
        import httpx
    except ImportError as exc:  # pragma: no cover - defensive
        raise ApiExtraMissingError(
            "httpx is required for the paid API adapters; install the optional "
            "extra with 'pip install \"subtitle-flow[api]\"' (or inject an "
            "httpx.Client explicitly)"
        ) from exc
    return httpx


def build_client(
    limits: ApiLimits,
    *,
    client: Any | None,
    client_factory: Callable[[Any, ApiLimits], Any] | None,
) -> Any:
    """Return an injected client or build a bounded one without auto-retry.

    Redirects are disabled so neither a hidden extra charge nor credential
    forwarding to another host can occur. ``httpx`` has no automatic retry, so
    every resend is explicit and therefore counted against the budget.
    """

    if client is not None:
        return client
    httpx = require_httpx()
    if client_factory is not None:
        return client_factory(httpx, limits)
    timeout = httpx.Timeout(
        connect=limits.connect_timeout_seconds,
        read=limits.read_timeout_seconds,
        write=limits.write_timeout_seconds,
        pool=limits.pool_timeout_seconds,
    )
    return httpx.Client(timeout=timeout, follow_redirects=False)


def classify_predispatch_transport_error(exc: BaseException, httpx: Any) -> bool:
    """Return ``True`` only for a proven pre-dispatch connection failure.

    ``ConnectError``/``ConnectTimeout``/``PoolTimeout`` mean the request body was
    never handed to the wire (no connection was obtained), so no remote work can
    have happened. Every other transport failure (read/write timeout, protocol
    loss after send) is treated as a possibly-dispatched, unknown outcome.
    """

    return isinstance(
        exc, (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)
    )


def parse_retry_after(value: str | None, *, now: datetime) -> float | None:
    """Parse a ``Retry-After`` header as delta-seconds or an HTTP date."""

    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        seconds = float(text)
    except ValueError:
        try:
            when = parsedate_to_datetime(text)
        except (TypeError, ValueError):
            return None
        if when is None:
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        return max(0.0, (when - now).total_seconds())
    if seconds != seconds or seconds in (float("inf"), float("-inf")):
        return None
    return max(0.0, seconds)


def refused_error(
    code: str,
    message: str,
    *,
    retryable: bool = False,
    remote_status_unknown: bool = False,
) -> ProviderCallError:
    """Build a pre-dispatch :class:`ProviderCallError` (never a silent success)."""

    return ProviderCallError(
        StageError(
            code=code,
            message=message,
            retryable=retryable,
            remote_status_unknown=remote_status_unknown,
        ),
        dispatched=False,
    )


# --------------------------------------------------------------------------- #
# Durable budget
# --------------------------------------------------------------------------- #
def reserved_total(store: JobStore, *, currency: str | None = None) -> Decimal:
    """Return the durable sum of active reservations (reserves minus releases).

    The sum is computed with exact decimal arithmetic so a long/accepted sequence
    of reservations can never be rounded across a configured cap.
    """

    amounts: list[Decimal] = []
    for record in store.read_api_trace():
        if record.kind is ApiTraceKind.reserve:
            if currency is not None and record.currency != currency:
                raise StorageCorruptionError(
                    "budget ledger mixes currencies; refusing new sends"
                )
            assert record.amount is not None
            amounts.append(Decimal(record.amount))
        elif record.kind is ApiTraceKind.release:
            assert record.amount is not None
            # ``copy_negate`` is context-free sign manipulation: unary minus
            # would round to the global Decimal precision *before*
            # ``exact_decimal_sum`` sees the value, so a release could fail to
            # cancel its reservation (or leave a spurious residual) whenever the
            # process runs under a small precision context.
            amounts.append(Decimal(record.amount).copy_negate())
    total = exact_decimal_sum(amounts)
    if total < 0:
        raise StorageCorruptionError(
            "budget ledger releases exceed reservations; refusing new sends"
        )
    return total


def _new_ids() -> tuple[str, str]:
    return secrets.token_hex(12), uuid.uuid4().hex


def budget_preflight(policy: PaidApiPolicy) -> None:
    """Validate the paid-call gate without touching any durable state.

    The policy is re-validated from its dumped form so a ``model_copy`` update
    that bypassed the field validators cannot smuggle an unsupported amount into
    a paid dispatch; such a policy becomes a typed, non-dispatched refusal
    instead of an invented remote-unknown outcome.
    """

    if isinstance(policy, PaidApiPolicy):
        try:
            policy = PaidApiPolicy.model_validate(policy.model_dump())
        except ValidationError as exc:
            raise refused_error(
                "PAID_POLICY_INVALID",
                "the paid-call policy is not valid; refusing to send",
            ) from exc
    if not policy.allow_paid_api_calls:
        raise refused_error(
            "PAID_CALLS_DISABLED",
            "paid API calls are disabled; set allow_paid_api_calls with explicit "
            "reservation caps and currency to opt in",
        )
    if not policy.configured:
        raise refused_error(
            "PAID_POLICY_INCOMPLETE",
            "paid API calls require total_reservation_cap, per_call_cap, "
            "per_call_upper_bound and currency",
        )
    assert policy.per_call_upper_bound is not None
    assert policy.per_call_cap is not None
    assert policy.total_reservation_cap is not None
    if policy.per_call_upper_bound > policy.per_call_cap:
        raise refused_error(
            "PAID_PER_CALL_LIMIT",
            "per_call_upper_bound exceeds the configured per-call cap",
        )


def reserve_attempt(
    store: JobStore,
    policy: PaidApiPolicy,
    *,
    stage: StageKind,
    group_id: str | None,
    pipeline_attempt: int | None,
    http_attempt: int,
    provider: str,
    model: str,
    now: Callable[[], datetime] = utc_now,
) -> tuple[str, str]:
    """Durably reserve one attempt's upper bound before any send.

    Returns ``(attempt_id, amount)``. Refuses with a ``ProviderCallError``
    before any reservation or HTTP when the policy is disabled/incomplete or
    the cumulative reservation would exceed the configured total cap.
    """

    budget_preflight(policy)
    assert policy.per_call_upper_bound is not None
    assert policy.total_reservation_cap is not None
    assert policy.currency is not None
    amount = policy.per_call_upper_bound
    consumed = reserved_total(store, currency=policy.currency)
    prospective = exact_decimal_sum([consumed, amount])
    if prospective > policy.total_reservation_cap:
        raise refused_error(
            "PAID_TOTAL_LIMIT",
            "reserving the next call would exceed the configured total "
            "reservation cap; the next/current request was not sent, and prior "
            "durable reservations from this job may already exist",
        )
    attempt_id, record_id = _new_ids()
    store.append_api_trace(
        ApiTraceRecord(
            record_id=record_id,
            attempt_id=attempt_id,
            kind=ApiTraceKind.reserve,
            stage=stage,
            group_id=group_id,
            pipeline_attempt=pipeline_attempt,
            http_attempt=http_attempt,
            provider=provider,
            model=model,
            amount=str(amount),
            currency=policy.currency,
            recorded_at_utc=now(),
        )
    )
    return attempt_id, str(amount)


def _require_release_candidate(
    store: JobStore,
    *,
    attempt_id: str,
    stage: StageKind,
    group_id: str | None,
    pipeline_attempt: int | None,
    http_attempt: int,
    provider: str,
    model: str,
    amount: str,
    currency: str,
) -> None:
    """Refuse a release whose durable evidence does not prove a non-dispatch.

    The whole ledger is re-read (and therefore re-validated) before the release
    is appended, so a semantic misuse is refused *before* a corrupt release line
    is written instead of being detected only on the next read.
    """

    records = store.read_api_trace()
    reserve: ApiTraceRecord | None = None
    outcome: ApiTraceRecord | None = None
    for record in records:
        if record.attempt_id != attempt_id:
            continue
        if record.kind is ApiTraceKind.reserve:
            reserve = record
        elif record.kind is ApiTraceKind.outcome:
            outcome = record
        elif record.kind is ApiTraceKind.release:
            raise StorageCorruptionError(
                "a release already exists for this attempt; refusing a duplicate"
            )
    if reserve is None:
        raise StorageCorruptionError(
            "api ledger release has no matching reservation"
        )
    if outcome is None:
        raise StorageCorruptionError(
            "api ledger release is not anchored by a recorded outcome"
        )
    if not outcome_proves_predispatch(outcome):
        raise StorageCorruptionError(
            "release requires a proven pre-dispatch outcome with no HTTP status "
            "and no archived raw body; refusing to erase a possibly-dispatched "
            "paid attempt from the budget"
        )
    if (
        reserve.stage,
        reserve.group_id,
        reserve.pipeline_attempt,
        reserve.http_attempt,
        reserve.provider,
        reserve.model,
        reserve.currency,
    ) != (
        stage,
        group_id,
        pipeline_attempt,
        http_attempt,
        provider,
        model,
        currency,
    ):
        raise StorageCorruptionError(
            "release identity/currency does not match its reservation"
        )
    assert reserve.amount is not None
    if Decimal(amount) > Decimal(reserve.amount):
        raise StorageCorruptionError("release exceeds its reservation")


def release_reservation(
    store: JobStore,
    *,
    attempt_id: str,
    stage: StageKind,
    group_id: str | None,
    pipeline_attempt: int | None,
    http_attempt: int,
    provider: str,
    model: str,
    amount: str,
    currency: str,
    reason: str,
    now: Callable[[], datetime] = utc_now,
) -> None:
    """Release a reservation for an attempt that provably was never sent.

    Only a recorded ``predispatch_error`` outcome with no HTTP status and no
    archived raw body permits a release; a successful, errored or unknown
    outcome is refused. The prior evidence is never rewritten.
    """

    _require_release_candidate(
        store,
        attempt_id=attempt_id,
        stage=stage,
        group_id=group_id,
        pipeline_attempt=pipeline_attempt,
        http_attempt=http_attempt,
        provider=provider,
        model=model,
        amount=amount,
        currency=currency,
    )
    _, record_id = _new_ids()
    store.append_api_trace(
        ApiTraceRecord(
            record_id=record_id,
            attempt_id=attempt_id,
            kind=ApiTraceKind.release,
            stage=stage,
            group_id=group_id,
            pipeline_attempt=pipeline_attempt,
            http_attempt=http_attempt,
            provider=provider,
            model=model,
            amount=amount,
            currency=currency,
            error_message=reason,
            recorded_at_utc=now(),
        )
    )


def record_outcome(
    store: JobStore,
    *,
    attempt_id: str,
    stage: StageKind,
    group_id: str | None,
    pipeline_attempt: int | None,
    http_attempt: int,
    provider: str,
    model: str,
    outcome: str,
    now: Callable[[], datetime] = utc_now,
    http_status: int | None = None,
    remote_status_unknown: bool = False,
    request_id: str | None = None,
    raw_reference: RawArtifactRef | None = None,
    response_headers: Mapping[str, str] | None = None,
    retry_after_seconds: float | None = None,
    error_code: str | None = None,
    error_message: str | None = None,
    elapsed_ms: int | None = None,
) -> None:
    """Append the durable outcome of one HTTP attempt (success or failure)."""

    _, record_id = _new_ids()
    store.append_api_trace(
        ApiTraceRecord(
            record_id=record_id,
            attempt_id=attempt_id,
            kind=ApiTraceKind.outcome,
            stage=stage,
            group_id=group_id,
            pipeline_attempt=pipeline_attempt,
            http_attempt=http_attempt,
            provider=provider,
            model=model,
            outcome=outcome,
            http_status=http_status,
            remote_status_unknown=remote_status_unknown,
            request_id=request_id,
            raw_reference=raw_reference,
            response_headers=dict(response_headers or {}),
            retry_after_seconds=retry_after_seconds,
            error_code=error_code,
            error_message=error_message,
            elapsed_ms=elapsed_ms,
            recorded_at_utc=now(),
        )
    )


def monotonic_ms(start: float, monotonic: Callable[[], float]) -> int:
    return max(0, int((monotonic() - start) * 1000))


__all__ += ["monotonic_ms"]
