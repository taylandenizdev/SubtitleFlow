"""Structured provider-call failures for Phase 2 adapters.

Phase 1 contracts return ``StageError`` records for representable failures. Real
adapters additionally need to tell the pipeline whether a failed call may have
reached the remote service, because replaying an in-flight paid request can bill
twice. :class:`ProviderCallError` is the small, explicit carrier for that state.

Rules the pipeline relies on:

* ``dispatched=False`` means the adapter can prove the request never left the
  process (for example a validation failure or a connection error before the
  request body was sent). Such a call is :attr:`safely_resumable`.
* ``dispatched=True`` means bytes may have reached the provider. It is only
  :attr:`remote_status_unknown` when the adapter cannot establish the remote
  outcome; a definitive rejection (for example an authorization or malformed
  request error) stays known and must set ``remote_status_unknown=False``.
* The embedded :class:`~subtitle_flow.schemas.StageError` keeps the code,
  message, retryability and remote-uncertainty flag that are persisted.

Adapters must never invent a success or downgrade an unknown remote outcome to a
safe retry. Unknown remote status is preserved until the caller decides.
"""

from __future__ import annotations

from subtitle_flow.schemas import StageError

__all__ = ["ProviderCallError"]


class ProviderCallError(Exception):
    """A provider call that failed with an explicit dispatch/uncertainty state.

    Args:
        error: The structured :class:`StageError` describing the failure.
        dispatched: ``True`` when the request may have reached the provider.
        cause: Optional original exception, preserved for diagnostics.

    Raises:
        TypeError: when ``error`` is not a ``StageError``.
    """

    def __init__(
        self,
        error: StageError,
        *,
        dispatched: bool,
        cause: BaseException | None = None,
    ) -> None:
        if not isinstance(error, StageError):
            raise TypeError(
                f"error must be a StageError, got {type(error).__name__}"
            )
        self.error = error
        self.dispatched = bool(dispatched)
        self.cause = cause
        super().__init__(f"{error.code}: {error.message}")

    @property
    def remote_status_unknown(self) -> bool:
        """``True`` only when the remote outcome cannot be established."""

        return self.error.remote_status_unknown

    @property
    def retryable(self) -> bool:
        """Whether the embedded :class:`StageError` marks the failure retryable."""

        return self.error.retryable

    @property
    def safely_resumable(self) -> bool:
        """``True`` only when no remote work can have happened *and* the error is retryable.

        A pre-dispatch failure that is explicitly retryable may be resumed
        automatically. A pre-dispatch failure that is *not* retryable, or any
        failure that may have reached the provider, is refused by an ordinary
        resume and needs an explicit ``allow_remote_retry`` decision. Callers
        must never downgrade a known or unknown remote outcome to a safe retry.
        """

        return (
            not self.dispatched
            and not self.error.remote_status_unknown
            and self.error.retryable
        )

    @property
    def auto_resumable(self) -> bool:
        """Alias of :attr:`safely_resumable` for pipeline policy."""

        return self.safely_resumable
