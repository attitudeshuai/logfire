"""Runtime detection of spans that stay open for too long.

The SDK already keeps a weak table (``OPEN_SPANS``) of every currently open span so that spans
lingering in suspended generators can be ended when the process exits. This module adds an optional
layer on top of that: when configured with `OpenSpansOptions`, spans that have been open for longer
than ``max_age`` are registered in a bounded, queryable list. Registering and querying never affects
the span lifecycle: spans can still be ended normally at any time, at which point their entry
disappears from the list.

When the process exits, any spans that are still open are force-ended as before, and a summary
stating how many spans were force-ended and for how long each had been open is logged.
"""

from __future__ import annotations

import weakref
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from threading import Lock
from typing import Any

import opentelemetry.trace as trace_api

from .utils import logger

CODE_FILEPATH_KEY = 'code.filepath'
CODE_LINENO_KEY = 'code.lineno'
CODE_FUNCTION_KEY = 'code.function'

_NS_PER_SECOND = 1_000_000_000


def _ns_to_timedelta(ns: int) -> timedelta:
    return timedelta(seconds=ns / _NS_PER_SECOND)


@dataclass(frozen=True)
class OpenSpansOptions:
    """Options for detecting spans that stay open for too long.

    When passed to `logfire.configure()` as the ``open_spans`` argument, spans that are still open
    after ``max_age`` are registered in a queryable list accessible via `logfire.get_open_spans()`.
    """

    max_age: timedelta
    """How long a span may stay open before it is registered in the list of long-running spans."""

    max_entries: int = 100
    """Maximum number of entries kept in the queryable list.

    When the list is full, the oldest (earliest started) entries are dropped and counted in the
    snapshot's ``dropped_count``.
    """


@dataclass(frozen=True)
class OpenSpanInfo:
    """Information about a span that has been open for longer than the configured maximum age."""

    name: str
    """The name of the span."""

    age: timedelta
    """How long the span has been open."""

    trace_id: str
    """The trace the span belongs to, formatted as a 32-character hexadecimal string."""

    span_id: str
    """The ID of the span, formatted as a 16-character hexadecimal string."""

    code_filepath: str | None = None
    """The path of the file where the span was started, if known."""

    code_lineno: int | None = None
    """The line in the file where the span was started, if known."""

    code_function: str | None = None
    """The fully qualified function name where the span was started, if known."""


@dataclass(frozen=True)
class OpenSpansSnapshot:
    """A snapshot of the long-running open spans list."""

    entries: tuple[OpenSpanInfo, ...]
    """The spans currently registered in the list, oldest first."""

    dropped_count: int
    """How many entries have been dropped because the list capacity was exceeded."""


@dataclass(frozen=True)
class ForceClosedSpanInfo:
    """Information about a span that was ended on the user's behalf when the process exited."""

    name: str
    duration: timedelta
    trace_id: str
    span_id: str
    code_filepath: str | None
    code_lineno: int | None
    code_function: str | None


@dataclass(frozen=True)
class ExitOpenSpansSummary:
    """Summary of open spans force-ended at process exit for one `OpenSpansMonitor`."""

    force_closed: tuple[ForceClosedSpanInfo, ...]
    """The spans that were force-ended, oldest first, each with how long it had been open."""

    dropped_stale_entries_count: int
    """How many long-running span entries had been dropped due to the list capacity."""


@dataclass(frozen=True)
class _TrackedSpan:
    span_ref: weakref.ReferenceType[trace_api.Span]
    start_time_ns: int
    name: str
    trace_id: int
    span_id: int
    code_filepath: str | None
    code_lineno: int | None
    code_function: str | None

    def open_span_info(self, now_ns: int) -> OpenSpanInfo:
        return OpenSpanInfo(
            name=self.name,
            age=_ns_to_timedelta(max(0, now_ns - self.start_time_ns)),
            trace_id=trace_api.format_trace_id(self.trace_id),
            span_id=trace_api.format_span_id(self.span_id),
            code_filepath=self.code_filepath,
            code_lineno=self.code_lineno,
            code_function=self.code_function,
        )

    def force_closed_info(self, now_ns: int) -> ForceClosedSpanInfo:
        return ForceClosedSpanInfo(
            name=self.name,
            duration=_ns_to_timedelta(max(0, now_ns - self.start_time_ns)),
            trace_id=trace_api.format_trace_id(self.trace_id),
            span_id=trace_api.format_span_id(self.span_id),
            code_filepath=self.code_filepath,
            code_lineno=self.code_lineno,
            code_function=self.code_function,
        )


class OpenSpansMonitor:
    """Tracks open spans and keeps the bounded list of spans open for longer than ``max_age``."""

    def __init__(self, options: OpenSpansOptions, now: Callable[[], int]) -> None:
        self._max_age_ns = int(options.max_age.total_seconds() * _NS_PER_SECOND)
        self._max_entries = options.max_entries
        self._now = now
        self._lock = Lock()
        # All spans started while this monitor is active, keyed by (trace_id, span_id).
        self._tracked: dict[tuple[int, int], _TrackedSpan] = {}
        # The subset of tracked spans open for longer than max_age, ordered oldest first.
        self._stale: OrderedDict[tuple[int, int], _TrackedSpan] = OrderedDict()
        # Keys of entries that were dropped from `_stale` due to the capacity limit.
        # Such spans are not re-registered if encountered again by a later sweep.
        self._evicted: set[tuple[int, int]] = set()
        self.dropped_count = 0

    def note_started(
        self,
        span: trace_api.Span,
        *,
        name: str,
        start_time_ns: int,
        attributes: Mapping[str, Any],
    ) -> None:
        """Record that a span has started. Never raises to the caller."""
        context = span.get_span_context()
        key = (context.trace_id, context.span_id)
        filepath = attributes.get(CODE_FILEPATH_KEY)
        lineno = attributes.get(CODE_LINENO_KEY)
        function_name = attributes.get(CODE_FUNCTION_KEY)
        tracked = _TrackedSpan(
            span_ref=weakref.ref(span),
            start_time_ns=start_time_ns,
            name=name,
            trace_id=context.trace_id,
            span_id=context.span_id,
            code_filepath=filepath if isinstance(filepath, str) else None,
            code_lineno=lineno if isinstance(lineno, int) else None,
            code_function=function_name if isinstance(function_name, str) else None,
        )
        with self._lock:
            self._tracked[key] = tracked

    def note_ended(self, key: tuple[int, int]) -> None:
        """Remove all records of a span that ended normally. Never raises to the caller."""
        with self._lock:
            self._tracked.pop(key, None)
            self._stale.pop(key, None)
            self._evicted.discard(key)

    def sweep(self) -> None:
        """Prune dead references and register spans open for longer than ``max_age``."""
        now_ns = self._now()
        with self._lock:
            new_stale: list[_TrackedSpan] = []
            for key, tracked in list(self._tracked.items()):
                if tracked.span_ref() is None:
                    # The span was garbage collected without being ended explicitly.
                    self._tracked.pop(key, None)
                    self._stale.pop(key, None)
                    continue
                if key in self._stale or key in self._evicted:
                    continue
                if now_ns - tracked.start_time_ns >= self._max_age_ns:
                    new_stale.append(tracked)
            # Spans are registered oldest first so that overflow eviction drops the earliest spans.
            new_stale.sort(key=lambda tracked: tracked.start_time_ns)
            for tracked in new_stale:
                key = (tracked.trace_id, tracked.span_id)
                self._stale[key] = tracked
                self._enforce_capacity_locked()

    def _enforce_capacity_locked(self) -> None:
        while len(self._stale) > self._max_entries:
            key, _tracked = self._stale.popitem(last=False)
            self._evicted.add(key)
            self.dropped_count += 1

    def snapshot(self) -> OpenSpansSnapshot:
        """Return a snapshot of the long-running spans list, running a sweep first."""
        self.sweep()
        now_ns = self._now()
        with self._lock:
            entries = tuple(tracked.open_span_info(now_ns) for tracked in self._stale.values())
            return OpenSpansSnapshot(entries=entries, dropped_count=self.dropped_count)

    def force_close(self, spans: Sequence[trace_api.Span]) -> ExitOpenSpansSummary:
        """Force-end the given spans at process exit and build a summary of their durations."""
        now_ns = self._now()
        with self._lock:
            tracked_by_key = dict(self._tracked)

        def start_time(span: trace_api.Span) -> int:
            key = _span_key(span)
            tracked = tracked_by_key.get(key)
            if tracked is not None:
                return tracked.start_time_ns
            inner_start = getattr(getattr(span, 'span', None), 'start_time', None)
            return inner_start if isinstance(inner_start, int) else 0

        infos: list[ForceClosedSpanInfo] = []
        for span in sorted(spans, key=start_time):
            key = _span_key(span)
            tracked = tracked_by_key.get(key)
            if tracked is not None:
                info = tracked.force_closed_info(now_ns)
            else:
                info = _force_closed_info_from_live_span(span, now_ns)
            infos.append(info)
            # Ending the span runs the normal `end()` path, which also removes it from this monitor.
            # Neutralize further calls so that interpreter shutdown garbage collection doesn't
            # trigger a warning about ending an ended span.
            span.end()
            span.end = lambda *_, **__: None  # pyright: ignore[reportUnknownLambdaType]

        summary = ExitOpenSpansSummary(
            force_closed=tuple(infos),
            dropped_stale_entries_count=self.dropped_count,
        )
        if infos:
            logger.warning(format_exit_summary(summary, self._max_entries))
        return summary


def _span_key(span: trace_api.Span) -> tuple[int, int]:
    context = span.get_span_context()
    return context.trace_id, context.span_id


def _force_closed_info_from_live_span(span: trace_api.Span, now_ns: int) -> ForceClosedSpanInfo:
    """Fallback for spans with no tracked record, e.g. whose start hook raised an internal error."""
    context = span.get_span_context()
    inner = getattr(span, 'span', None)
    name = getattr(inner, 'name', None) or '<span>'
    start_time_ns = getattr(inner, 'start_time', None)
    duration = _ns_to_timedelta(max(0, now_ns - start_time_ns)) if isinstance(start_time_ns, int) else timedelta(0)
    attributes: Mapping[str, Any] = getattr(inner, 'attributes', None) or {}
    filepath = attributes.get(CODE_FILEPATH_KEY)
    lineno = attributes.get(CODE_LINENO_KEY)
    function = attributes.get(CODE_FUNCTION_KEY)
    return ForceClosedSpanInfo(
        name=name,
        duration=duration,
        trace_id=trace_api.format_trace_id(context.trace_id),
        span_id=trace_api.format_span_id(context.span_id),
        code_filepath=filepath if isinstance(filepath, str) else None,
        code_lineno=lineno if isinstance(lineno, int) else None,
        code_function=function if isinstance(function, str) else None,
    )


def format_exit_summary(summary: ExitOpenSpansSummary, max_entries: int) -> str:
    """Format the process-exit summary as a human-readable message."""
    count = len(summary.force_closed)
    lines = [
        f'Logfire force-ended {count} span{"s" if count != 1 else ""} that '
        f'{"were" if count != 1 else "was"} still open when the process exited:'
    ]
    for info in summary.force_closed:
        origin = ''
        if info.code_filepath:
            origin = f', started at {info.code_filepath}:{info.code_lineno or "?"}'
            if info.code_function:
                origin += f' in {info.code_function}'
        lines.append(f'  - {info.name!r} open for {info.duration} (trace id {info.trace_id}{origin})')
    dropped = summary.dropped_stale_entries_count
    if dropped:
        lines.append(
            f'{dropped} long-running span entr{"ies" if dropped != 1 else "y"} '
            f'{"were" if dropped != 1 else "was"} dropped because the list capacity '
            f'({max_entries}) was exceeded.'
        )
    return '\n'.join(lines)


# The monitor used by `logfire.get_open_spans()`. Set by the global Logfire config when the feature
# is enabled. Spans additionally keep a direct reference to the monitor that tracks them, so local
# (non-global) configs and reconfiguration don't confuse process-exit bookkeeping.
_active_monitor: OpenSpansMonitor | None = None


def set_active_monitor(monitor: OpenSpansMonitor | None) -> None:
    """Set the monitor backing `logfire.get_open_spans()`. Called when the global config changes."""
    global _active_monitor
    _active_monitor = monitor


def get_open_spans() -> OpenSpansSnapshot:
    """List spans that have been open for longer than the configured maximum age.

    Returns an empty snapshot if the feature is not enabled. See `OpenSpansOptions` and
    `logfire.configure()`.
    """
    if _active_monitor is None:
        return OpenSpansSnapshot(entries=(), dropped_count=0)
    return _active_monitor.snapshot()


def end_open_spans_at_exit() -> list[ExitOpenSpansSummary]:
    """Force-end every open span, logging a duration summary for monitored spans.

    Spans started while the feature was disabled are ended exactly as before. Monitored spans are
    ended via their `OpenSpansMonitor`, which logs a summary. Returns the summaries (mainly for
    testing).
    """
    from .tracer import OPEN_SPANS

    # Group spans by the monitor tracking them (the key 0 is for spans without a monitor).
    groups: dict[int, tuple[OpenSpansMonitor | None, list[trace_api.Span]]] = {}
    for span in list(OPEN_SPANS.values()):
        monitor = getattr(span, 'open_spans_monitor', None)
        group_id = id(monitor) if monitor is not None else 0
        group = groups.get(group_id)
        if group is None:
            groups[group_id] = (monitor, [span])
        else:
            group[1].append(span)

    summaries: list[ExitOpenSpansSummary] = []
    for monitor, spans in groups.values():
        if monitor is None:
            for span in spans:
                span.end()
                # Interpreter shutdown may trigger another call to .end(),
                # which would log a warning "Calling end() on an ended span."
                span.end = lambda *_, **__: None  # pyright: ignore[reportUnknownLambdaType]
        else:
            summaries.append(monitor.force_close(spans))
    return summaries
