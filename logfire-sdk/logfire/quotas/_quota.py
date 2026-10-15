from __future__ import annotations

import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from fnmatch import fnmatchcase, translate as fnmatch_translate
from threading import Lock
from typing import Any, Literal, cast

from opentelemetry._logs import LogRecord, SeverityNumber
from opentelemetry.context import Context
from opentelemetry.sdk._logs import LogRecordProcessor, ReadWriteLogRecord
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags, set_span_in_context

from logfire._internal.constants import (
    ATTRIBUTES_LOG_LEVEL_NUM_KEY,
    ATTRIBUTES_MESSAGE_KEY,
    ATTRIBUTES_MESSAGE_TEMPLATE_KEY,
    ATTRIBUTES_SPAN_TYPE_KEY,
    LEVEL_NUMBERS,
)
from logfire._internal.exporters.wrapper import WrapperLogProcessor, WrapperSpanProcessor
from logfire._internal.utils import ReadableSpanDict, handle_internal_errors, span_to_dict

QuotaKind = Literal['span', 'log']
"""The kind of telemetry a quota applies to: regular spans or logs."""

QuotaAction = Literal['drop', 'aggregate']
"""What to do with records produced beyond a rule's limit.

- `'drop'`: discard over-limit records outright; nothing is emitted for them.
- `'aggregate'`: fold over-limit records into a single count summary record
  emitted when the quota window closes.
"""

QuotaTarget = Literal['any', 'span', 'log']

#: Attribute keys used on synthetic aggregate summary records.
QUOTA_ATTRIBUTES_NAMESPACE = 'logfire.quota'
QUOTA_RULE_KEY = f'{QUOTA_ATTRIBUTES_NAMESPACE}.rule'
QUOTA_STRATEGY_KEY = f'{QUOTA_ATTRIBUTES_NAMESPACE}.strategy'
QUOTA_KIND_KEY = f'{QUOTA_ATTRIBUTES_NAMESPACE}.kind'
QUOTA_MATCHED_NAME_KEY = f'{QUOTA_ATTRIBUTES_NAMESPACE}.matched_name'
QUOTA_COUNT_KEY = f'{QUOTA_ATTRIBUTES_NAMESPACE}.count'
QUOTA_PERIOD_KEY = f'{QUOTA_ATTRIBUTES_NAMESPACE}.period'
QUOTA_LIMIT_KEY = f'{QUOTA_ATTRIBUTES_NAMESPACE}.limit'

#: Name/message template of synthetic aggregate summary records.
QUOTA_SUMMARY_NAME = 'logfire.quota_summary'

#: Log level assigned to aggregate summaries when none of the folded records carried a level.
_QUOTA_SUMMARY_DEFAULT_LEVEL = LEVEL_NUMBERS['notice']

#: Maximum number of (kind, name) entries cached as rule-matching results.
#: Span names and log templates are low cardinality by design; past this size matching
#: simply skips the cache rather than growing without bound.
_MATCH_CACHE_MAX_SIZE = 4096


@dataclass(frozen=True)
class QuotaRule:
    """A production quota rule.

    Caps how many telemetry records matching `name_pattern` may be *produced* per `period`.
    Records over the limit are either dropped outright or folded into one count summary,
    according to `on_exceed`.

    Example:
        ```python
        import logfire
        from logfire.quotas import QuotaOptions, QuotaRule

        logfire.configure(
            quotas=QuotaOptions(
                rules=[
                    # At most 100 spans with this exact name per second; drop anything more.
                    QuotaRule(name_pattern='chat completion', limit=100, period=1, on_exceed='drop'),
                    # At most 20 matching retry logs per 5 seconds; summarize the excess.
                    QuotaRule(
                        name_pattern='retrying request *',
                        limit=20,
                        period=5,
                        on_exceed='aggregate',
                        applies_to='log',
                    ),
                ]
            )
        )
        ```
    """

    name_pattern: str | re.Pattern[str]
    """Pattern matched against span names and log message templates.

    A plain string matches exactly but also supports `fnmatch` glob wildcards (`*` and `?`),
    so `'retrying *'` matches every name starting with `'retrying '`.
    A compiled `re.Pattern` is matched with `Pattern.search`.
    """

    limit: int
    """Maximum number of matching records that may be produced in each `period`.

    Must be a positive integer.
    """

    period: float = 1.0
    """Length of the quota window in seconds. Must be greater than 0."""

    on_exceed: QuotaAction = 'drop'
    """What happens to records produced beyond `limit`: `'drop'` or `'aggregate'`."""

    applies_to: QuotaTarget = 'any'
    """Which record kinds the rule applies to: `'span'`, `'log'`, or `'any'` (the default)."""

    name: str | None = None
    """Human-readable identifier of the rule used in the per-rule accounting.

    Defaults to the pattern itself (or `'regex:<pattern>'` for compiled patterns).
    Rule names must be unique within a `QuotaOptions` instance.
    """

    def __post_init__(self) -> None:
        # The isinstance checks are runtime validation for callers without type checking.
        if isinstance(self.limit, bool) or not isinstance(self.limit, int) or self.limit <= 0:  # pyright: ignore[reportUnnecessaryIsInstance]
            raise ValueError(f'quota rule {self.name_pattern!r}: limit must be a positive integer')
        if isinstance(self.period, bool) or not isinstance(self.period, (int, float)) or self.period <= 0:  # pyright: ignore[reportUnnecessaryIsInstance]
            raise ValueError(f'quota rule {self.name_pattern!r}: period must be greater than 0 seconds')
        if self.on_exceed not in ('drop', 'aggregate'):
            raise ValueError(f'quota rule {self.name_pattern!r}: on_exceed must be "drop" or "aggregate"')
        if self.applies_to not in ('any', 'span', 'log'):
            raise ValueError(f'quota rule {self.name_pattern!r}: applies_to must be "any", "span" or "log"')


@dataclass(frozen=True)
class QuotaOptions:
    """Options for [`logfire.configure(quotas=...)`][logfire.configure].

    Holds the [`QuotaRule`][logfire.quotas.QuotaRule] list that caps span/log production.
    With no rules (the default) no quota processing is added to the production path at all,
    so behaviour and overhead characteristics are unchanged.
    """

    rules: Sequence[QuotaRule] = ()
    """The quota rules, evaluated in declaration order.

    The first rule matching a record owns it; later rules never see it.
    """

    def __post_init__(self) -> None:
        # Normalize lists (including dict round-trips from `dataclasses.asdict`) to a tuple so
        # serialization is stable and rule order is immutable after configuration.
        object.__setattr__(self, 'rules', tuple(self.rules))
        seen: set[str] = set()
        for rule in self.rules:
            name = rule.name or default_rule_name(rule.name_pattern)
            if name in seen:
                raise ValueError(f'quota rule names must be unique, {name!r} is duplicated')
            seen.add(name)


def default_rule_name(pattern: str | re.Pattern[str]) -> str:
    """The rule name used when a [`QuotaRule`][logfire.quotas.QuotaRule] doesn't set `name`."""
    return pattern if isinstance(pattern, str) else f'regex:{pattern.pattern}'


def coerce_quota_options(options: Any) -> QuotaOptions:
    """Normalize the `quotas` config value, including the dict form produced by `dataclasses.asdict`.

    `LogfireConfig` is serialized to a dict for ProcessPoolExecutor config forwarding, so nested
    dataclasses come back as dicts and must be rebuilt by hand.
    """
    if isinstance(options, QuotaOptions):
        return options
    if options is None:
        return QuotaOptions()
    raw_rules = options.get('rules', ())
    rules = tuple(
        QuotaRule(**rule) if isinstance(rule, dict) else rule  # pyright: ignore[reportUnknownArgumentType]
        for rule in raw_rules
    )
    return QuotaOptions(rules=rules)


@dataclass(frozen=True)
class QuotaStats:
    """Accounting snapshot for a single quota rule.

    Returned by [`logfire.get_quota_stats()`][logfire.get_quota_stats] and
    [`Logfire.get_quota_stats()`][logfire.Logfire.get_quota_stats]. Totals are cumulative
    for the lifetime of the current configuration and remain readable after shutdown.
    """

    name: str
    """The rule's accounting name."""

    name_pattern: str
    """The rule's pattern as a human-readable string."""

    applies_to: QuotaTarget
    """Which record kinds the rule applies to."""

    limit: int
    """The rule's per-window production limit."""

    period: float
    """The rule's window length in seconds."""

    on_exceed: QuotaAction
    """The rule's over-limit strategy."""

    matched: int
    """Total number of records that matched the rule."""

    allowed: int
    """Number of matched records produced unchanged (`matched == allowed + dropped + aggregated`)."""

    dropped: int
    """Number of over-limit records discarded by a `'drop'` rule."""

    aggregated: int
    """Number of over-limit records folded into summary records by an `'aggregate'` rule."""

    summaries_emitted: int
    """Number of aggregate count summary records emitted so far."""

    current_window: int
    """Index of the current quota window (monotonic nanoseconds divided by the window length)."""

    current_window_used: int
    """Number of matching records already seen in the current window."""


@dataclass
class PendingAggregate:
    """Over-limit records folded together while waiting for the quota window to close.

    The `carrier` is opaque pipeline-specific data (a `ReadableSpanDict` for the span
    pipeline, a `_LogRecordCarrier` for the OTel log pipeline) used to render the summary.
    """

    rule_name: str
    pattern_repr: str
    limit: int
    period: float
    kind: QuotaKind
    matched_name: str
    max_level: int
    carrier: Any
    count: int = 0

    @property
    def source(self) -> str:
        return 'log-record' if isinstance(self.carrier, _LogRecordCarrier) else 'span'


@dataclass
class _LogRecordCarrier:
    """Everything needed to synthesize an aggregate summary in the OTel log pipeline."""

    resource: Any
    instrumentation_scope: Any
    trace_id: int
    span_id: int
    trace_flags: TraceFlags | None


class _RuleState:
    """Mutable quota accounting for one rule. Every lock is private to the rule."""

    def __init__(self, rule: QuotaRule, regex: re.Pattern[str]) -> None:
        self.rule = rule
        self.regex = regex
        self.period_ns = max(1, int(rule.period * 1_000_000_000))
        self.name = rule.name or default_rule_name(rule.name_pattern)
        self.lock = Lock()
        self.window_index = -1
        self.window_used = 0
        # Keyed by (matched name, kind) so a broad pattern still produces one summary
        # per distinct name/template rather than mixing records together.
        self.pending: dict[tuple[str, QuotaKind], PendingAggregate] = {}
        self.matched = 0
        self.allowed = 0
        self.dropped = 0
        self.aggregated = 0
        self.summaries_emitted = 0

    def matches(self, kind: QuotaKind, name: str) -> bool:
        if self.rule.applies_to != 'any' and self.rule.applies_to != kind:
            return False
        if isinstance(self.rule.name_pattern, str):
            return fnmatchcase(name, self.rule.name_pattern)
        return self.regex.search(name) is not None


class QuotaManager:
    """Owns per-rule rate limiting and accounting.

    A single manager is shared by the span and log pipelines so that the two never
    double-count a rule. Each rule has its own lock and counters, so concurrent
    over-limit events on different rules cannot block or cross-charge each other.
    """

    def __init__(
        self,
        rules: Sequence[QuotaRule] = (),
        *,
        now: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        compiled: list[tuple[QuotaRule, re.Pattern[str]]] = []
        for rule in rules:
            pattern = rule.name_pattern
            compiled.append((rule, re.compile(fnmatch_translate(pattern)) if isinstance(pattern, str) else pattern))
        self.states: tuple[_RuleState, ...] = tuple(_RuleState(rule, regex) for rule, regex in compiled)
        self.now = now
        # Values are tuples of matching states in declaration order, or an empty tuple.
        self._match_cache: dict[tuple[QuotaKind, str], tuple[_RuleState, ...]] = {}
        self._match_cache_lock = Lock()

    @classmethod
    def from_options(
        cls,
        options: QuotaOptions | None,
        *,
        now: Callable[[], int] = time.monotonic_ns,
    ) -> QuotaManager:
        return cls(() if options is None else options.rules, now=now)

    @property
    def enabled(self) -> bool:
        """Whether any rules exist. When false the production path is left untouched."""
        return bool(self.states)

    def _matching_states(self, kind: QuotaKind, name: str) -> tuple[_RuleState, ...]:
        key = (kind, name)
        cache = self._match_cache
        # The read is deliberately lock-free: individual dict operations are atomic under the
        # GIL, and a stale/duplicate result only means a redundant (and still correct) rule
        # evaluation. The hot path therefore never blocks on the cache.
        cached = cache.get(key)
        if cached is not None:
            return cached
        matches = tuple(state for state in self.states if state.matches(kind, name))
        with self._match_cache_lock:
            # Bound the cache; past the cap keep matching correctly but don't grow. Races may
            # insert a key twice or overshoot by a handful of entries, both harmless.
            if key not in cache and len(cache) < _MATCH_CACHE_MAX_SIZE:
                cache[key] = matches
        return matches

    def record(
        self,
        kind: QuotaKind,
        name: str,
        data: Any,
        level: int = 0,
    ) -> tuple[bool, tuple[PendingAggregate, ...]]:
        """Apply the first matching rule to one produced record.

        `data` is the raw pipeline record (a `ReadableSpan` or `ReadWriteLogRecord`); it is
        only converted into summary render data if the record is the first one folded into
        a new aggregate, so the common path allocates nothing extra.

        Returns `(forward, closed_aggregates)`: when `forward` is false the record must be
        withheld from the downstream pipeline. `closed_aggregates` are aggregates whose
        window just ended; the caller should render and emit their summaries.
        """
        states = self._matching_states(kind, name)
        if not states:
            return True, ()

        state = states[0]
        now_ns = self.now()

        with state.lock:
            window_index = now_ns // state.period_ns
            closed: tuple[PendingAggregate, ...] = ()
            if state.window_index != window_index:
                if state.pending:
                    # The caller renders these and calls `mark_emitted` once they're sent,
                    # so aggregates requeued by the other pipeline are never counted.
                    closed = tuple(state.pending.values())
                    state.pending = {}
                state.window_index = window_index
                state.window_used = 0

            state.matched += 1
            state.window_used += 1
            if state.window_used <= state.rule.limit:
                state.allowed += 1
                return True, closed

            if state.rule.on_exceed == 'drop':
                state.dropped += 1
                return False, closed

            state.aggregated += 1
            key = (name, kind)
            aggregate = state.pending.get(key)
            if aggregate is None:
                # The factory turns the (already final) record into render data; only the
                # first over-limit record of each window/name pays this cost.
                aggregate = PendingAggregate(
                    rule_name=state.name,
                    pattern_repr=_pattern_repr(state.rule.name_pattern),
                    limit=state.rule.limit,
                    period=state.rule.period,
                    kind=kind,
                    matched_name=name,
                    max_level=level,
                    # Dispatch on the pipeline record type: logfire's own logs also reach
                    # the span pipeline (kind 'log') but as ReadableSpan instances.
                    carrier=span_to_dict(data) if isinstance(data, ReadableSpan) else _log_record_carrier(data),
                )
                state.pending[key] = aggregate
            elif level > aggregate.max_level:
                aggregate.max_level = level
            aggregate.count += 1
            return False, closed

    def drain_pending(self, *, source: str) -> tuple[PendingAggregate, ...]:
        """Remove and return all pending aggregates belonging to `source`.

        `source` is `'span'` or `'log-record'` (see `PendingAggregate.source`).
        Used at flush/shutdown so the final window's summaries are not lost; aggregates
        belonging to the other pipeline stay queued for its own drain.
        """
        drained: list[PendingAggregate] = []
        for state in self.states:
            with state.lock:
                remaining: dict[tuple[str, QuotaKind], PendingAggregate] = {}
                for key, aggregate in state.pending.items():
                    if aggregate.source == source:
                        drained.append(aggregate)
                    else:
                        remaining[key] = aggregate
                state.pending = remaining
        return tuple(drained)

    def mark_emitted(self, aggregates: Sequence[PendingAggregate]) -> None:
        """Account for window-rollover summaries that have actually been rendered and sent."""
        per_rule: dict[str, int] = {}
        for aggregate in aggregates:
            per_rule[aggregate.rule_name] = per_rule.get(aggregate.rule_name, 0) + 1
        states_by_name = {state.name: state for state in self.states}
        for rule_name, count in per_rule.items():
            state = states_by_name.get(rule_name)
            if state is not None:
                with state.lock:
                    state.summaries_emitted += count

    def requeue(self, aggregates: Sequence[PendingAggregate]) -> None:
        """Give back aggregates that the caller could not render (foreign pipeline carrier).

        Counts are merged into the still-open window entry if any, so nothing is lost or
        double-charged.
        """
        states_by_name = {state.name: state for state in self.states}
        for aggregate in aggregates:
            state = states_by_name.get(aggregate.rule_name)
            if state is None:  # pragma: no cover
                continue
            with state.lock:
                key = (aggregate.matched_name, aggregate.kind)
                existing = state.pending.get(key)
                if existing is None:
                    state.pending[key] = aggregate
                else:
                    existing.count += aggregate.count
                    existing.max_level = max(existing.max_level, aggregate.max_level)

    def stats(self) -> list[QuotaStats]:
        """Return a point-in-time accounting snapshot for every rule, in declaration order."""
        result: list[QuotaStats] = []
        for state in self.states:
            with state.lock:
                result.append(
                    QuotaStats(
                        name=state.name,
                        name_pattern=_pattern_repr(state.rule.name_pattern),
                        applies_to=state.rule.applies_to,
                        limit=state.rule.limit,
                        period=state.rule.period,
                        on_exceed=state.rule.on_exceed,
                        matched=state.matched,
                        allowed=state.allowed,
                        dropped=state.dropped,
                        aggregated=state.aggregated,
                        summaries_emitted=state.summaries_emitted,
                        current_window=state.window_index,
                        current_window_used=state.window_used,
                    )
                )
        return result


def _pattern_repr(pattern: str | re.Pattern[str]) -> str:
    return pattern if isinstance(pattern, str) else pattern.pattern


def _record_level(attributes: Any) -> int:
    """Extract the logfire level number from span/log attributes, defaulting to 0."""
    level = attributes.get(ATTRIBUTES_LOG_LEVEL_NUM_KEY) if attributes else None
    return level if isinstance(level, int) else 0


def _log_summary_message(aggregate: PendingAggregate) -> str:
    noun = 'log' if aggregate.kind == 'log' else 'span'
    return (
        f'Logfire quota {aggregate.rule_name!r} aggregated {aggregate.count} over-limit {noun} '
        f'record(s) (limit {aggregate.limit} per {aggregate.period:g}s, pattern '
        f'{aggregate.pattern_repr!r}, e.g. {aggregate.matched_name!r})'
    )


def _summary_attributes(aggregate: PendingAggregate) -> dict[str, Any]:
    level = aggregate.max_level or _QUOTA_SUMMARY_DEFAULT_LEVEL
    attributes: dict[str, Any] = {
        ATTRIBUTES_MESSAGE_TEMPLATE_KEY: QUOTA_SUMMARY_NAME,
        ATTRIBUTES_MESSAGE_KEY: _log_summary_message(aggregate),
        ATTRIBUTES_LOG_LEVEL_NUM_KEY: level,
        QUOTA_RULE_KEY: aggregate.rule_name,
        QUOTA_STRATEGY_KEY: 'aggregate',
        QUOTA_KIND_KEY: aggregate.kind,
        QUOTA_MATCHED_NAME_KEY: aggregate.matched_name,
        QUOTA_COUNT_KEY: aggregate.count,
        QUOTA_PERIOD_KEY: aggregate.period,
        QUOTA_LIMIT_KEY: aggregate.limit,
    }
    if aggregate.kind == 'log':
        attributes[ATTRIBUTES_SPAN_TYPE_KEY] = 'log'
    return attributes


def _classify_span(span: ReadableSpan) -> tuple[QuotaKind, str, int] | None:
    """Return `(kind, matching name, level)` for a span-pipeline record, or None to pass it through.

    Both regular spans and logfire logs travel through the span processor pipeline: logs are
    spans carrying `logfire.span_type == 'log'`. Pending spans are synthetic duplicates and
    are never quota-controlled.
    """
    attributes = span.attributes
    span_type: object = attributes.get(ATTRIBUTES_SPAN_TYPE_KEY) if attributes is not None else None
    if span_type == 'pending_span':
        return None
    level = _record_level(attributes)
    if span_type == 'log':
        template: object = attributes.get(ATTRIBUTES_MESSAGE_TEMPLATE_KEY) if attributes is not None else None
        return 'log', template if isinstance(template, str) else span.name, level
    return 'span', span.name, level


def _classify_log_record(record: LogRecord) -> tuple[str, int] | None:
    """Return `(template name, level)` for an OTel log record, or None to pass it through."""
    attributes = record.attributes
    template: object = attributes.get(ATTRIBUTES_MESSAGE_TEMPLATE_KEY) if attributes is not None else None
    if not isinstance(template, str):
        body: object = record.body
        if not isinstance(body, str):
            return None
        template = body
    return template, _record_level(attributes)


def _log_record_carrier(log_record: ReadWriteLogRecord) -> _LogRecordCarrier:
    """Capture the data needed to render an aggregate summary from an OTel log record."""
    record = log_record.log_record
    return _LogRecordCarrier(
        resource=log_record.resource,
        instrumentation_scope=log_record.instrumentation_scope,
        trace_id=record.trace_id,
        span_id=record.span_id,
        trace_flags=record.trace_flags,
    )


def _log_record_carrier_context(carrier: _LogRecordCarrier) -> Context:
    span_context = SpanContext(
        trace_id=carrier.trace_id,
        span_id=carrier.span_id,
        is_remote=False,
        trace_flags=TraceFlags(carrier.trace_flags or 0),
    )
    return set_span_in_context(NonRecordingSpan(span_context))


class QuotaEnforcingSpanProcessor(WrapperSpanProcessor):
    """Span processor enforcing production quotas.

    Placed inside the logfire main processor wrapper so it sees final names and scrubbed
    attributes, and around (after) tail sampling so sampled-out traces are never charged.
    Over-limit records are not forwarded; aggregate summaries are synthesized downstream.
    """

    manager: QuotaManager
    ns_timestamp_generator: Callable[[], int]

    def __init__(
        self,
        processor: SpanProcessor,
        manager: QuotaManager,
        ns_timestamp_generator: Callable[[], int],
    ) -> None:
        super().__init__(processor)
        self.manager = manager
        self.ns_timestamp_generator = ns_timestamp_generator

    def on_end(self, span: ReadableSpan) -> None:
        classification = _classify_span(span)
        if classification is None or not self.manager.enabled:
            super().on_end(span)
            return

        kind, name, level = classification
        forward, closed = self.manager.record(kind, name, span, level)
        for aggregate in closed:
            self._emit_aggregate(aggregate)
        if forward:
            super().on_end(span)

    def _emit_aggregate(self, aggregate: PendingAggregate) -> None:
        if aggregate.source != 'span':
            # Belongs to the OTel log pipeline; hand it back for that pipeline's drain.
            self.manager.requeue([aggregate])
            return
        with handle_internal_errors:
            self.processor.on_end(self._build_summary_span(aggregate))
            self.manager.mark_emitted([aggregate])

    def _build_summary_span(self, aggregate: PendingAggregate) -> ReadableSpan:
        # The carrier is the ReadableSpanDict of the first over-limit record in the window.
        representative = cast(ReadableSpanDict, aggregate.carrier)
        now_ns = self.ns_timestamp_generator()
        return ReadableSpan(
            name=QUOTA_SUMMARY_NAME,
            context=representative['context'],
            parent=representative['parent'],
            resource=representative['resource'],
            attributes=_summary_attributes(aggregate),
            events=(),
            links=(),
            kind=representative['kind'],
            instrumentation_scope=representative['instrumentation_scope'],
            start_time=now_ns,
            end_time=now_ns,
        )

    def _drain(self) -> None:
        for aggregate in self.manager.drain_pending(source='span'):
            self._emit_aggregate(aggregate)

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        self._drain()
        return super().force_flush(timeout_millis)

    def shutdown(self) -> None:
        # Flush before shutting the downstream processors down so the final window
        # summaries are still exported.
        self._drain()
        super().shutdown()


class QuotaEnforcingLogProcessor(WrapperLogProcessor):
    """Log record processor enforcing the same quota rules on the OTel log pipeline.

    Only a few integrations (e.g. Monty, GenAI) emit through this pipeline directly;
    logfire's own logs go through the span pipeline and are covered there.
    """

    manager: QuotaManager
    ns_timestamp_generator: Callable[[], int]

    def __init__(
        self,
        processor: LogRecordProcessor,
        manager: QuotaManager,
        ns_timestamp_generator: Callable[[], int],
    ) -> None:
        super().__init__(processor)
        self.manager = manager
        self.ns_timestamp_generator = ns_timestamp_generator

    def on_emit(self, log_record: ReadWriteLogRecord) -> None:
        record = log_record.log_record
        classification = _classify_log_record(record)
        if classification is None or not self.manager.enabled:
            super().on_emit(log_record)
            return

        name, level = classification
        forward, closed = self.manager.record('log', name, log_record, level)
        for aggregate in closed:
            self._emit_aggregate(aggregate)
        if forward:
            super().on_emit(log_record)

    def _emit_aggregate(self, aggregate: PendingAggregate) -> None:
        if aggregate.source != 'log-record':
            self.manager.requeue([aggregate])
            return
        with handle_internal_errors:
            self.processor.on_emit(self._build_summary_log_record(aggregate))
            self.manager.mark_emitted([aggregate])

    def _build_summary_log_record(self, aggregate: PendingAggregate) -> ReadWriteLogRecord:
        now_ns = self.ns_timestamp_generator()
        level = aggregate.max_level or LEVEL_NUMBERS['notice']
        carrier = cast(_LogRecordCarrier, aggregate.carrier)
        attributes = _summary_attributes(aggregate)
        record = LogRecord(
            timestamp=now_ns,
            observed_timestamp=now_ns,
            context=_log_record_carrier_context(carrier),
            severity_text=None,
            severity_number=SeverityNumber(level),
            body=attributes[ATTRIBUTES_MESSAGE_KEY],
            attributes=attributes,
        )
        return ReadWriteLogRecord(
            log_record=record,
            resource=carrier.resource,
            instrumentation_scope=carrier.instrumentation_scope,
        )

    def _drain(self) -> None:
        for aggregate in self.manager.drain_pending(source='log-record'):
            self._emit_aggregate(aggregate)

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        self._drain()
        return super().force_flush(timeout_millis)

    def shutdown(self) -> None:
        self._drain()
        super().shutdown()
