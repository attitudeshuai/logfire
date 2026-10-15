from __future__ import annotations

import copy
import json
import re
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from threading import Lock
from typing import Any, TypedDict, cast

import typing_extensions
from opentelemetry._logs import LogRecord
from opentelemetry.attributes import BoundedAttributes
from opentelemetry.sdk.trace import Event
from opentelemetry.trace import Link

from .constants import (
    ATTRIBUTES_CONFIG,
    ATTRIBUTES_JSON_SCHEMA_KEY,
    ATTRIBUTES_LOG_LEVEL_NAME_KEY,
    ATTRIBUTES_LOG_LEVEL_NUM_KEY,
    ATTRIBUTES_LOGGING_NAME,
    ATTRIBUTES_MESSAGE_KEY,
    ATTRIBUTES_MESSAGE_TEMPLATE_KEY,
    ATTRIBUTES_PACKAGE_VERSIONS,
    ATTRIBUTES_PENDING_SPAN_REAL_PARENT_KEY,
    ATTRIBUTES_SAMPLE_RATE_KEY,
    ATTRIBUTES_SCRUBBED_KEY,
    ATTRIBUTES_SPAN_TYPE_KEY,
    ATTRIBUTES_TAGS_KEY,
    MESSAGE_FORMATTED_VALUE_LENGTH_LIMIT,
    RESOURCE_ATTRIBUTES_VERSION,
)
from .integrations.llm_providers import semconv as gen_ai_semconv
from .stack_info import STACK_INFO_KEYS
from .utils import ReadableSpanDict, truncate_string

DEFAULT_PATTERNS = [
    'password',
    'passwd',
    'mysql_pwd',
    'secret',
    r'auth(?!ors?\b)',
    'credential',
    'private[._ -]?key',
    'api[._ -]?key',
    'session',
    'cookie',
    'social[._ -]?security',
    'credit[._ -]?card',
    'logfire[._ -]?token',
    # OAuth-style token names. A bare 'token' is left out because it would also match LLM usage fields like input_tokens.
    r'(?:access|refresh|bearer|(?:\b|_)id)[._ -]?token',
    r'pylf_v\d+_',
    *[
        # Require these to be surrounded by word boundaries or underscores,
        # to reduce the chance of accidentally matching them in a big blob of random chars, e.g. base64.
        rf'(?:\b|_){acronym}(?:\b|_)'
        for acronym in [
            'csrf',
            'xsrf',
            'jwt',
            'ssn',
        ]
    ],
]

# Every default pattern starts matching at one of these characters. Checking this
# inexpensive character class first avoids trying every alternative at every
# position in large strings. Custom patterns are not constrained by this prefix.
_DEFAULT_PATTERN_START_CHARS = 'pmsacljx_rbi'
_DEFAULT_PATTERN = rf'(?=[{_DEFAULT_PATTERN_START_CHARS}])(?:{"|".join(DEFAULT_PATTERNS)})'

JsonPath: typing_extensions.TypeAlias = 'tuple[str | int, ...]'


@dataclass
class ScrubMatch:
    """An object passed to a [`ScrubbingOptions.callback`][logfire.ScrubbingOptions.callback] function."""

    path: JsonPath
    """The path to the value in the span being considered for redaction, e.g. `('attributes', 'password')`."""

    value: Any
    """The value in the span being considered for redaction, e.g. `'my_password'`."""

    pattern_match: re.Match[str]
    """
    The regex match object indicating why the value is being redacted.
    Use `pattern_match.group(0)` to get the matched string.
    """


ScrubCallback = Callable[[ScrubMatch], Any]


class ScrubbedNote(TypedDict):
    path: JsonPath
    matched_substring: str


# The policy used when the summary is full. Only `'drop-oldest'` is currently supported,
# but the dedicated alias leaves room to add more policies without changing the option type.
ScrubbingSummaryEvictionPolicy = typing_extensions.Literal['drop-oldest']
SCRUBBING_SUMMARY_DROP_OLDEST: ScrubbingSummaryEvictionPolicy = 'drop-oldest'

DEFAULT_SCRUBBING_SUMMARY_MAX_ENTRIES = 10_000


@dataclass(frozen=True)
class ScrubbingPathSummary:
    """Hit counts for one pattern at one attribute path."""

    hits: int
    """Total number of hits at this path, including both redacted and unchanged values."""

    redacted: int
    """
    Number of hits whose value was redacted or replaced.
    In dry-run mode this counts values that *would* have been redacted/replaced.
    """

    unchanged: int
    """Number of hits left unchanged because the callback returned the original value."""


@dataclass(frozen=True)
class ScrubbingPatternSummary:
    """Hit counts for one scrubbing pattern, across all attribute paths."""

    pattern: str
    """The source of the regular expression that matched, e.g. `'password'`."""

    hits: int
    """Total number of times the pattern matched."""

    redacted: int
    """Number of matches that were redacted or replaced (or would have been in dry-run mode)."""

    unchanged: int
    """Number of matches left unchanged because the callback returned the original value."""

    paths: dict[JsonPath, ScrubbingPathSummary]
    """How hits were distributed across attribute paths, keyed by path e.g. `('attributes', 'password')`."""

    first_hit: float
    """Unix timestamp of the first recorded hit (`time.time()`)."""

    last_hit: float
    """Unix timestamp of the most recent recorded hit (`time.time()`)."""


@dataclass(frozen=True)
class ScrubbingSummary:
    """An immutable snapshot of scrubbing hits, as returned by [`logfire.get_scrubbing_summary`][logfire.get_scrubbing_summary]."""

    patterns: dict[str, ScrubbingPatternSummary]
    """Per-pattern summaries, keyed by pattern source string."""

    dropped_entries: int
    """Number of `(pattern, path)` entries discarded because `max_entries` was reached."""

    max_entries: int
    """The configured maximum number of `(pattern, path)` entries."""

    eviction_policy: ScrubbingSummaryEvictionPolicy
    """The policy used when entries had to be discarded."""

    generated_at: float
    """Unix timestamp at which this snapshot was taken (`time.time()`)."""


@dataclass
class ScrubbingSummaryOptions:
    """Options for collecting a summary of scrubbing hits.

    See [`logfire.get_scrubbing_summary`][logfire.get_scrubbing_summary] and the dry-run example in
    [`ScrubbingOptions.dry_run`][logfire.ScrubbingOptions.dry_run].
    """

    max_entries: int = DEFAULT_SCRUBBING_SUMMARY_MAX_ENTRIES
    """
    Maximum number of `(pattern, attribute path)` entries kept in the summary.
    When the summary is full, entries are discarded according to `eviction_policy`.
    Pattern-level hit counts always stay exact even when entries are discarded;
    only the per-path breakdown loses detail.
    """

    eviction_policy: ScrubbingSummaryEvictionPolicy = SCRUBBING_SUMMARY_DROP_OLDEST
    """
    What to do once `max_entries` is reached.
    `'drop-oldest'` (the only supported value) discards the entry that was hit least recently
    and increments [`ScrubbingSummary.dropped_entries`][logfire.ScrubbingSummary.dropped_entries].
    """

    def __post_init__(self) -> None:
        # Use an exact type check rather than `isinstance`: the dataclass annotation isn't
        # enforced at runtime, and `bool` must not be accepted as an `int` here.
        if type(self.max_entries) is not int:
            raise TypeError('`max_entries` must be an integer')
        if self.max_entries <= 0:
            raise ValueError('`max_entries` must be a positive integer')
        if self.eviction_policy != SCRUBBING_SUMMARY_DROP_OLDEST:
            raise ValueError(
                f'Unknown scrubbing summary eviction policy {self.eviction_policy!r}; '
                f'expected {SCRUBBING_SUMMARY_DROP_OLDEST!r}'
            )


@dataclass(frozen=True)
class _ScrubbingHit:
    """A single match, accumulated locally while a span/log/value is being scrubbed."""

    pattern: str
    path: JsonPath
    redacted: bool


@dataclass
class _ScrubbingPathEntry:
    hits: int = 0
    redacted: int = 0
    unchanged: int = 0


@dataclass
class _ScrubbingPatternAggregate:
    first_hit: float
    last_hit: float
    hits: int = 0
    redacted: int = 0
    unchanged: int = 0


class _ScrubbingSummaryRecorder:
    """Thread-safe accumulator of scrubbing hits.

    Hits found while scrubbing a single span/log/value are collected locally by `SpanScrubber`
    and committed here in a single locked operation (`commit`). A concurrent reader
    (`get_summary`) therefore never sees a partially committed span.
    """

    def __init__(self, options: ScrubbingSummaryOptions) -> None:
        self._options = options
        self._lock = Lock()
        # Ordered by recency of the last hit: the least recently hit entry is at the start,
        # so evicting the oldest entry is `popitem(last=False)`.
        self._entries: OrderedDict[tuple[str, JsonPath], _ScrubbingPathEntry] = OrderedDict()
        self._patterns: dict[str, _ScrubbingPatternAggregate] = {}
        self._dropped_entries = 0

    def commit(self, hits: Sequence[_ScrubbingHit], now: float) -> None:
        """Commit all hits from one scrubbed span/log/value atomically."""
        if not hits:
            return
        with self._lock:
            for hit in hits:
                aggregate = self._patterns.get(hit.pattern)
                if aggregate is None:
                    aggregate = _ScrubbingPatternAggregate(first_hit=now, last_hit=now)
                    self._patterns[hit.pattern] = aggregate
                aggregate.hits += 1
                aggregate.last_hit = now
                if hit.redacted:
                    aggregate.redacted += 1
                else:
                    aggregate.unchanged += 1

                key = (hit.pattern, hit.path)
                entry = self._entries.get(key)
                if entry is None:
                    # Only creating a new entry can exceed the capacity; an existing entry
                    # being hit again just moves to the most-recently-used end.
                    if len(self._entries) >= self._options.max_entries:
                        self._entries.popitem(last=False)
                        self._dropped_entries += 1
                    entry = _ScrubbingPathEntry()
                    self._entries[key] = entry
                else:
                    self._entries.move_to_end(key)
                entry.hits += 1
                if hit.redacted:
                    entry.redacted += 1
                else:
                    entry.unchanged += 1

    def get_summary(self) -> ScrubbingSummary:
        generated_at = time.time()
        with self._lock:
            paths_by_pattern: dict[str, dict[JsonPath, ScrubbingPathSummary]] = {name: {} for name in self._patterns}
            for (pattern, path), entry in self._entries.items():
                paths_by_pattern[pattern][path] = ScrubbingPathSummary(
                    hits=entry.hits,
                    redacted=entry.redacted,
                    unchanged=entry.unchanged,
                )
            patterns = {
                name: ScrubbingPatternSummary(
                    pattern=name,
                    hits=aggregate.hits,
                    redacted=aggregate.redacted,
                    unchanged=aggregate.unchanged,
                    paths=paths_by_pattern[name],
                    first_hit=aggregate.first_hit,
                    last_hit=aggregate.last_hit,
                )
                for name, aggregate in self._patterns.items()
            }
            return ScrubbingSummary(
                patterns=patterns,
                dropped_entries=self._dropped_entries,
                max_entries=self._options.max_entries,
                eviction_policy=self._options.eviction_policy,
                generated_at=generated_at,
            )


@dataclass
class ScrubbingOptions:
    """Options for redacting sensitive data."""

    callback: ScrubCallback | None = None
    """
    A function that is called for each match found by the scrubber.
    If it returns `None`, the value is redacted.
    Otherwise, the returned value replaces the matched value.
    The function accepts a single argument of type [`logfire.ScrubMatch`][logfire.ScrubMatch].
    """

    extra_patterns: Sequence[str] | None = None
    """
    A sequence of regular expressions to detect sensitive data that should be redacted.
    For example, the default includes `'password'`, `'secret'`, and `'api[._ -]?key'`.
    The specified patterns are combined with the default patterns.
    """

    dry_run: bool = False
    """
    If `True`, scrubbing runs in dry-run mode: matched values are left in place and exported
    normally, but every hit is still recorded in the summary configured with `summary`.
    Use this to validate a new pattern against real traffic before enforcing it.
    The callback (if any) is still consulted so that hits it would refuse to redact are
    recorded as unchanged, but its return value never modifies the output in dry-run mode.
    Defaults to `False`, which keeps the existing redaction semantics.
    """

    summary: ScrubbingSummaryOptions | None = None
    """
    Options for recording a summary of scrubbing hits (counts, attribute path distribution
    and most recent hit time per pattern). `None` (the default) disables the summary and
    leaves scrubbing behavior exactly as before. Retrieve the summary with
    [`logfire.get_scrubbing_summary`][logfire.get_scrubbing_summary].
    """


class BaseScrubber(ABC):
    # These keys and everything within are safe to keep in spans, even if they match the scrubbing pattern.
    # Some of these are just here for performance.
    SAFE_KEYS = {
        ATTRIBUTES_MESSAGE_KEY,  # Formatted field values are scrubbed separately
        ATTRIBUTES_MESSAGE_TEMPLATE_KEY,
        ATTRIBUTES_JSON_SCHEMA_KEY,
        ATTRIBUTES_TAGS_KEY,
        ATTRIBUTES_LOG_LEVEL_NAME_KEY,
        ATTRIBUTES_LOG_LEVEL_NUM_KEY,
        ATTRIBUTES_SPAN_TYPE_KEY,
        ATTRIBUTES_PENDING_SPAN_REAL_PARENT_KEY,
        ATTRIBUTES_SAMPLE_RATE_KEY,
        ATTRIBUTES_LOGGING_NAME,
        ATTRIBUTES_SCRUBBED_KEY,
        ATTRIBUTES_CONFIG,
        ATTRIBUTES_PACKAGE_VERSIONS,
        RESOURCE_ATTRIBUTES_VERSION,
        *STACK_INFO_KEYS,
        'exception.stacktrace',
        'exception.type',
        'exception.message',
        'error.type',
        'http.method',
        'http.status_code',
        'http.scheme',
        'http.url',
        'http.target',
        'http.route',
        'db.statement',
        'db.query.text',
        'db.plan',
        'fastapi.route.name',
        'fastapi.route.operation_id',
        'url.full',
        'url.path',
        'url.query',
        'event.name',
        'agent_session_id',
        'do_not_scrub',
        'binary_content',
        'pydantic_ai.all_messages',
        'rpc.method',
        'model_request_parameters',
        'langsmith.metadata.session_id',
        'langsmith.trace.session_name',
        gen_ai_semconv.INPUT_MESSAGES,
        gen_ai_semconv.OUTPUT_MESSAGES,
        gen_ai_semconv.SYSTEM_INSTRUCTIONS,
        gen_ai_semconv.TOOL_DEFINITIONS,
        gen_ai_semconv.TOOL_NAME,
        gen_ai_semconv.TOOL_CALL_ID,
        gen_ai_semconv.INPUT_TOKENS,
        gen_ai_semconv.OUTPUT_TOKENS,
        gen_ai_semconv.CACHE_READ_INPUT_TOKENS,
        gen_ai_semconv.CACHE_CREATION_INPUT_TOKENS,
        gen_ai_semconv.USAGE_RAW,
        gen_ai_semconv.CONVERSATION_ID,
        gen_ai_semconv.SYSTEM,
        gen_ai_semconv.PROVIDER_NAME,
        gen_ai_semconv.REQUEST_MODEL,
        gen_ai_semconv.RESPONSE_MODEL,
    }

    @abstractmethod
    def scrub_span(self, span: ReadableSpanDict): ...

    @abstractmethod
    def scrub_log(self, log: LogRecord) -> LogRecord: ...

    @abstractmethod
    def scrub_value(self, path: JsonPath, value: Any) -> tuple[Any, list[ScrubbedNote]]: ...

    def get_scrubbing_summary(self) -> ScrubbingSummary | None:
        """Return an immutable snapshot of scrubbing hits, or `None` if no summary is being recorded."""
        return None


class NoopScrubber(BaseScrubber):
    def scrub_span(self, span: ReadableSpanDict):
        pass

    def scrub_log(self, log: LogRecord) -> LogRecord:
        return log

    def scrub_value(self, path: JsonPath, value: Any) -> tuple[Any, list[ScrubbedNote]]:  # pragma: no cover
        return value, []


NOOP_SCRUBBER = NoopScrubber()


class Scrubber(BaseScrubber):
    """Redacts potentially sensitive data."""

    def __init__(
        self,
        patterns: Sequence[str] | None,
        callback: ScrubCallback | None = None,
        *,
        dry_run: bool = False,
        summary_options: ScrubbingSummaryOptions | None = None,
    ):
        # See ScrubbingOptions for more info on these parameters.
        extra_patterns = list(patterns or [])
        patterns = [_DEFAULT_PATTERN, *extra_patterns]
        self._pattern = re.compile('|'.join(patterns), re.IGNORECASE | re.DOTALL)
        self._callback = callback
        self._dry_run = dry_run
        self._summary_recorder = _ScrubbingSummaryRecorder(summary_options) if summary_options is not None else None
        if self._summary_recorder is None:
            self._attribution_patterns: Sequence[tuple[str, re.Pattern[str]]] | None = None
        else:
            # The combined `self._pattern` regex only exposes the winning alternative via
            # `lastindex`-style tricks that user patterns with groups would interfere with.
            # Instead compile each pattern separately in the same order as the combined
            # regex's alternatives so a hit can be attributed to its exact pattern.
            # This is only built when a summary is actually being recorded.
            self._attribution_patterns = [
                (source, re.compile(source, re.IGNORECASE | re.DOTALL))
                for source in (*DEFAULT_PATTERNS, *extra_patterns)
            ]

    def scrub_log(self, log: LogRecord) -> LogRecord:
        span_scrubber = SpanScrubber(self)
        result = span_scrubber.scrub_log(log)
        self._commit_summary(span_scrubber)
        return result

    def scrub_span(self, span: ReadableSpanDict):
        scope = span['instrumentation_scope']
        if scope and scope.name in ['logfire.openai', 'logfire.anthropic']:
            return

        span_scrubber = SpanScrubber(self)
        span_scrubber.scrub_span(span)
        self._commit_summary(span_scrubber)
        if span_scrubber.scrubbed:
            attributes = span['attributes']
            already_scrubbed = cast('str', attributes.get(ATTRIBUTES_SCRUBBED_KEY, '[]'))
            try:
                already_scrubbed = cast('list[ScrubbedNote]', json.loads(already_scrubbed))
            except json.JSONDecodeError:  # pragma: no cover
                already_scrubbed = []
            span['attributes'] = {
                **attributes,
                ATTRIBUTES_SCRUBBED_KEY: json.dumps(already_scrubbed + span_scrubber.scrubbed),
            }

    def scrub_value(self, path: JsonPath, value: Any) -> tuple[Any, list[ScrubbedNote]]:
        span_scrubber = SpanScrubber(self)
        result = span_scrubber.scrub(path, value)
        self._commit_summary(span_scrubber)
        return result, span_scrubber.scrubbed

    def get_scrubbing_summary(self) -> ScrubbingSummary | None:
        if self._summary_recorder is None:
            return None
        return self._summary_recorder.get_summary()

    def _commit_summary(self, span_scrubber: SpanScrubber) -> None:
        # All hits found while processing one span/log/value are committed in a single
        # locked operation so concurrent readers never see a half-processed span.
        if self._summary_recorder is not None:
            self._summary_recorder.commit(span_scrubber.hits, time.time())


class SpanScrubber:
    """Does the actual scrubbing work.

    This class is separate from Scrubber so that it can be instantiated more regularly
    and hold and mutate state about the span being scrubbed, specifically the scrubbed notes.
    """

    def __init__(self, parent: Scrubber):
        self._pattern = parent._pattern  # pyright: ignore[reportPrivateUsage]
        self._callback = parent._callback  # pyright: ignore[reportPrivateUsage]
        self._dry_run = parent._dry_run  # pyright: ignore[reportPrivateUsage]
        self._summary_recorder = parent._summary_recorder  # pyright: ignore[reportPrivateUsage]
        self._attribution_patterns = parent._attribution_patterns  # pyright: ignore[reportPrivateUsage]
        self.scrubbed: list[ScrubbedNote] = []
        self.did_scrub = False
        # Hits are accumulated locally here and committed atomically by the parent Scrubber
        # only after this whole span/log/value has finished being scrubbed.
        self.hits: list[_ScrubbingHit] = []

    def scrub_span(self, span: ReadableSpanDict):
        # We need to use BoundedAttributes because:
        # 1. For events and links, we get an error otherwise:
        #      https://github.com/open-telemetry/opentelemetry-python/issues/3761
        # 2. The callback might return a value that isn't of the type required by OTEL;
        #      BoundedAttributes handles it according to the installed OTEL version.
        # TODO silently throwing away the result is bad, and BoundedAttributes is bad for performance.
        new_attributes = self.scrub(('attributes',), span['attributes'])
        if self.did_scrub:
            span['attributes'] = BoundedAttributes(attributes=new_attributes)

        span['events'] = [
            Event(
                # We don't scrub the event name because in theory it should be a low-cardinality general description,
                # not containing actual data. The same applies to the span name, which just isn't mentioned here.
                name=event.name,
                attributes=BoundedAttributes(attributes=self.scrub_event_attributes(event, i)),
                timestamp=event.timestamp,
            )
            for i, event in enumerate(span['events'])
        ]
        span['links'] = [
            Link(
                context=link.context,
                attributes=BoundedAttributes(attributes=self.scrub(('links', i, 'attributes'), link.attributes)),
            )
            for i, link in enumerate(span['links'])
        ]

    def scrub_log(self, log: LogRecord) -> LogRecord:
        new_attributes: dict[str, Any] | None = self.scrub(('attributes',), log.attributes)
        new_body = self.scrub(('log_body',), log.body)

        if not self.did_scrub:
            return log

        if self.scrubbed:
            new_attributes = new_attributes or {}
            new_attributes[ATTRIBUTES_SCRUBBED_KEY] = json.dumps(self.scrubbed)

        result = copy.copy(log)
        result.attributes = BoundedAttributes(attributes=new_attributes)
        result.body = new_body
        return result

    def scrub_event_attributes(self, event: Event, index: int):
        attributes = event.attributes or {}
        path = ('otel_events', index, 'attributes')
        new_attributes = self.scrub(path, attributes)
        # We used to scrub exception messages here, git blame this line if you want to restore that logic.
        return new_attributes

    def scrub(self, path: JsonPath, value: Any) -> Any:
        """Redacts sensitive data from `value`, recursing into nested sequences and mappings.

        `path` is a list of keys and indices leading to `value` in the span.
        Similar to the truncation code, it should use the field names in the frontend, e.g. `otel_events`.
        """
        if isinstance(value, str):
            if match := self._pattern.search(value):
                if match.span() == (0, len(value)):
                    # If the *whole* string matches, e.g. the value is literally 'password' and nothing more,
                    # it's considered safe.
                    return value
                try:
                    parsed_value = json.loads(value)
                except json.JSONDecodeError:
                    return self._redact(ScrubMatch(path, value, match))
                else:
                    # Recurse into the parsed value (which records any hits). In dry-run mode
                    # return the original string verbatim rather than re-serializing it, so
                    # that the output is unchanged.
                    scrubbed_value = self.scrub(path, parsed_value)
                    if self._dry_run:
                        return value
                    return json.dumps(scrubbed_value)
        elif isinstance(value, Sequence):
            return [self.scrub(path + (i,), x) for i, x in enumerate(cast('Sequence[Any]', value))]
        elif isinstance(value, Mapping):
            result: dict[str, Any] = {}
            for k, v in cast('Mapping[str, Any]', value).items():
                if k in BaseScrubber.SAFE_KEYS:
                    result[k] = v
                elif match := self._pattern.search(k):
                    redacted = self._redact(ScrubMatch(path + (k,), v, match))
                    if isinstance(redacted, str) and isinstance(v, Sequence) and not isinstance(v, str):
                        redacted = [redacted]
                    result[k] = redacted
                else:
                    result[k] = self.scrub(path + (k,), v)
            return result
        return value

    def _redact(self, match: ScrubMatch) -> Any:
        if self._callback is not None:
            callback_result: Any = self._callback(match)
        else:
            callback_result = None
        if callback_result is not None:
            # The callback overrode the default redaction.
            new_value = callback_result
            redacted = callback_result is not match.value
        else:
            # No callback, or the callback returned `None` to accept the default redaction.
            new_value = f'[Scrubbed due to {match.pattern_match.group(0)!r}]'
            redacted = True

        self._record_hit(match, redacted)

        if self._dry_run:
            # Dry run: the hit has been recorded above, but the value is left untouched.
            return match.value
        if not redacted:
            # The callback returned the original value, so nothing changes.
            return match.value

        self.did_scrub = True
        if callback_result is None:
            self.scrubbed.append(ScrubbedNote(path=match.path, matched_substring=match.pattern_match.group(0)))
        return new_value

    def _record_hit(self, match: ScrubMatch, redacted: bool) -> None:
        if self._summary_recorder is None:
            return
        attribution_patterns = self._attribution_patterns
        assert attribution_patterns is not None
        # Python's `re` alternation is ordered and leftmost-first: at the match position,
        # the first alternative that matches there is the one the combined regex used.
        matched_string = match.pattern_match.string
        start = match.pattern_match.start()
        for source, regex in attribution_patterns:
            if regex.match(matched_string, start) is not None:
                pattern = source
                break
        else:  # pragma: no cover
            pattern = match.pattern_match.group(0)
        self.hits.append(_ScrubbingHit(pattern=pattern, path=match.path, redacted=redacted))


class MessageValueCleaner:
    """Scrubs and truncates formatted field values to be included in the message attribute.

    Use to construct the message for a single span, e.g:

        cleaner = MessageValueCleaner(scrubber, check_keys=...)
        message_parts = [cleaner.clean_value(field_name, str(value)) for field_name, value in fields]
        message = <construct from message parts>
        attributes = {**other_attributes, **cleaner.extra_attrs(), ATTRIBUTES_MESSAGE_KEY: message}

    check_keys determines whether the key should be accounted for in scrubbing.
    Set to False if the user explicitly provided the key, e.g. `logfire.info(f'... {password} ...')`
    means that the password is clearly expected to be logged.
    The password will therefore not be scrubbed here and will appear in the message.
    However it may still be scrubbed out of the attributes, just because that process is independent.
    """

    def __init__(self, scrubber: BaseScrubber, *, check_keys: bool):
        self.scrubber = scrubber
        self.scrubbed: list[ScrubbedNote] = []
        self.check_keys = check_keys

    def clean_value(self, field_name: str, value: str) -> str:
        # Scrub before truncating so that the scrubber can see the full value.
        # For example, if the value contains 'password=123' and 'password' is replaced by '...'
        # because of truncation, then that leaves '=123' in the message, which is not good.
        if field_name not in self.scrubber.SAFE_KEYS:
            if self.check_keys:
                # Scrubbing a dict with only one key is a simple way to check that key during the scrubbing.
                scrubbed_value, scrubbed_notes = self.scrubber.scrub_value(('message',), {field_name: value})
                value = scrubbed_value[field_name]
            else:
                # Whereas having the key in the path doesn't affect the scrubbing result,
                # so this only looks at `value` itself.
                value, scrubbed_notes = self.scrubber.scrub_value(('message', field_name), value)
            self.scrubbed.extend(scrubbed_notes)
        return self.truncate(value)

    def truncate(self, value: str) -> str:
        return truncate_string(value, max_length=MESSAGE_FORMATTED_VALUE_LENGTH_LIMIT)

    def extra_attrs(self) -> dict[str, Any]:
        if self.scrubbed:
            return {ATTRIBUTES_SCRUBBED_KEY: json.dumps(self.scrubbed)}
        return {}
