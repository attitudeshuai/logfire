from __future__ import annotations

import gc
import logging
import subprocess
import sys
import weakref
from collections import OrderedDict
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from opentelemetry import trace

import logfire
from logfire._internal.config import GLOBAL_CONFIG
from logfire._internal.open_spans import (
    ExitOpenSpansSummary,
    ForceClosedSpanInfo,
    OpenSpansMonitor,
    end_open_spans_at_exit,
    format_exit_summary,
)
from logfire._internal.tracer import OPEN_SPANS
from logfire.exceptions import LogfireConfigError
from logfire.testing import TestExporter

ONE_SECOND = 1_000_000_000


def start_open_span(name: str) -> logfire.LogfireSpan:
    """Start a span without leaving it attached as the current span, keeping it open."""
    span = logfire.span(name)
    span.__enter__()
    # Detach the context token so that the span isn't left as the current span,
    # but the span itself stays open on purpose.
    span._detach()  # pyright: ignore[reportPrivateUsage]
    return span


def configure_open_spans(
    config_kwargs: dict[str, Any], *, max_age: timedelta, max_entries: int = 100
) -> OpenSpansMonitor:
    logfire.configure(
        **config_kwargs,
        open_spans=logfire.OpenSpansOptions(max_age=max_age, max_entries=max_entries),
    )
    monitor = GLOBAL_CONFIG._open_spans_monitor  # pyright: ignore[reportPrivateUsage]
    assert isinstance(monitor, OpenSpansMonitor)
    return monitor


def test_disabled_by_default(exporter: TestExporter, config_kwargs: dict[str, Any]) -> None:
    logfire.configure(**config_kwargs)
    assert GLOBAL_CONFIG._open_spans_monitor is None  # pyright: ignore[reportPrivateUsage]
    assert logfire.get_open_spans() == logfire.OpenSpansSnapshot(entries=(), dropped_count=0)

    with logfire.span('normal span'):
        pass

    # Spans are started, ended and exported exactly as before.
    [exported] = exporter.exported_spans_as_dict()
    assert exported['name'] == 'normal span'


def test_spans_younger_than_max_age_are_not_listed(config_kwargs: dict[str, Any], time_generator: Any) -> None:
    logfire.configure(**config_kwargs, open_spans=logfire.OpenSpansOptions(max_age=timedelta(seconds=5)))

    span = start_open_span('fresh span')
    start_ns = time_generator.ns_time

    snapshot = logfire.get_open_spans()
    now_ns = time_generator.ns_time
    assert now_ns - start_ns < 5 * ONE_SECOND
    assert snapshot == logfire.OpenSpansSnapshot(entries=(), dropped_count=0)

    span.__exit__(None, None, None)


def test_spans_older_than_max_age_are_listed(config_kwargs: dict[str, Any], time_generator: Any) -> None:
    logfire.configure(**config_kwargs, open_spans=logfire.OpenSpansOptions(max_age=timedelta(seconds=5)))

    span = start_open_span('slow span')
    start_ns = time_generator.ns_time

    time_generator.ns_time += 10 * ONE_SECOND
    snapshot = logfire.get_open_spans()

    assert snapshot.dropped_count == 0
    [entry] = snapshot.entries
    assert entry.name == 'slow span'
    assert entry.age == timedelta(seconds=(time_generator.ns_time - start_ns) / 1e9)
    assert entry.trace_id == trace.format_trace_id(1)
    assert entry.span_id == trace.format_span_id(1)
    assert entry.code_filepath is not None
    assert entry.code_filepath.endswith('test_open_spans.py')
    assert entry.code_function == 'start_open_span'
    assert isinstance(entry.code_lineno, int)

    span.__exit__(None, None, None)


def test_spans_without_code_attributes_are_listed(config_kwargs: dict[str, Any], time_generator: Any) -> None:
    # Spans created directly through an OpenTelemetry tracer (e.g. third-party instrumentation)
    # don't carry code location attributes.
    logfire.configure(**config_kwargs, open_spans=logfire.OpenSpansOptions(max_age=timedelta(seconds=5)))

    raw_span = trace.get_tracer('some-third-party-library').start_span('raw otel span')
    time_generator.ns_time += 10 * ONE_SECOND
    [entry] = logfire.get_open_spans().entries
    assert entry.name == 'raw otel span'
    assert entry.code_filepath is None
    assert entry.code_lineno is None
    assert entry.code_function is None

    raw_span.end()


def test_ending_a_span_removes_its_entry(config_kwargs: dict[str, Any], time_generator: Any) -> None:
    logfire.configure(**config_kwargs, open_spans=logfire.OpenSpansOptions(max_age=timedelta(seconds=5)))

    span = start_open_span('span that ends')
    time_generator.ns_time += 10 * ONE_SECOND
    assert len(logfire.get_open_spans().entries) == 1

    span.__exit__(None, None, None)
    assert logfire.get_open_spans().entries == ()
    assert OPEN_SPANS == {}


def test_capacity_evicts_oldest_entries_and_counts(config_kwargs: dict[str, Any], time_generator: Any) -> None:
    logfire.configure(**config_kwargs, open_spans=logfire.OpenSpansOptions(max_age=timedelta(seconds=5), max_entries=2))

    first = start_open_span('first')
    second = start_open_span('second')
    third = start_open_span('third')

    time_generator.ns_time += 10 * ONE_SECOND
    snapshot = logfire.get_open_spans()
    assert [entry.name for entry in snapshot.entries] == ['second', 'third']
    assert snapshot.dropped_count == 1

    # Ending a listed span frees a slot, but the previously evicted span is not re-registered.
    second.__exit__(None, None, None)
    snapshot = logfire.get_open_spans()
    assert [entry.name for entry in snapshot.entries] == ['third']
    assert snapshot.dropped_count == 1

    # A newly stale span can still take the free slot.
    fourth = start_open_span('fourth')
    time_generator.ns_time += 10 * ONE_SECOND
    snapshot = logfire.get_open_spans()
    assert [entry.name for entry in snapshot.entries] == ['third', 'fourth']
    assert snapshot.dropped_count == 1

    first.__exit__(None, None, None)
    third.__exit__(None, None, None)
    fourth.__exit__(None, None, None)


def test_garbage_collected_spans_are_pruned(config_kwargs: dict[str, Any], time_generator: Any) -> None:
    monitor = configure_open_spans(config_kwargs, max_age=timedelta(seconds=5))

    # A span created directly through OpenTelemetry has no other references once dropped.
    dying = trace.get_tracer('some-third-party-library').start_span('dying span')
    span_ref = weakref.ref(dying)
    time_generator.ns_time += 10 * ONE_SECOND
    assert [entry.name for entry in logfire.get_open_spans().entries] == ['dying span']

    del dying
    gc.collect()
    assert span_ref() is None

    # Both the tracking record and the stale entry are pruned once the span is gone.
    assert logfire.get_open_spans().entries == ()
    assert monitor._tracked == {}  # pyright: ignore[reportPrivateUsage]
    assert monitor._stale == OrderedDict()  # pyright: ignore[reportPrivateUsage]


def test_force_close_logs_summary(
    config_kwargs: dict[str, Any], time_generator: Any, exporter: TestExporter, caplog: Any
) -> None:
    monitor = configure_open_spans(config_kwargs, max_age=timedelta(seconds=5))

    open_spans = [start_open_span('first open span'), start_open_span('second open span')]
    assert len(open_spans) == 2  # keep the span handles alive
    first_start = 1 * ONE_SECOND
    second_start = 2 * ONE_SECOND
    time_generator.ns_time += 10 * ONE_SECOND

    caplog.set_level(logging.WARNING, logger='logfire')
    # force_close calls the timestamp generator once for "now"; ending each span advances it again.
    force_close_now = time_generator.ns_time + ONE_SECOND
    [summary] = end_open_spans_at_exit()

    assert [info.name for info in summary.force_closed] == ['first open span', 'second open span']
    assert [info.duration for info in summary.force_closed] == [
        timedelta(seconds=(force_close_now - first_start) / 1e9),
        timedelta(seconds=(force_close_now - second_start) / 1e9),
    ]
    assert all(
        info.code_filepath and info.code_filepath.endswith('test_open_spans.py') for info in summary.force_closed
    )
    assert summary.dropped_stale_entries_count == 0

    assert OPEN_SPANS == {}
    assert monitor._tracked == {}  # pyright: ignore[reportPrivateUsage]
    exported_names = [span['name'] for span in exporter.exported_spans_as_dict()]
    assert 'first open span' in exported_names
    assert 'second open span' in exported_names

    [record] = caplog.records
    assert 'Logfire force-ended 2 spans that were still open when the process exited:' in record.getMessage()
    assert "'first open span'" in record.getMessage()
    assert format_exit_summary(summary, 100) == record.getMessage()

    # Calling again is harmless once all spans are closed.
    assert end_open_spans_at_exit() == []


def test_force_close_without_monitor_ends_spans_without_logging(
    config_kwargs: dict[str, Any], exporter: TestExporter, caplog: Any
) -> None:
    logfire.configure(**config_kwargs)
    unmonitored = [start_open_span('not monitored')]
    assert len(unmonitored) == 1  # keep the span handle alive

    caplog.set_level(logging.WARNING, logger='logfire')
    assert end_open_spans_at_exit() == []
    assert OPEN_SPANS == {}
    assert caplog.records == []
    assert [span_dict['name'] for span_dict in exporter.exported_spans_as_dict()] == ['not monitored']


def test_force_close_drop_count_in_summary(config_kwargs: dict[str, Any], time_generator: Any, caplog: Any) -> None:
    monitor = configure_open_spans(config_kwargs, max_age=timedelta(seconds=5), max_entries=1)
    evicted = start_open_span('evicted span')
    survivor = start_open_span('survivor span')
    time_generator.ns_time += 10 * ONE_SECOND
    assert logfire.get_open_spans().dropped_count == 1

    caplog.set_level(logging.WARNING, logger='logfire')
    [summary] = end_open_spans_at_exit()
    assert summary.dropped_stale_entries_count == 1
    message = caplog.records[0].getMessage()
    assert '1 long-running span entry was dropped because the list capacity (1) was exceeded.' in message

    evicted.__exit__(None, None, None)
    survivor.__exit__(None, None, None)
    assert monitor._tracked == {}  # pyright: ignore[reportPrivateUsage]


def test_force_close_singular_summary_message(config_kwargs: dict[str, Any], time_generator: Any, caplog: Any) -> None:
    configure_open_spans(config_kwargs, max_age=timedelta(seconds=5))
    lonely = start_open_span('lonely span')
    time_generator.ns_time += 10 * ONE_SECOND

    caplog.set_level(logging.WARNING, logger='logfire')
    [summary] = end_open_spans_at_exit()
    assert len(summary.force_closed) == 1
    message = caplog.records[0].getMessage()
    assert 'Logfire force-ended 1 span that was still open when the process exited:' in message
    lonely.__exit__(None, None, None)


def test_force_close_fallback_without_tracked_record(
    config_kwargs: dict[str, Any], time_generator: Any, caplog: Any
) -> None:
    """Span info can still be built if the start hook somehow didn't record the span."""
    monitor = configure_open_spans(config_kwargs, max_age=timedelta(seconds=5))

    fallback_spans = [
        start_open_span('manual span'),
        trace.get_tracer('some-third-party-library').start_span('raw otel span'),
    ]
    assert len(fallback_spans) == 2  # keep the span handles alive
    # Simulate the tracking records having been lost.
    monitor._tracked.clear()  # pyright: ignore[reportPrivateUsage]

    time_generator.ns_time += 10 * ONE_SECOND
    caplog.set_level(logging.WARNING, logger='logfire')
    [summary] = end_open_spans_at_exit()
    by_name = {info.name: info for info in summary.force_closed}
    assert set(by_name) == {'manual span', 'raw otel span'}
    assert by_name['manual span'].code_filepath is not None
    assert by_name['manual span'].code_function is not None
    assert by_name['raw otel span'].code_filepath is None
    assert all(info.duration >= timedelta(0) for info in summary.force_closed)


def test_force_close_fallback_for_sampled_out_span(
    config_kwargs: dict[str, Any], time_generator: Any, caplog: Any
) -> None:
    """A sampled-out (non-recording) span without a tracked record falls back to placeholder info."""
    logfire.configure(
        **config_kwargs,
        sampling=logfire.SamplingOptions(head=0.0),
        open_spans=logfire.OpenSpansOptions(max_age=timedelta(seconds=5)),
    )
    monitor = GLOBAL_CONFIG._open_spans_monitor  # pyright: ignore[reportPrivateUsage]
    assert monitor is not None
    span = start_open_span('sampled out span')
    monitor._tracked.clear()  # pyright: ignore[reportPrivateUsage]

    time_generator.ns_time += 10 * ONE_SECOND
    caplog.set_level(logging.WARNING, logger='logfire')
    [summary] = end_open_spans_at_exit()
    [info] = summary.force_closed
    assert info.name == '<span>'
    assert info.duration == timedelta(0)
    assert info.code_filepath is None
    # The SDK still allocates a context even for sampled-out non-recording spans.
    assert len(info.trace_id) == 32
    assert len(info.span_id) == 16
    span.__exit__(None, None, None)


def test_force_close_empty_monitor_does_not_log(config_kwargs: dict[str, Any], caplog: Any) -> None:
    monitor = configure_open_spans(config_kwargs, max_age=timedelta(seconds=5))
    caplog.set_level(logging.WARNING, logger='logfire')
    summary = monitor.force_close([])
    assert summary.force_closed == ()
    assert caplog.records == []


@pytest.mark.parametrize(
    'kwargs',
    [
        {'max_age': timedelta(0)},
        {'max_age': timedelta(seconds=-1)},
        {'max_age': timedelta(seconds=1), 'max_entries': 0},
    ],
)
def test_invalid_options_raise(config_kwargs: dict[str, Any], kwargs: dict[str, Any]) -> None:
    with pytest.raises(LogfireConfigError):
        logfire.configure(**config_kwargs, open_spans=logfire.OpenSpansOptions(**kwargs))


def test_options_round_trip_from_dict(config_kwargs: dict[str, Any]) -> None:
    # dataclasses.asdict() (used when sending config to subprocesses) turns the options into a dict.
    options_as_dict = {'max_age': timedelta(seconds=2), 'max_entries': 5}
    logfire.configure(**config_kwargs, open_spans=options_as_dict)  # pyright: ignore[reportArgumentType]
    assert GLOBAL_CONFIG.open_spans == logfire.OpenSpansOptions(max_age=timedelta(seconds=2), max_entries=5)
    assert isinstance(
        GLOBAL_CONFIG._open_spans_monitor,  # pyright: ignore[reportPrivateUsage]
        OpenSpansMonitor,
    )


def test_format_exit_summary() -> None:
    info_without_origin = ForceClosedSpanInfo(
        name='no origin',
        duration=timedelta(seconds=1),
        trace_id='a' * 32,
        span_id='b' * 16,
        code_filepath=None,
        code_lineno=None,
        code_function=None,
    )
    info_filepath_only = ForceClosedSpanInfo(
        name='module level',
        duration=timedelta(seconds=2),
        trace_id='c' * 32,
        span_id='d' * 16,
        code_filepath='main.py',
        code_lineno=4,
        code_function=None,
    )
    message = format_exit_summary(
        ExitOpenSpansSummary(
            force_closed=(info_without_origin, info_filepath_only),
            dropped_stale_entries_count=0,
        ),
        max_entries=100,
    )
    assert "'no origin' open for 0:00:01 (trace id " + 'a' * 32 + ')' in message
    assert "'module level' open for 0:00:02 (trace id " + 'c' * 32 + ', started at main.py:4)' in message
    assert ' in ' not in message


def test_exit_open_spans_subprocess_summary() -> None:
    script_path = Path(__file__).parent / 'import_used_for_tests' / 'open_spans_at_shutdown.py'
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        cwd=Path(__file__).parents[1],
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert 'open span at shutdown' in result.stdout
    assert 'Logfire force-ended 1 span that was still open when the process exited:' in result.stderr
    assert "'open span at shutdown'" in result.stderr
    assert 'open for 0:00:0' in result.stderr
    assert 'started at' in result.stderr
    assert 'open_spans_at_shutdown.py' in result.stderr
    assert 'open_span_in_suspended_generator' in result.stderr
